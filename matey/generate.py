# from copyreg import pickle
import os
from random import seed
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
from collections import OrderedDict
from torchinfo import summary
from .data_utils.datasets import get_data_loader
from .models.avit import build_avit
from .models.svit import build_svit
from .models.vit import build_vit
from .models.turbt import build_turbt
from .models.diffusion_model import build_diffusion_model
from .utils.distributed_utils import determine_turt_levels, get_sequence_parallel_group
from .utils.forward_options import ForwardOptionsBase
from .utils.training_utils import EDMLoss
import json
import numpy as np
import pickle
import copy

class Generator:
    def __init__(self, params, global_rank, local_rank, device):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.log_to_screen = self.params.log_to_screen
        # Basic setup
        self.diffusion_loss = EDMLoss() 
        self.startEpoch = 0
        self.epoch = 0
        self.mp_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.half

        self.cond_diffusion = getattr(self.params, "cond_diffusion", False)
        self.profiling = self.params.profiling if hasattr(self.params, "profiling") else False

        self.output_dir = self.params.output_dir
         #define sequence parallel groups and local group info
        if hasattr(self.params, "sp_groupsize"):
            self.current_group, self.group_id, self.num_sequence_parallel_groups = get_sequence_parallel_group(sequence_parallel_groupsize=self.params.sp_groupsize)
        else:
            self.current_group, self.group_id, self.num_sequence_parallel_groups = get_sequence_parallel_group(num_sequence_parallel_groups=self.params.num_sequence_parallel_groups if hasattr(self.params, "num_sequence_parallel_groups") else self.world_size)

        self.group_rank = dist.get_rank(self.current_group)
        self.group_size = dist.get_world_size(self.current_group)

        self.initialize_data()
        #print(f"Initializing model on rank {self.global_rank}")

        #checking input_states value
        labels_total=[self.train_dataset.subset_dict[dset] for dset in self.train_dataset.subset_dict]
        labels_total = [item  for sublist in labels_total for item in sublist]
        if self.params.n_states<max(labels_total)+1:
            print(f"Warning, reserved n_states {self.params.n_states} is too small for datasets, set it to {max(labels_total)+1} instead")
            self.params.n_states = max(labels_total)+1

        self.initialize_model()
        print("Loading checkpoint %s"%self.params.checkpoint_path)
        self.restore_checkpoint(self.params.checkpoint_path)

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(' '.join([str(t) for t in text]), flush =True)
    
    def initialize_data(self):
        """
        data_rank=None
        num_replicas=None
        if  hasattr(self.params, "sp_groupsize") or hasattr(self.params, "num_sequence_parallel_groups"):
            data_rank=True
        if data_rank:
            parallel_group_size = self.group_size
            in_rank = self.global_rank//parallel_group_size #SP group ID
            group_rank = self.global_rank%parallel_group_size #local rank inside each SP group
            num_replicas = len(self.sequence_parallel_groups)
        else:
            in_rank = self.global_rank
            parallel_group_size=self.group_size
            group_rank=0
        """
        #print("Pei debugging", self.group_size, group_rank, in_rank, parallel_group_size, num_replicas, flush=True)
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}", flush=True)
        #print(f"Pei debugging trainpy, {self.group_size}, {self.global_rank}, {len(self.sequence_parallel_groups)}, {self.sequence_parallel_groups}", flush=True)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_data_paths,
                          dist.is_initialized(), split='train', train_offset=self.params.embedding_offset,
                          group_size= self.group_size, global_rank= self.global_rank, num_sp_groups=self.num_sequence_parallel_groups)
                          
        self.valid_data_loader, self.valid_dataset, self.val_sampler = get_data_loader(self.params, self.params.valid_data_paths,
                          dist.is_initialized(), split='val', 
                          group_size= self.group_size, global_rank= self.global_rank, num_sp_groups=self.num_sequence_parallel_groups)
        self.single_print("self.train_data_loader:",  len(self.train_data_loader), "valid_data_loader:", len(self.valid_data_loader))
        
    def initialize_model(self):
        if self.params.diffusion:
            self.model = build_diffusion_model(self.params).to(self.device)
        else:
            raise NotImplementedError("Only diffusion model generation is implemented currently. Please set params.diffusion to True.")


        if dist.is_initialized() and self.params.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                            output_device=[self.local_rank], find_unused_parameters=True)
        
        self.single_print(f'Model parameter count: {sum([p.numel() for p in self.model.parameters()])}')

    def restore_checkpoint(self, checkpoint_path):
        print(f"restoring checkpoint........{checkpoint_path}")
        """
        print("before pei debug scheduler")
        current_lrs = self.scheduler.get_last_lr()
        print("before Current LR(s):", current_lrs)
        for i, pg in enumerate(self.optimizer.param_groups):
            print(f"Param group {i} LR: {pg['lr']}")
        import pprint
        state = self.scheduler.state_dict()
        pprint.pprint(state)
        """

        
        """ Load model/opt from path """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank) if torch.cuda.is_available() else torch.device('cpu'),  weights_only=False)
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
        else:
            model_state = checkpoint
        try: 
            self.model.load_state_dict(model_state)
        except: 
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(model_state)
            else:
                new_state_dict = OrderedDict()
                for key, val in model_state.items():
                    name = key[7:]
                    new_state_dict[name] = val
                self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
 
    def generate(self, seed=None, num_samples=1, batch_list=[0]):
        if self.global_rank == 0:
            summary(self.model)
        self.single_print("Starting Generation Loop...")

        if seed is not None:
            seed_value = seed
            torch.manual_seed(seed_value)

        for batch_idx in batch_list:
            data_iter = iter(self.valid_data_loader)
            
            data = next(data_iter) 
            for _ in range(batch_idx):
                data = next(data_iter) 


            if "graph" in data:
                graphdata = data["graph"].to(self.device)
                tar = graphdata.y #[nnodes, C_tar] 
                leadtime = graphdata.leadtime #[nnodes, 1]
                dset_index, field_labels, field_labels_out, bcs = map(lambda x: x.to(self.device), [data[varname] for varname in ["dset_idx", "field_labels", "field_labels_out", "bcs"]])
            else: 
                inp, dset_index, field_labels, bcs, tar, leadtime = map(lambda x: x.to(self.device), [data[varname] for varname in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
                field_labels_out = field_labels
            supportdata = True if hasattr(self.params, 'supportdata') else False
            if supportdata:
                cond_input = data["cond_input"].to(self.device)
            else:
                cond_input = None

            cond_dict = {}
            try:
                cond_dict["labels"] = data["cond_field_labels"].to(self.device)
                cond_dict["fields"] = rearrange(data["cond_fields"].to(self.device), 'b t c d h w -> t b c d h w')
            except:
                pass

            blockdict = getattr(self.valid_dataset.sub_dsets[dset_index[0]], "blockdict", None)
            dset_type = self.valid_dataset.sub_dsets[dset_index[0]].type
            tkhead_name = self.valid_dataset.sub_dsets[dset_index[0]].tkhead_name            
            imod = self.params.hierarchical["nlevels"]-1 if hasattr(self.params, "hierarchical") else 0
            if "graph" in data:
                isgraph = True
                inp = graphdata
                imod_bottom = imod
            else:
                inp = rearrange(inp.to(self.device), 'b t c d h w -> t b c d h w')
                isgraph = False
                imod_bottom = determine_turt_levels(self.model.module.tokenizer_heads_params[tkhead_name][-1], inp.shape[-3:], imod) if imod>0 else 0

            seq_group = self.current_group if dset_type in self.valid_dataset.DP_dsets else None
                    
            torch.save(inp.cpu(), os.path.join(self.output_dir, f'inputdata_batch_{batch_idx}.pt'))
            self.single_print(f'input saved to {os.path.join(self.output_dir, f"inputdata_batch_{batch_idx}.pt")}')
            torch.save(tar.cpu(), os.path.join(self.output_dir, f'targetdata_batch_{batch_idx}.pt'))
            self.single_print(f'target saved to {os.path.join(self.output_dir, f"targetdata_batch_{batch_idx}.pt")}')

            torch.save(torch.tensor(leadtime).cpu(), os.path.join(self.output_dir, f'leadtimedata_batch_{batch_idx}.pt'))
            self.single_print(f'leadtime saved to {os.path.join(self.output_dir, f"leadtimedata_batch_{batch_idx}.pt")}')

            for sample_idx in range(num_samples):

                
                init_inp = torch.randn(inp.shape).to(self.device)

                
                self.model.eval()

                with torch.no_grad():


                    sigma_min = 0.002
                    sigma_max = 80
                    num_steps = 18
                    rho=7
                    S_churn=0
                    S_min=0
                    S_max=float('inf')
                    S_noise=1

                    step_indices = torch.arange(num_steps, device=self.device)

                    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
                    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

                    # tar = tar.to(self.device)
                    # init_inp = rearrange(init_inp.to(self.device), 'b t c d h w -> t b c d h w')

                    
                    x_next = init_inp * t_steps[0]

                    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
                        x_cur = x_next

                        # Increase noise temporarily.
                        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                        t_hat = t_cur + gamma * t_cur
                        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)


                        opts = ForwardOptionsBase(
                        imod=imod, 
                        imod_bottom=imod_bottom ,
                        tkhead_name=tkhead_name,
                        sequence_parallel_group=seq_group,
                        leadtime=leadtime,
                        blockdict=copy.deepcopy(blockdict),
                        cond_dict=copy.deepcopy(cond_dict),
                        cond_input=cond_input,
                        isgraph=isgraph,
                        field_labels_out= field_labels
                        )
                        if self.cond_diffusion:
                            opts.diffusion_cond = rearrange(inp.to(self.device), 't b c d h w -> b t c d h w') # conditioning on input history
                        # Euler step.
                        # denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
                        # denoised = net(x_hat, t_hat, None)
                        
                        # denoised = self.inference(x_hat, t_hat, field_labels, bcs, opts)
                        denoised = self.model(x_hat, t_hat.repeat(x_hat.shape[1]), field_labels, bcs, opts)

                        # print(f"denoised shape: {denoised.shape}, x_hat shape: {x_hat.shape}, t_hat shape: {t_hat.shape}")
                        d_cur = (x_hat - denoised) / t_hat
                        x_next = x_hat + (t_next - t_hat) * d_cur

                        # print(f"x_next shape: {x_next.shape}, d_cur shape: {d_cur.shape}, t_next shape: {t_next.shape}, t_hat shape: {t_hat.shape}")
                        # Apply 2nd order correction.
                        if i < num_steps - 1:

                            opts = ForwardOptionsBase(
                            imod=imod, 
                            imod_bottom=imod_bottom ,
                            tkhead_name=tkhead_name,
                            sequence_parallel_group=seq_group,
                            leadtime=leadtime,
                            blockdict=copy.deepcopy(blockdict),
                            cond_dict=copy.deepcopy(cond_dict),
                            cond_input=cond_input,
                            isgraph=isgraph,
                            field_labels_out= field_labels
                            )
                            if self.cond_diffusion:
                                opts.diffusion_cond = rearrange(inp.to(self.device), 't b c d h w -> b t c d h w') # conditioning on input history
                            # denoised = net(x_next, t_next, class_labels).to(torch.float64)
                            # denoised = net(x_next, t_next, None)
                            # denoised = self.inference(x_next, t_next, field_labels, bcs, opts)
                            denoised = self.model(x_next, t_next.repeat(x_next.shape[1]), field_labels, bcs, opts)

                            d_prime = (x_next - denoised) / t_next
                            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                        torch.save(x_next.cpu(), os.path.join(self.output_dir, f'generation_step_{i}_batch_{batch_idx}_sample{sample_idx}.pt'))

                    output = x_next    
                    
                    ###full resolution###
                    # residuals = output - tar
                    # torch.save(output.cpu(), os.path.join('Demo_Diffusion_CIFAR10_finemodel/basic_config/demo_diffusion/training_checkpoints/', 'generation_output.pt'))
                    # self.single_print(f'Generation output saved to {"Demo_Diffusion_CIFAR10_finemodel/basic_config/demo_diffusion/training_checkpoints/generation_output.pt"}')
                    torch.save(output.cpu(), os.path.join(self.output_dir, f'generation_output_batch_{batch_idx}_sample{sample_idx}.pt'))
                    self.single_print(f'Generation output saved to {os.path.join(self.output_dir, f"generation_output_batch_{batch_idx}_sample{sample_idx}.pt")}')


                torch.cuda.empty_cache()



    def autoregressive_generate(self, seed=None, num_samples=1, num_steps=10):
        if self.global_rank == 0:
            summary(self.model)
        self.single_print("Starting Generation Loop...")

        if seed is not None:
            seed_value = seed
            torch.manual_seed(seed_value)

        data_iter = iter(self.valid_data_loader)
        for step_idx in range(num_steps):
            
            
            if step_idx==0:
                data = next(data_iter)             

                if "graph" in data:
                    graphdata = data["graph"].to(self.device)
                    tar = graphdata.y #[nnodes, C_tar] 
                    leadtime = graphdata.leadtime #[nnodes, 1]
                    dset_index, field_labels, field_labels_out, bcs = map(lambda x: x.to(self.device), [data[varname] for varname in ["dset_idx", "field_labels", "field_labels_out", "bcs"]])
                else: 
                    inp, dset_index, field_labels, bcs, tar, leadtime = map(lambda x: x.to(self.device), [data[varname] for varname in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
                    field_labels_out = field_labels
                supportdata = True if hasattr(self.params, 'supportdata') else False
                if supportdata:
                    cond_input = data["cond_input"].to(self.device)
                else:
                    cond_input = None

                cond_dict = {}
                try:
                    cond_dict["labels"] = data["cond_field_labels"].to(self.device)
                    cond_dict["fields"] = rearrange(data["cond_fields"].to(self.device), 'b t c d h w -> t b c d h w')
                except:
                    pass

                blockdict = getattr(self.valid_dataset.sub_dsets[dset_index[0]], "blockdict", None)
                dset_type = self.valid_dataset.sub_dsets[dset_index[0]].type
                tkhead_name = self.valid_dataset.sub_dsets[dset_index[0]].tkhead_name            
                imod = self.params.hierarchical["nlevels"]-1 if hasattr(self.params, "hierarchical") else 0
                if "graph" in data:
                    isgraph = True
                    inp = graphdata
                    imod_bottom = imod
                else:
                    inp = rearrange(inp.to(self.device), 'b t c d h w -> t b c d h w')
                    isgraph = False
                    imod_bottom = determine_turt_levels(self.model.module.tokenizer_heads_params[tkhead_name][-1], inp.shape[-3:], imod) if imod>0 else 0

                seq_group = self.current_group if dset_type in self.valid_dataset.DP_dsets else None
                    
            torch.save(inp.cpu(), os.path.join(self.output_dir, f'inputdata_batch_{batch_idx}.pt'))
            self.single_print(f'input saved to {os.path.join(self.output_dir, f"inputdata_batch_{batch_idx}.pt")}')
            torch.save(tar.cpu(), os.path.join(self.output_dir, f'targetdata_batch_{batch_idx}.pt'))
            self.single_print(f'target saved to {os.path.join(self.output_dir, f"targetdata_batch_{batch_idx}.pt")}')

            torch.save(torch.tensor(leadtime).cpu(), os.path.join(self.output_dir, f'leadtimedata_batch_{batch_idx}.pt'))
            self.single_print(f'leadtime saved to {os.path.join(self.output_dir, f"leadtimedata_batch_{batch_idx}.pt")}')

            for sample_idx in range(num_samples):

                
                init_inp = torch.randn(inp.shape).to(self.device)

                
                self.model.eval()

                with torch.no_grad():


                    sigma_min = 0.002
                    sigma_max = 80
                    num_steps = 18
                    rho=7
                    S_churn=0
                    S_min=0
                    S_max=float('inf')
                    S_noise=1

                    step_indices = torch.arange(num_steps, device=self.device)

                    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
                    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

                    # tar = tar.to(self.device)
                    # init_inp = rearrange(init_inp.to(self.device), 'b t c d h w -> t b c d h w')

                    
                    x_next = init_inp * t_steps[0]

                    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
                        x_cur = x_next

                        # Increase noise temporarily.
                        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                        t_hat = t_cur + gamma * t_cur
                        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)


                        opts = ForwardOptionsBase(
                        imod=imod, 
                        imod_bottom=imod_bottom ,
                        tkhead_name=tkhead_name,
                        sequence_parallel_group=seq_group,
                        leadtime=leadtime,
                        blockdict=copy.deepcopy(blockdict),
                        cond_dict=copy.deepcopy(cond_dict),
                        cond_input=cond_input,
                        isgraph=isgraph,
                        field_labels_out= field_labels
                        )
                        if self.cond_diffusion:
                            opts.diffusion_cond = rearrange(inp.to(self.device), 't b c d h w -> b t c d h w') # conditioning on input history
                        # Euler step.
                        # denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
                        # denoised = net(x_hat, t_hat, None)
                        
                        # denoised = self.inference(x_hat, t_hat, field_labels, bcs, opts)
                        denoised = self.model(x_hat, t_hat.repeat(x_hat.shape[1]), field_labels, bcs, opts)

                        # print(f"denoised shape: {denoised.shape}, x_hat shape: {x_hat.shape}, t_hat shape: {t_hat.shape}")
                        d_cur = (x_hat - denoised) / t_hat
                        x_next = x_hat + (t_next - t_hat) * d_cur

                        # print(f"x_next shape: {x_next.shape}, d_cur shape: {d_cur.shape}, t_next shape: {t_next.shape}, t_hat shape: {t_hat.shape}")
                        # Apply 2nd order correction.
                        if i < num_steps - 1:

                            opts = ForwardOptionsBase(
                            imod=imod, 
                            imod_bottom=imod_bottom ,
                            tkhead_name=tkhead_name,
                            sequence_parallel_group=seq_group,
                            leadtime=leadtime,
                            blockdict=copy.deepcopy(blockdict),
                            cond_dict=copy.deepcopy(cond_dict),
                            cond_input=cond_input,
                            isgraph=isgraph,
                            field_labels_out= field_labels
                            )
                            if self.cond_diffusion:
                                opts.diffusion_cond = rearrange(inp.to(self.device), 't b c d h w -> b t c d h w') # conditioning on input history
                            # denoised = net(x_next, t_next, class_labels).to(torch.float64)
                            # denoised = net(x_next, t_next, None)
                            # denoised = self.inference(x_next, t_next, field_labels, bcs, opts)
                            denoised = self.model(x_next, t_next.repeat(x_next.shape[1]), field_labels, bcs, opts)

                            d_prime = (x_next - denoised) / t_next
                            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                        # torch.save(x_next.cpu(), os.path.join(self.output_dir, f'generation_output_step_{i}_batch_{batch_idx}_sample{sample_idx}.pt'))

                    output = x_next    
                    
                    ###full resolution###
                    # residuals = output - tar
                    # torch.save(output.cpu(), os.path.join('Demo_Diffusion_CIFAR10_finemodel/basic_config/demo_diffusion/training_checkpoints/', 'generation_output.pt'))
                    # self.single_print(f'Generation output saved to {"Demo_Diffusion_CIFAR10_finemodel/basic_config/demo_diffusion/training_checkpoints/generation_output.pt"}')
                    torch.save(output.cpu(), os.path.join(self.output_dir, f'generation_output_batch_{batch_idx}_sample{sample_idx}.pt'))
                    self.single_print(f'Generation output saved to {os.path.join(self.output_dir, f"generation_output_batch_{batch_idx}_sample{sample_idx}.pt")}')


                torch.cuda.empty_cache()


