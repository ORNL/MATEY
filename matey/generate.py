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


class EDMSampler:
    """
    EDM (Karras et al. 2022) 2nd-order Heun sampler.

    Encapsulates the denoising schedule and loop so that alternative samplers
    can be dropped in by subclassing and overriding `sample`.
    """
    def __init__(self, sigma_min=0.002, sigma_max=80, n_steps=18, rho=7,
                 S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_steps = n_steps
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    def _t_steps(self, device):
        idx = torch.arange(self.n_steps, device=device)
        t = (self.sigma_max ** (1 / self.rho)
             + idx / (self.n_steps - 1)
             * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        return torch.cat([t, torch.zeros_like(t[:1])])  # t_N = 0

    def _make_opts(self, opts_kwargs, blockdict, cond_dict_b):
        return ForwardOptionsBase(**{**opts_kwargs,
                                     'blockdict': copy.deepcopy(blockdict),
                                     'cond_dict': copy.deepcopy(cond_dict_b)})

    def sample(self, model, inp_b, num_samples, field_labels_b, bcs_b,
               opts_kwargs, blockdict, cond_dict_b,
               cond_diffusion=False, output_dir=None, batch_idx=None):
        """
        Run the denoising loop over a batched input.

        inp_b : [T, B*num_samples, C, D, H, W]
        Returns output grouped as [num_samples, T, B, C, D, H, W].
        """
        t_steps = self._t_steps(inp_b.device)
        x_next = torch.randn_like(inp_b) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = (min(self.S_churn / self.n_steps, np.sqrt(2) - 1)
                     if self.S_min <= t_cur <= self.S_max else 0)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step
            opts = self._make_opts(opts_kwargs, blockdict, cond_dict_b)
            if cond_diffusion:
                opts.diffusion_cond = rearrange(inp_b, 't b c d h w -> b t c d h w')
            denoised = model(x_hat, t_hat.repeat(x_hat.shape[1]), field_labels_b, bcs_b, opts)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # 2nd-order Heun correction
            if i < self.n_steps - 1:
                opts = self._make_opts(opts_kwargs, blockdict, cond_dict_b)
                if cond_diffusion:
                    opts.diffusion_cond = rearrange(inp_b, 't b c d h w -> b t c d h w')
                denoised = model(x_next, t_next.repeat(x_next.shape[1]), field_labels_b, bcs_b, opts)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            if output_dir is not None:
                grouped = rearrange(x_next, 't (b s) c d h w -> s t b c d h w', s=num_samples)
                torch.save(grouped.cpu(), os.path.join(output_dir, f'generation_step_{i}_batch_{batch_idx}.pt'))

        return rearrange(x_next, 't (b s) c d h w -> s t b c d h w', s=num_samples)


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
        self.sampler = EDMSampler()
        self.startEpoch = 0
        self.epoch = 0
        self.mp_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.half

        self.diffusion_config = getattr(self.params, 'diffusion_config', None) or {}
        self.cond_diffusion = self.diffusion_config.get("cond_diffusion", False)
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
        if self.diffusion_config.get("diffusion", False):
            self.model = build_diffusion_model(self.params).to(self.device)
        else:
            raise NotImplementedError("Only diffusion model generation is implemented currently. Please set params.diffusion_config.diffusion to True.")


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

            self.model.eval()

            with torch.no_grad():
                # Expand batch dimension to run all num_samples in one forward pass.
                # inp: [T, B, C, D, H, W] -> [T, B*num_samples, C, D, H, W]
                inp_b = inp.repeat_interleave(num_samples, dim=1)
                field_labels_b = field_labels.repeat_interleave(num_samples, dim=0)
                bcs_b = bcs.repeat_interleave(num_samples, dim=0)
                leadtime_b = leadtime.repeat_interleave(num_samples, dim=0) if leadtime is not None else None
                cond_input_b = cond_input.repeat_interleave(num_samples, dim=0) if cond_input is not None else None
                cond_dict_b = {}
                if cond_dict:
                    cond_dict_b["labels"] = cond_dict["labels"].repeat_interleave(num_samples, dim=0)
                    cond_dict_b["fields"] = cond_dict["fields"].repeat_interleave(num_samples, dim=1)

                opts_kwargs = dict(
                    imod=imod,
                    imod_bottom=imod_bottom,
                    tkhead_name=tkhead_name,
                    sequence_parallel_group=seq_group,
                    leadtime=leadtime_b,
                    cond_input=cond_input_b,
                    isgraph=isgraph,
                    field_labels_out=field_labels_b,
                )
                output = self.sampler.sample(
                    self.model, inp_b, num_samples, field_labels_b, bcs_b,
                    opts_kwargs=opts_kwargs,
                    blockdict=blockdict,
                    cond_dict_b=cond_dict_b,
                    cond_diffusion=self.cond_diffusion,
                    output_dir=self.output_dir,
                    batch_idx=batch_idx,
                )
                torch.save(output.cpu(), os.path.join(self.output_dir, f'generation_output_batch_{batch_idx}.pt'))
                self.single_print(f'Generation output saved to {os.path.join(self.output_dir, f"generation_output_batch_{batch_idx}.pt")}')

            torch.cuda.empty_cache()



    def autoregressive_generate(self, seed=None, num_samples=1, num_steps=10):
    ### FIXME: UNTESTED ###
        if self.global_rank == 0:
            summary(self.model)
        self.single_print("Starting Autoregressive Generation Loop...")

        if seed is not None:
            torch.manual_seed(seed)

        # Load one batch of data
        data_iter = iter(self.valid_data_loader)
        data = next(data_iter)

        if "graph" in data:
            graphdata = data["graph"].to(self.device)
            tar = graphdata.y
            leadtime = graphdata.leadtime
            dset_index, field_labels, field_labels_out, bcs = map(lambda x: x.to(self.device), [data[varname] for varname in ["dset_idx", "field_labels", "field_labels_out", "bcs"]])
        else:
            inp, dset_index, field_labels, bcs, tar, leadtime = map(lambda x: x.to(self.device), [data[varname] for varname in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
            field_labels_out = field_labels
        supportdata = True if hasattr(self.params, 'supportdata') else False
        cond_input = data["cond_input"].to(self.device) if supportdata else None

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
            imod_bottom = determine_turt_levels(self.model.module.tokenizer_heads_params[tkhead_name][-1], inp.shape[-3:], imod) if imod > 0 else 0

        seq_group = self.current_group if dset_type in self.valid_dataset.DP_dsets else None

        # Save initial conditioning input and target once
        torch.save(inp.cpu(), os.path.join(self.output_dir, 'inputdata_step0.pt'))
        self.single_print(f'input saved to {os.path.join(self.output_dir, "inputdata_step0.pt")}')
        torch.save(tar.cpu(), os.path.join(self.output_dir, 'targetdata_step0.pt'))
        self.single_print(f'target saved to {os.path.join(self.output_dir, "targetdata_step0.pt")}')
        torch.save(torch.tensor(leadtime).cpu(), os.path.join(self.output_dir, 'leadtimedata_step0.pt'))
        self.single_print(f'leadtime saved to {os.path.join(self.output_dir, "leadtimedata_step0.pt")}')

        self.model.eval()
        inp_cur = inp.clone()  # conditioning that advances each autoregressive step

        for step_idx in range(num_steps):
            with torch.no_grad():
                # Expand batch dimension to run all num_samples in one forward pass.
                # inp_cur: [T, B, C, D, H, W] -> [T, B*num_samples, C, D, H, W]
                inp_b = inp_cur.repeat_interleave(num_samples, dim=1)
                field_labels_b = field_labels.repeat_interleave(num_samples, dim=0)
                bcs_b = bcs.repeat_interleave(num_samples, dim=0)
                leadtime_b = leadtime.repeat_interleave(num_samples, dim=0) if leadtime is not None else None
                cond_input_b = cond_input.repeat_interleave(num_samples, dim=0) if cond_input is not None else None
                cond_dict_b = {}
                if cond_dict:
                    cond_dict_b["labels"] = cond_dict["labels"].repeat_interleave(num_samples, dim=0)
                    cond_dict_b["fields"] = cond_dict["fields"].repeat_interleave(num_samples, dim=1)

                opts_kwargs = dict(
                    imod=imod,
                    imod_bottom=imod_bottom,
                    tkhead_name=tkhead_name,
                    sequence_parallel_group=seq_group,
                    leadtime=leadtime_b,
                    cond_input=cond_input_b,
                    isgraph=isgraph,
                    field_labels_out=field_labels_b,
                )
                # output: [num_samples, T, B, C, D, H, W]
                output = self.sampler.sample(
                    self.model, inp_b, num_samples, field_labels_b, bcs_b,
                    opts_kwargs=opts_kwargs,
                    blockdict=blockdict,
                    cond_dict_b=cond_dict_b,
                    cond_diffusion=self.cond_diffusion,
                )

            torch.save(output.cpu(), os.path.join(self.output_dir, f'generation_output_step_{step_idx}.pt'))
            self.single_print(f'Generation output saved to {os.path.join(self.output_dir, f"generation_output_step_{step_idx}.pt")}')

            # Mean across samples becomes the conditioning for the next autoregressive step
            inp_cur = output.mean(dim=0)  # [T, B, C, D, H, W]
            torch.save(inp_cur.cpu(), os.path.join(self.output_dir, f'generation_mean_step_{step_idx}.pt'))
            self.single_print(f'Step {step_idx} ensemble mean saved to {os.path.join(self.output_dir, f"generation_mean_step_{step_idx}.pt")}')

            torch.cuda.empty_cache()


