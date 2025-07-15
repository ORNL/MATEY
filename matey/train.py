import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
from dadaptation import DAdaptAdam
from collections import OrderedDict
import gc, psutil
from torchinfo import summary
from collections import defaultdict
from .data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT
from .models.avit import build_avit
from .models.svit import build_svit
from .models.vit import build_vit
from .models.turbt import build_turbt
from .utils.logging_utils import Timer, record_function_opt
from .utils.distributed_utils import get_sequence_parallel_group, locate_group, add_weight_decay
from .utils.visualization_utils import checking_data_pred_tar
import json
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_model_state_dict,set_optimizer_state_dict

class Trainer:
    def __init__(self, params, global_rank, local_rank, device):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.log_to_screen = self.params.log_to_screen
        # Basic setup
        self.train_loss = nn.MSELoss()
        self.startEpoch = 0
        self.epoch = 0
        self.mp_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.half

        self.profiling = self.params.profiling if hasattr(self.params, "profiling") else False

        #define sequence parallel groups and local group info
        if hasattr(self.params, "sp_groupsize"):
            self.sequence_parallel_groups, self.group_size = get_sequence_parallel_group(sequence_parallel_groupsize=self.params.sp_groupsize)
        else:
            self.sequence_parallel_groups, self.group_size = get_sequence_parallel_group(num_sequence_parallel_groups=self.params.num_sequence_parallel_groups if hasattr(self.params, "num_sequence_parallel_groups") else self.world_size)

        self.current_group, self.group_rank = locate_group(self.sequence_parallel_groups, self.global_rank)

        self.iters = 0
        self.initialize_data()
        #print(f"Initializing model on rank {self.global_rank}")
        self.refine_resol=None
        if self.params.resuming == False and hasattr(self.params, "startfrom_path"):
            if hasattr(self.params, "tokenizer_heads"):
                raise NotImplementedError("Tokenizer_heads for startfrom_path needs to be implemented")
            if len(self.params.patch_size)>1:
                self.refine_resol=self.params.patch_size
                self.params.patch_size=self.params.patch_size[-1:]
            elif hasattr(self.params, "patch_size_input"):
                self.refine_resol=self.params.patch_size
                self.params.patch_size=self.params.patch_size_input

            self.sts_model=self.params.sts_model
            if self.params.sts_model:
                self.params.sts_model=False
        #checking input_states value
        labels_total=[self.train_dataset.subset_dict[dset] for dset in self.train_dataset.subset_dict]
        labels_total = [item  for sublist in labels_total for item in sublist]
        if self.params.n_states<max(labels_total)+1:
            print(f"Warning, reserved n_states {self.params.n_states} is too small for datasets, set it to {max(labels_total)+1} instead")
            self.params.n_states = max(labels_total)+1

        self.initialize_model()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.timer=Timer(enable_sync=self.params.enable_sync)
        if self.params.resuming:
            if self.params.pretrained:
                #if fresh from pretrainig:
                    #model construction (same with pretrained) --> load pretrained weights --> expand input&output based on append_datasets --> freeze --> update optimizer
                #elif resume from finetuning:
                    #model construction  (same with pretrained) --> expand input and output (same with expanded) and freeeze --> load saved finetuned weights
                self.expand_model_pretraining()
                self.freeze_model_pretraining()
                self.initialize_optimizer()
                self.initialize_scheduler()
            #print("Loading best checkpoint (default) %s"%params.best_checkpoint_path)
            #try:
            #    self.restore_checkpoint(self.params.best_checkpoint_path)
            #except:
            if True:
                print("Instead, loading checkpoint %s"%self.params.checkpoint_path)
                self.restore_checkpoint(self.params.checkpoint_path)
                #self.restore_checkpoint(self.params.best_checkpoint_path)


        if self.params.resuming == False and self.params.pretrained:
            print("Starting from pretrained model at %s"%self.params.pretrained_ckpt_path)
            if os.path.exists(self.params.pretrained_ckpt_path):
                self.restore_checkpoint(self.params.pretrained_ckpt_path)
            elif self.params.pretrained_ckpt_path =="INIT":
                pass
            else:
                raise ValueError("%s not found" %self.params.pretrained_ckpt_path)
            self.expand_model_pretraining()
            self.freeze_model_pretraining()
            self.iters = 0
            self.startEpoch = 0
            self.initialize_optimizer()
            self.initialize_scheduler()

        if self.params.resuming == False and hasattr(self.params, "startfrom_path"):
            self.restore_checkpoint(self.params.startfrom_path)
            self.expand_model_pretraining_convheads()
            if self.sts_model and not self.params.sts_model:
                self.expand_model_pretraining_sts_model()
            self.startEpoch = 0
            self.initialize_optimizer()
            self.initialize_scheduler()
            self.single_print(f'After loading and expanding, model parameter count: {sum([p.numel() for p in self.model.parameters()])}')

            if self.global_rank == 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data.numel())

        # as initialize_scheduler needs self.startEpoch from self.restore_checkpoint
        #self.initialize_scheduler(self.params)

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(' '.join([str(t) for t in text]), flush =True)
    
    def check_memory(self, message):
        if self.global_rank == 0:
            print("Memory summary: %s, CUDA %f GB; RAM %f GB, %f percentage"%(message, torch.cuda.memory_allocated()/ 1024**3, psutil.virtual_memory().used/1024**3, psutil.virtual_memory().percent))
    
    def initialize_data(self):
        data_rank=None
        num_replicas=None
        if  hasattr(self.params, "sp_groupsize") or hasattr(self.params, "num_sequence_parallel_groups"):
            data_rank=True
        if self.params.tie_batches:
            in_rank = 0
            parallel_group_size=1
            group_rank=0
        elif data_rank:
            parallel_group_size = self.group_size
            in_rank = self.global_rank//parallel_group_size #SP group ID
            group_rank = self.global_rank%parallel_group_size #local rank inside each SP group
            num_replicas = len(self.sequence_parallel_groups)
        else:
            in_rank = self.global_rank
            parallel_group_size=self.group_size
            group_rank=0
        #print("Pei debugging", self.group_size, group_rank, in_rank, parallel_group_size, num_replicas, flush=True)
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}", flush=True)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_data_paths,
                          dist.is_initialized(), split='train', rank=in_rank, group_rank=group_rank, group_size=parallel_group_size, 
                          num_replicas=num_replicas, train_offset=self.params.embedding_offset)
        self.valid_data_loader, self.valid_dataset, self.val_sampler = get_data_loader(self.params, self.params.valid_data_paths,
                          dist.is_initialized(), split='val',   rank=in_rank, group_rank=group_rank, group_size=parallel_group_size,
                          num_replicas=num_replicas)
        
        self.single_print("self.train_data_loader:",  len(self.train_data_loader), "valid_data_loader:", len(self.valid_data_loader))
        if dist.is_initialized():
            self.train_sampler.set_epoch(0)
            self.val_sampler.set_epoch(0)
    
    def initialize_model(self):
        if self.params.model_type == 'avit':
            self.model = build_avit(self.params).to(self.device)
        elif self.params.model_type == "svit":
            self.model = build_svit(self.params).to(self.device)
        elif self.params.model_type == "vit_all2all":
            self.model = build_vit(self.params).to(self.device)
        elif self.params.model_type == "turbt":
            self.model = build_turbt(self.params).to(self.device)

        if self.params.compile:
            print('WARNING: BFLOAT NOT SUPPORTED IN SOME COMPILE OPS SO SWITCHING TO FLOAT16')
            self.mp_type = torch.half
            self.model = torch.compile(self.model)

        if dist.is_initialized():
            if self.params.use_fsdp:
                self.model = FSDP(self.model, use_orig_params=True,
                                auto_wrap_policy=size_based_auto_wrap_policy,
                                cpu_offload=CPUOffload(offload_params=False))
            elif self.params.use_ddp:
                self.model = DDP(self.model, device_ids=[self.local_rank],
                                output_device=[self.local_rank], find_unused_parameters=True)
            else:
                raise ValueError("checkp distributed option, only support ddp and fsdp")

        self.single_print(f'Model parameter count: {sum([p.numel() for p in self.model.parameters()])}')
        if self.global_rank == 0:
            print(self.model)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.data.numel())

    def initialize_optimizer(self):
        parameters = add_weight_decay(self.model, self.params.weight_decay) # Dont use weight decay on bias/scaling terms
        if self.params.use_fsdp:
            #FIXME: DCP.load does not work properly with multiple parameter groups (introduced in add_weight_decay);
            # it changes the keys of the 2nd group (https://github.com/pytorch/pytorch/issues/143828#issuecomment-2568700480) and
            # causes errors when resume optimizer from checkpoint; so for now we skip the hybrid weight decay
            parameters=self.model.parameters()
        if self.params.optimizer == 'DAdaptAdam':
            self.optimizer =  DAdaptAdam(parameters, lr=1., growth_rate=1.05, log_every=100, decouple=True )
        elif  self.params.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(parameters, lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.params.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.params.optimizer} not supported")
        self.gscaler = amp.GradScaler(enabled= (self.mp_type == torch.half and self.params.enable_amp))

    def initialize_scheduler(self):
        self.scheduler = None
        if self.params.scheduler_epochs > 0:
            sched_epochs = self.params.scheduler_epochs
        else:
            sched_epochs = self.params.max_epochs
        if self.params.scheduler == 'cosine':
            if self.params.learning_rate < 0:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                            #last_epoch = (self.startEpoch*params.epoch_size) - 1,
                                                                            T_max=sched_epochs*self.params.epoch_size//self.params.accum_grad,
                                                                            eta_min=self.params.learning_rate / 100)
            else:
                k = self.params.warmup_steps
                warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=.01, end_factor=1.0, total_iters=k)
                decay = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=self.params.learning_rate / 100, T_max=sched_epochs*self.params.epoch_size//self.params.accum_grad-k)
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup, decay], [k])#, last_epoch=(self.params.epoch_size*self.startEpoch)-1)
        elif self.params.scheduler == 'warmuponly':
            k = self.params.warmup_steps
            #self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=.01, end_factor=1.0, total_iters=k)
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=.01, end_factor=1.0, total_iters=k)#, last_epoch=(params.epoch_size*self.startEpoch)//self.params.accum_grad-1)
        elif self.params.scheduler == 'linear':
            k = self.params.warmup_steps
            #if (self.startEpoch*params.epoch_size) < k*self.params.accum_grad:
            warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=.01, end_factor=1.0, total_iters=k)
            decay  = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=.01, total_iters=sched_epochs*self.params.epoch_size//self.params.accum_grad-k)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup, decay], [k])#, last_epoch=(self.params.epoch_size*self.startEpoch)-1)
        elif self.params.scheduler == 'steplr':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)#, last_epoch=self.startEpoch-1) #gamma: lr decay rate; step_size: period of learning rate decay
        else:
            self.scheduler = None

    def save_checkpoint(self, checkpoint_path, model=None):
        """ Save model and optimizer to checkpoint """
        if not model:
            model = self.model

        if self.params.use_fsdp:
            model_state_dict, optimizer_state_dict = torch.distributed.checkpoint.state_dict.get_state_dict(model, self.optimizer)
            dcp.save({'model_state': model_state_dict,'optimizer_state_dict': optimizer_state_dict}, checkpoint_id=checkpoint_path) 
            if self.global_rank == 0:
                torch.save({'iters': self.epoch*self.params.epoch_size, 'epoch': self.epoch}, os.path.join(checkpoint_path,"iters_epoch.pth"))
        else:
            if self.global_rank == 0:
                if self.scheduler is not None:
                    torch.save({'iters': self.epoch*self.params.epoch_size, 'epoch': self.epoch, 'model_state': model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),'scheduler_state_dict': self.scheduler.state_dict()}, checkpoint_path)
                else:
                    torch.save({'iters': self.epoch*self.params.epoch_size, 'epoch': self.epoch, 'model_state': model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint_dcp(self, checkpoint_path):
        
        module_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        state_dict = {"model_state": module_state_dict, "optimizer_state_dict": optimizer_state_dict}
        dcp.load(state_dict, checkpoint_id=checkpoint_path)
        set_model_state_dict(self.model, model_state_dict=state_dict["model_state"])
        if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict=state_dict["optimizer_state_dict"])
            iters_epoch = torch.load(os.path.join(checkpoint_path,"iters_epoch.pth"), map_location='cuda:{}'.format(self.local_rank) if torch.cuda.is_available() else torch.device('cpu'))
            self.iters = iters_epoch["iters"]
            self.startEpoch = iters_epoch['epoch']
            self.epoch = self.startEpoch
        else:
            self.iters = 0

    def restore_checkpoint(self, checkpoint_path):
        print(f"restoring checkpoint........{checkpoint_path}")
        print("before", self.optimizer.param_groups[0]['lr'])
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

        if self.params.use_fsdp:
            self.restore_checkpoint_dcp(checkpoint_path)
        else:
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
            if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
                self.iters = checkpoint['iters']
                #note the order for sequentialLR: lr -> optimizer, see https://github.com/pytorch/pytorch/issues/119168
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.startEpoch = checkpoint['epoch']
                self.epoch = self.startEpoch
            else:
                self.iters = 0
            self.model = self.model.to(self.device)
        print(f"epoch:{self.epoch}")
        print("after", self.optimizer.param_groups[0]['lr'])

    def expand_model_pretraining(self):
        # See how much we need to expand the projections
        exp_proj = 0
        # Iterate through the appended datasets and add on enough embeddings for all of them.
        for add_on in self.params.append_datasets:
            exp_proj += len(DSET_NAME_TO_OBJECT[add_on]._specifics()[2])
        try:
            self.model.module.expand_projections(exp_proj)
        except:
            self.model.expand_projections(exp_proj)

        expand_leadtime=False
        if hasattr(self.params, 'leadtime_max_finetuning') and self.params.leadtime_max_finetuning>1:
            if hasattr(self.params, 'leadtime_max'):
                if self.params.leadtime_max==1:
                    expand_leadtime=True
            else:
                expand_leadtime=True
        try:
            self.model.module.expand_leadtime(expand_leadtime, self.params.embed_dim)
        except:
            self.model.expand_leadtime(expand_leadtime, self.params.embed_dim)

        self.model = self.model.to(self.device)

    def expand_model_pretraining_convheads(self):
        if self.refine_resol is None:
            return

        try:
            self.model.module.expand_conv_projections(self.refine_resol)
        except:
            self.model.expand_conv_projections(self.refine_resol)

        self.model = self.model.to(self.device)

    def expand_model_pretraining_sts_model(self):
        try:
            self.model.module.expand_sts_model()
        except:
            self.model.expand_sts_model()

        self.model = self.model.to(self.device)

    def freeze_model_pretraining(self):
        if self.params.freeze_middle:
            try:
                self.model.module.freeze_middle()
            except:
                self.model.freeze_middle()
        elif self.params.freeze_processor:
            try:
                self.model.module.freeze_processor()
            except:
                self.model.freeze_processor()
        else:
            try:
                self.model.module.unfreeze()
            except:
                self.model.unfreeze()

        self.model = self.model.to(self.device)

    def train_one_epoch(self):

        self.epoch += 1
        tr_time = 0
        data_time = 0
        data_start = self.timer.get_time()
        self.model.train()
        logs = {'train_rmse': torch.zeros(1).to(self.device),
                'train_nrmse': torch.zeros(1).to(self.device),
            'train_l1': torch.zeros(1).to(self.device)}
        steps = 0
        grad_logs = defaultdict(lambda: torch.zeros(1, device=self.device))
        grad_counts = defaultdict(lambda: torch.zeros(1, device=self.device))
        loss_logs = defaultdict(lambda: torch.zeros(1, device=self.device))
        loss_counts = defaultdict(lambda: torch.zeros(1, device=self.device))
        self.single_print('train_loader_size', len(self.train_data_loader), 'train_dataset size', len(self.train_dataset), 'valid_dataset size', len(self.valid_dataset))
        sts_train=self.params.sts_train if hasattr(self.params, 'sts_train') else False

        data_iter = iter(self.train_data_loader)
        num_batches = min(len(self.train_data_loader), self.params.epoch_size)
        
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx in range(num_batches):
            steps += 1
            #if steps>self.params.epoch_size:
            #    break
            self.single_print('Training batch:', batch_idx, "of Total:", num_batches)
            ##############################################################################################################
            with record_function_opt("data loading", enabled=self.profiling):
                data = next(data_iter)
                inp, dset_index, field_labels, bcs, tar, leadtime = map(lambda x: x.to(self.device), [data[varname] for varname in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
                try:
                    refineind = data["refineind"].to(self.device)
                except:
                    refineind = None
                try:
                    blockdict = self.train_dataset.sub_dsets[dset_index[0]].blockdict
                except:
                    blockdict = None
                #if self.group_rank==0:
                #    print(f"{self.global_rank}, {batch_idx}, Pei checking data shape, ", inp.shape, tar.shape, blockdict, flush=True)
            dset_type = self.train_dataset.sub_dsets[dset_index[0]].type
            tkhead_name = self.train_dataset.sub_dsets[dset_index[0]].tkhead_name
            loss_counts[dset_type] += 1
            data_time += self.timer.get_time() - data_start
            dtime = self.timer.get_time() - data_start

            self.model.require_backward_grad_sync = ((1+batch_idx) % self.params.accum_grad == 0)
            with amp.autocast(self.params.enable_amp, dtype=self.mp_type):
                model_start = self.timer.get_time()
                tar = tar.to(self.device)
                inp = rearrange(inp.to(self.device), 'b t c d h w -> t b c d h w')
                imod = self.params.hierarchical["nlevels"]-1 if hasattr(self.params, "hierarchical") else 0
                with record_function_opt("model forward", enabled=self.profiling):
                    output= self.model(inp, field_labels, bcs, imod=imod,
                                    sequence_parallel_group=self.current_group, leadtime=leadtime, 
                                    refineind=refineind, tkhead_name=tkhead_name, blockdict=blockdict)
                ###full resolution###
                spatial_dims = tuple(range(output.ndim))[2:] # B,C,D,H,W
                residuals = output - tar
                if self.params.pei_debug:
                    checking_data_pred_tar(tar, output, blockdict, self.global_rank, self.current_group, self.group_rank, self.group_size, 
                                           self.device, self.params.debug_outdir, istep=steps, imod=-1)
                # Differentiate between log and accumulation losses
                #B,C,D,H,W->B,C
                raw_loss = residuals.pow(2).mean(spatial_dims)/ (1e-7 + tar.pow(2).mean(spatial_dims))
                # Scale loss for accum
                loss = raw_loss.mean() / self.params.accum_grad
                # Logging
                with torch.no_grad():
                    logs['train_l1'] += F.l1_loss(output, tar)
                    log_nrmse = raw_loss.sqrt().mean()
                    logs['train_nrmse'] += log_nrmse 
                    loss_logs[dset_type] += loss.item()
                    logs['train_rmse'] += residuals.pow(2).mean(spatial_dims).sqrt().mean()
                #################################
                forward_end = self.timer.get_time()
                forward_time = forward_end-model_start
                if torch.isnan(loss) or  not torch.isfinite(loss):
                    print(f"NaN detected in loss at batch {batch_idx}. Skipping batch...")
                with record_function_opt("model backward", enabled=self.profiling):
                    self.gscaler.scale(loss).backward()
                backward_end = self.timer.get_time()
                backward_time = backward_end - forward_end
                # Only take step once per accumulation cycle
                optimizer_step = 0
                with record_function_opt("optimization", enabled=self.profiling):
                    if self.model.require_backward_grad_sync:
                        self.gscaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.gscaler.step(self.optimizer)
                        self.gscaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler is not None and self.params.scheduler != 'steplr':
                            self.scheduler.step()
                        optimizer_step = self.timer.get_time() - backward_end
                tr_time += self.timer.get_time() - model_start
                if self.log_to_screen and batch_idx % self.params.log_interval == 0 and self.global_rank == 0:
                    print(f"Epoch {self.epoch} Batch {batch_idx} Train Loss {log_nrmse.item()}")
                if self.log_to_screen:
                    print('Total Times. Batch: {}, Rank: {}, Data Shape: {}, Data time: {}, Forward: {}, Backward: {}, Optimizer: {}, lr:{}, leadtime.max: {}'.format(
                        batch_idx, self.global_rank, inp.shape, dtime, forward_time, backward_time, optimizer_step, self.optimizer.param_groups[0]['lr'], leadtime.max()))
                data_start = self.timer.get_time()
            self.check_memory("train-end %d"%batch_idx)
        if self.params.scheduler == 'steplr':
            self.scheduler.step()
        logs = {k: v/steps for k, v in logs.items()}
        with record_function_opt("log_dist_update", enabled=self.profiling):
            # If distributed, do lots of logging things
            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    dist.all_reduce(logs[key].detach())
                    logs[key] = float(logs[key]/dist.get_world_size())
                for key in sorted(loss_logs.keys()):
                    dist.all_reduce(loss_logs[key].detach())
                for key in sorted(grad_logs.keys()):
                    dist.all_reduce(grad_logs[key].detach())
                for key in sorted(loss_counts.keys()):
                    dist.all_reduce(loss_counts[key].detach())
                for key in sorted(grad_counts.keys()):
                    dist.all_reduce(grad_counts[key].detach())

            for key in loss_logs.keys():
                logs[f'{key}/train_nrmse'] = loss_logs[key] / loss_counts[key]

            self.iters += steps
            if self.global_rank == 0:
                logs['iters'] = self.iters
            self.single_print('all reduces executed!')

        return tr_time, data_time, logs

    def validate_one_epoch(self, full=False, cutoff_skip=False):
        self.model.eval()
        self.single_print('STARTING VALIDATION!!!')
        logs = {'valid_rmse':  torch.zeros(1).to(self.device),
                'valid_nrmse': torch.zeros(1).to(self.device),
                'valid_l1':    torch.zeros(1).to(self.device)}
        if cutoff_skip:
            return logs
        loss_dset_logs      = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_l1_dset_logs   = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_rmse_dset_logs = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_dset_counts    = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}

        self.single_print('val_loader_size', len(self.valid_data_loader), len(self.valid_dataset))
        steps = 0
        valid_iter = iter(self.valid_data_loader)
        
        if full:
            cutoff = len(self.valid_data_loader)
        else:
            cutoff = 5 #40

        for idx in range(len(self.valid_data_loader)):
            self.check_memory("validate-data")
            self.single_print("valid index:", idx, "of:", len(self.valid_data_loader))
            ##############################################################################################################
            try:
                data = next(valid_iter)
            except:
                self.single_print(f"No more data to sample in valid_data_loader after {idx} batches")
                break

            inp, file_index, field_labels, bcs, tar, leadtime = map(lambda x: x.to(self.device), [data[varname] for varname in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
            try:
                refineind = data["refineind"].to(self.device)
            except:
                refineind = None
            try:
                blockdict = self.valid_dataset.sub_dsets[dset_index[0]].blockdict
            except:
                blockdict = None
            #if self.group_rank==0:
            #    print(f"{self.global_rank}, {idx}, Pei checking val data shape, ", inp.shape, tar.shape, blockdict, flush=True)
            dset_type = self.valid_dataset.sub_dsets[dset_index[0]].type
            tkhead_name = self.valid_dataset.sub_dsets[dset_index[0]].tkhead_name            
            ##############################################################################################################
            if loss_dset_counts[dset_type]>cutoff: 
                break
            with amp.autocast(self.params.enable_amp, dtype=self.mp_type):
                steps += 1
                loss_dset_counts[dset_type] += 1
                with torch.no_grad():
                    tar = tar.to(self.device)
                    inp = rearrange(inp.to(self.device), 'b t c d h w -> t b c d h w')
                    imod = self.params.hierarchical["nlevels"]-1 if hasattr(self.params, "hierarchical") else 0
                    output= self.model(inp, field_labels, bcs, imod=imod, 
                                       sequence_parallel_group=self.current_group, leadtime=leadtime, 
                                       refineind=refineind, tkhead_name=tkhead_name, blockdict=blockdict)                   
                    #################################
                    ###full resolution###
                    spatial_dims = tuple(range(output.ndim))[2:]
                    residuals = output - tar
                    # Differentiate between log and accumulation losses
                    raw_loss = residuals.pow(2).mean(spatial_dims)/(1e-7+ tar.pow(2).mean(spatial_dims))
                    raw_loss = raw_loss.sqrt().mean()
                    raw_l1_loss = F.l1_loss(output, tar)
                    raw_rmse_loss = residuals.pow(2).mean(spatial_dims).sqrt().mean()
                    logs['valid_nrmse'] += raw_loss
                    logs['valid_l1']    += raw_l1_loss
                    logs['valid_rmse']  += raw_rmse_loss

                    loss_dset_logs[dset_type]      += raw_loss
                    loss_l1_dset_logs[dset_type]   += raw_l1_loss
                    loss_rmse_dset_logs[dset_type] += raw_rmse_loss
                    #################################        
            self.check_memory("validate-end")
        self.single_print('DONE VALIDATING - NOW SYNCING')
        logs = {k: v/steps for k, v in logs.items()}
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key])
                logs[key] = float(logs[key]/dist.get_world_size())

            for key in sorted(loss_dset_logs.keys()):
                dist.all_reduce(loss_dset_logs[key])
                dist.all_reduce(loss_l1_dset_logs[key])
                dist.all_reduce(loss_rmse_dset_logs[key])
                dist.all_reduce(loss_dset_counts[key])

        for key in loss_dset_logs.keys():
            logs[f'{key}/valid_nrmse'] = loss_dset_logs[key]     / loss_dset_counts[key]
            logs[f'{key}/valid_l1']    = loss_l1_dset_logs[key]  / loss_dset_counts[key]
            logs[f'{key}/valid_rmse']  = loss_rmse_dset_logs[key]/ loss_dset_counts[key]
        self.single_print('DONE SYNCING - NOW LOGGING')

        return logs

    def train(self):
        if self.global_rank == 0:
            summary(self.model)
        self.single_print("Starting Training Loop...")
        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
            start = self.timer.get_time()
            self.check_memory("before train %d"%epoch)
            # with torch.autograd.detect_anomaly(check_nan=True):
            with record_function_opt("train_one_epoch", enabled=self.profiling):
                tr_time, data_time, train_logs = self.train_one_epoch()

            self.check_memory("after train %d"%epoch)
            valid_start = self.timer.get_time()
            # Only do full validation set on last epoch - don't waste time
            with record_function_opt("validate_one_epoch", enabled=self.profiling):
                if epoch==self.params.max_epochs-1:
                    valid_logs = self.validate_one_epoch(full = True)
                else:
                    valid_skip=False
                    valid_skipsteps = self.params.valid_skipsteps if hasattr(self.params, 'valid_skipsteps') else 1
                    if epoch%valid_skipsteps>0:
                        valid_skip=True
                    valid_logs = self.validate_one_epoch(cutoff_skip=valid_skip)

            self.check_memory("after validate %d"%epoch)
            post_start = self.timer.get_time()
            train_logs.update(valid_logs)
            train_logs['time/train_time'] = valid_start-start
            train_logs['time/train_data_time'] = data_time
            train_logs['time/train_compute_time'] = tr_time
            train_logs['time/valid_time'] = post_start-valid_start
            gc.collect()
            torch.cuda.empty_cache()
            self.check_memory("after memory clean %d"%epoch)

            if self.global_rank == 0:
                logs_config = {}
                for k, v in train_logs.items():
                    if isinstance(v, torch.Tensor):
                        logs_config[k] = v.cpu().item()
                    else:
                        logs_config[k] = v
                with open(os.path.join(self.params.experiment_dir, f"train_log_epoch{epoch}.json"), 'w') as fp:
                    json.dump(logs_config, fp)
            if self.global_rank == 0:
                if self.params.save_checkpoint:
                    print("checkpoint saved:",self.params.checkpoint_path )
                    self.save_checkpoint(self.params.checkpoint_path)
                if epoch % self.params.checkpoint_save_interval == 0:
                    self.save_checkpoint(self.params.checkpoint_path + f'_epoch{epoch}')
                if  valid_logs['valid_nrmse']>0 and valid_logs['valid_nrmse'] <= best_valid_loss:
                    print("Best checkpoint saved:",self.params.best_checkpoint_path )
                    self.save_checkpoint(self.params.best_checkpoint_path)
                    best_valid_loss = valid_logs['valid_nrmse']

            cur_time = self.timer.get_time()
            self.single_print(f'Time for train {valid_start-start}. For valid: {post_start-valid_start}. For postprocessing:{cur_time-post_start}')
            self.single_print('Time taken for epoch {} is {} sec'.format(epoch + 1, self.timer.get_time()-start))
            self.single_print('Train loss: {}. Valid loss: {}'.format(train_logs['train_nrmse'], valid_logs['valid_nrmse']))

