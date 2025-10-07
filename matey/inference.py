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
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from .utils.metrics import SSIM3D, remove_edges
from .data_utils.blasnet_3Ddatasets import SR_Benchmark
import torch.nn.functional as F


class Inferencer:
    def __init__(self, params, global_rank, local_rank, device):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.log_to_screen = self.params.log_to_screen
        # Basic setup
        self.startEpoch = 0
        self.epoch = 0
        self.n_calls = 0
        self.cubic_interp = self.params.cubic_interp if hasattr(self.params, "cubic_interp") else False
        self.SSIM = SSIM3D()
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
        labels_total=[self.valid_dataset.subset_dict[dset] for dset in self.valid_dataset.subset_dict]
        labels_total = [item  for sublist in labels_total for item in sublist]
        if self.params.n_states<max(labels_total)+1:
            print(f"Warning, reserved n_states {self.params.n_states} is too small for datasets, set it to {max(labels_total)+1} instead")
            self.params.n_states = max(labels_total)+1

        self.initialize_model()
        self.timer=Timer(enable_sync=self.params.enable_sync)
        if self.params.resuming:
            if self.params.pretrained:
                #if fresh from pretrainig:
                    #model construction (same with pretrained) --> load pretrained weights --> expand input&output based on append_datasets --> freeze --> update optimizer
                #elif resume from finetuning:
                    #model construction  (same with pretrained) --> expand input and output (same with expanded) and freeeze --> load saved finetuned weights
                self.expand_model_pretraining()
                self.freeze_model_pretraining()
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

        if self.params.resuming == False and hasattr(self.params, "startfrom_path"):
            self.restore_checkpoint(self.params.startfrom_path)
            self.expand_model_pretraining_convheads()
            if self.sts_model and not self.params.sts_model:
                self.expand_model_pretraining_sts_model()
            self.startEpoch = 0
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
            if self.global_rank == 0 and self.params.data_augmentation:
                print(f"Data augmentation is enabled: {self.params.data_augmentation}", flush=True)
                
        self.valid_data_loader, self.valid_dataset, self.val_sampler = get_data_loader(self.params, self.params.valid_data_paths,
                          dist.is_initialized(), split='test',   rank=in_rank, group_rank=group_rank, group_size=parallel_group_size,
                          num_replicas=num_replicas)
        
        self.single_print("valid_data_loader:", len(self.valid_data_loader))
        if dist.is_initialized():
            self.val_sampler.set_epoch(1)
    
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
            self.optimizer =  DAdaptAdam(parameters, lr=1., growth_rate=1.05, log_every=100, decouple=True)
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
            iters_epoch = torch.load(os.path.join(checkpoint_path,"iters_epoch.pth"), map_location='cuda:{}'.format(self.local_rank) if torch.cuda.is_available() else torch.device('cpu'))
            self.iters = iters_epoch["iters"]
            self.startEpoch = iters_epoch['epoch']
            self.epoch = self.startEpoch
        else:
            self.iters = 0

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

            else:
                self.iters = 0
            self.model = self.model.to(self.device)
        print(f"epoch:{self.epoch}")

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

    def total_variation_3d(self,x):
        """
        x: tensor of shape (B, C, T, D, H, W)
        Returns: scalar total variation loss
        """
        tv_z = torch.abs(x[:, :, :, 1:, :, :] - x[:, :, :, :-1, :, :]).mean()
        tv_y = torch.abs(x[:, :, :, :, 1:, :] - x[:, :, :, :, :-1, :]).mean()
        tv_x = torch.abs(x[:, :, :, :, :, 1:] - x[:, :, :, :, :, :-1]).mean()
        
        return tv_z + tv_y + tv_x

    def stitch_blocks(self, blocks, full_size, block_size):
        """
            blocks: list of 8 arrays [num_vars, block_size, block_size, block_size]
            full_size: final spatial size (128 for high-res, 16 for low-res)
            block_size: spatial size per block (64 for high-res, 8 for low-res)
        """
        arr = np.empty((blocks[0].shape[0], full_size, full_size, full_size), dtype=blocks[0].dtype)
        idx = 0
        for i in range(2):       # depth
            for j in range(2):   # height
                for k in range(2):  # width
                    arr[:, 
                        i*block_size:(i+1)*block_size, 
                        j*block_size:(j+1)*block_size, 
                        k*block_size:(k+1)*block_size
                    ] = blocks[idx]
                    idx += 1
        return arr

    def compute_energy_spectrum(self, tensor_rescaled, dk=1.0):
        """
        Compute 3D kinetic energy spectrum from a tensor of [B,C,D,H,W].
        
        tensor_rescaled : torch.Tensor
            Rescaled tensor of shape [B,C,D,H,W] (or [C,D,H,W]) containing at least 
        channels 0–3: rho,u,v,w.
        dk : float
            Bin width in wavenumber space.
        """

        # Extract components
        rho = tensor_rescaled[:, 0, :]
        u   = tensor_rescaled[:, 1, :]
        v   = tensor_rescaled[:, 2, :]
        w   = tensor_rescaled[:, 3, :]
        
        B, D, H, W = u.shape
        N = D * H * W   # total number of grid points

        # Compute FFTs
        u_hat = torch.fft.fftn(u)/N
        v_hat = torch.fft.fftn(v)/N
        w_hat = torch.fft.fftn(w)/N

        # Energy in spectral space
        E_hat = 0.5 * (torch.abs(u_hat)**2 + torch.abs(v_hat)**2 + torch.abs(w_hat)**2)

        # Wavenumbers
        kx = torch.fft.fftfreq(W) * W
        ky = torch.fft.fftfreq(H) * H
        kz = torch.fft.fftfreq(D) * D

        KX, KY, KZ = torch.meshgrid(kz, ky, kx, indexing='ij')
        K_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)

        # Flatten and bin
        K_flat = K_mag.flatten().cpu().numpy()
        E_flat = E_hat.flatten().cpu().numpy()

        k_bins = np.arange(0.5, K_flat.max(), dk)
        bin_idx = np.digitize(K_flat, k_bins)

        E_spectrum = np.zeros(len(k_bins))
        for i in range(1, len(k_bins)):
            E_spectrum[i - 1] = E_flat[bin_idx == i].sum()

        max_rr= 64
        k_bins = k_bins[:max_rr+1]
        E_spectrum = E_spectrum[:max_rr+1]

        return k_bins, E_spectrum

    def compute_energy_spectrum_TKE(self, tensor_rescaled, L=2.0*np.pi):
        """
        Compute 3D turbulent kinetic energy spectrum from a tensor of shape [1,C,D,H,W].
        
        tensor_rescaled : torch.Tensor
            Tensor of shape [1,C,D,H,W] containing at least channels 1–3: u,v,w.
        L : float
            Physical box size (default 2*pi).
        """
        # Extract velocity components
        uvec = tensor_rescaled[:, 1:4, :, :, :].cpu().numpy()  # [1,3,D,H,W]
        B, C, D, H, W = uvec.shape
        assert B == 1, "This function assumes batch size = 1"
        assert C == 3, "Expecting 3 velocity components (u,v,w)"
        assert D == H == W, "Assumes cubic domain"

        N = D
        dx = L / N
        k0 = 2.0 * np.pi / L

        # Extract single snapshot
        u1 = uvec[0,0]
        u2 = uvec[0,1]
        u3 = uvec[0,2]

        # --- Compute velocity RMS ---
        u_rms = np.sqrt(np.mean(u1**2 + u2**2 + u3**2))

        # FFT and 3D energy
        U1 = np.fft.fftn(u1)/(N**3)
        U2 = np.fft.fftn(u2)/(N**3)
        U3 = np.fft.fftn(u3)/(N**3)

        E_k_3D = 0.5 * (np.abs(U1)**2 + np.abs(U2)**2 + np.abs(U3)**2)
        E_k_3D = np.fft.fftshift(E_k_3D)

        # Spherical wavenumber binning
        center = N//2
        i_vals = np.arange(N)
        j_vals = np.arange(N)
        k_vals = np.arange(N)
        ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
        rr = np.sqrt((ii - center)**2 + (jj - center)**2 + (kk - center)**2)
        wn_array = np.rint(rr).astype(np.int32)

        wn_flat = wn_array.ravel()
        E_flat  = E_k_3D.ravel()
        Ek = np.bincount(wn_flat, weights=E_flat)

        # Wavenumber array
        kvals = np.arange(len(Ek)) * k0

        max_rr= 64
        kvals= kvals[:max_rr+1]
        Ek   = Ek[:max_rr+1]

        # # --- Normalize ---
        # kvals_norm = kvals * L / (2*np.pi)   # dimensionless wavenumber k* = k/k0
        # Ek_norm = Ek / (u_rms**2)            # normalized energy spectrum

        return kvals, Ek

    def validate_one_epoch(self, full=False, cutoff_skip=False):
        self.model.eval()
        self.single_print('STARTING VALIDATION!!!')
        logs = {'valid_rmse':  torch.zeros(1).to(self.device),
                'valid_nrmse': torch.zeros(1).to(self.device),
                'valid_nmse': torch.zeros(1).to(self.device),
                'valid_mse': torch.zeros(1).to(self.device),
                'valid_interp_nrmse': torch.zeros(1).to(self.device),
                'valid_interp_rmse': torch.zeros(1).to(self.device),
                'valid_interp_nmse': torch.zeros(1).to(self.device),
                'valid_interp_mse': torch.zeros(1).to(self.device),
                'valid_l1':    torch.zeros(1).to(self.device),
                'valid_ssim':  torch.zeros(1).to(self.device),
                'valid_interp_ssim':  torch.zeros(1).to(self.device),}
        if cutoff_skip:
            return logs
        loss_rmse_dset_logs      = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_nrmse_dset_logs = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_nmse_dset_logs = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_mse_dset_logs = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_l1_dset_logs   = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_interp_rmse_dset_logs   = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_interp_nrmse_dset_logs   = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_interp_nmse_dset_logs   = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_interp_mse_dset_logs   = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_dset_counts    = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_ssim_dset_logs = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_interp_ssim_dset_logs = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}

        self.single_print('val_loader_size', len(self.valid_data_loader), len(self.valid_dataset))
        steps = 0
        valid_iter = iter(self.valid_data_loader)
        if full:
            cutoff = len(self.valid_data_loader)
        else:
            cutoff = 2 #40 
        plot_spectra = False
        if plot_spectra:
            spectra_list = []
        for idx in range(cutoff):
            self.check_memory("validate-data")
            self.single_print("valid index:", idx, "of:", len(self.valid_data_loader))
            ##############################################################################################################
            try:
                data = next(valid_iter) 
            except:
                self.single_print(f"No more data to sample in valid_data_loader after {idx} batches")
                break
            try:
                inp, dset_index, field_labels, bcs, tar, inp_up, leadtime =  data
                refineind = None
            except:
                inp, dset_index, field_labels, bcs, tar, refineind, inp_up, leadtime = data
            try:
                blockdict = self.valid_dataset.sub_dsets[dset_index[0]].blockdict
            except:
                blockdict = None
            #if self.group_rank==0:
            #    print(f"{self.global_rank}, {idx}, Pei checking val data shape, ", inp.shape, tar.shape, blockdict, flush=True)
            dset_type = self.valid_dataset.sub_dsets[dset_index[0]].type
            tkhead_name = self.valid_dataset.sub_dsets[dset_index[0]].tkhead_name            
            ##############################################################################################################
            if self.valid_dataset.sub_dsets[dset_index[0]].split == 'train':
                mean,std = self.valid_dataset.sub_dsets[dset_index[0]].get_mean_std()
                mean,std = torch.tensor(mean, device = self.device), torch.tensor(std, device = self.device)
            elif self.valid_dataset.sub_dsets[dset_index[0]].split == 'test':
                mean,std = self.valid_dataset.sub_dsets[dset_index[0]].get_mean_std_test()
                mean,std = torch.tensor(mean, device = self.device), torch.tensor(std, device = self.device)
            if loss_dset_counts[dset_type]>cutoff: 
                break
            with amp.autocast(self.params.enable_amp, dtype=self.mp_type):
                steps += 1
                loss_dset_counts[dset_type] += 1
                with torch.no_grad():
                    tar = tar.to(self.device)
                    inp = rearrange(inp.to(self.device), 'b t c d h w -> t b c d h w')
                    inp_up = rearrange(inp_up.to(self.device), 'b t c d h w -> t b c d h w')
                    inp_up = inp_up.to(self.device)

                    inp_reshaped = rearrange(inp, 't b c1 d h w -> (t b) c1 d h w')
                    inp_low = inp_reshaped.clone()

                    inp = inp_up
                    imod = self.params.hierarchical["nlevels"]-1 if hasattr(self.params, "hierarchical") else 0
                    output= self.model(inp, field_labels, bcs, imod=imod, 
                                       sequence_parallel_group=self.current_group, leadtime=leadtime, 
                                       refineind=refineind, tkhead_name=tkhead_name, blockdict=blockdict)                   
                    #################################
                    ###full resolution###
                    spatial_dims = tuple(range(output.ndim))[2:]
                    output = output.unsqueeze(1)#+inp_out

                    # Print results
                    channel_names = ["rho", "ux", "uy", "uz"]
                    chunks_tar = [torch.zeros_like(tar[0,0,:,:,:,:]) for _ in range(dist.get_world_size())]
                    chunks_out = [torch.zeros_like(output[0,0,:,:,:,:]) for _ in range(dist.get_world_size())]
                    chunks_inp = [torch.zeros_like(inp[0,0,:,:,:,:]) for _ in range(dist.get_world_size())]
                    chunks_inp_low = [torch.zeros_like(inp_low[0,:,:,:,:]) for _ in range(dist.get_world_size())]

                    dist.all_gather(chunks_tar, tar[0,0,:,:,:,:])
                    dist.all_gather(chunks_out, output[0,0,:,:,:,:])
                    dist.all_gather(chunks_inp, inp[0,0,:,:,:,:])
                    dist.all_gather(chunks_inp_low,inp_low[0,:,:,:,:])

                    if self.global_rank == 0:
                        tar_chunks = [c.cpu().detach().numpy() for c in chunks_tar]
                        out_chunks = [c.cpu().detach().numpy() for c in chunks_out]
                        inp_chunks = [c.cpu().detach().numpy() for c in chunks_inp]
                        inp_low_chunks = [c.cpu().detach().numpy() for c in chunks_inp_low]

                    if self.global_rank == 0:

                        full_tar = self.stitch_blocks(tar_chunks, full_size=128, block_size=64)
                        full_out = self.stitch_blocks(out_chunks, full_size=128, block_size=64)
                        full_inp = self.stitch_blocks(inp_chunks, full_size=128, block_size=64)
                        full_inp_low = self.stitch_blocks(inp_low_chunks, full_size=16, block_size=8)


                        if False:
                            save_dir = "saved_HIT_samples"
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"batch_{idx:04d}.npz")
                            np.savez_compressed(
                                save_path,
                                target=full_tar,
                                output=full_out,
                                interpolated=full_inp,
                                low_res_input=full_inp_low
                            )
                            print(f"Saveded HIT sample {idx} to {save_path}")
                            
                        print("Plot results - valid.")

                        slice_idx = 64  # mid-plane
                        tar_slice = full_tar[:, slice_idx, :, :]
                        out_slice = full_out[:, slice_idx, :, :]
                        inp_slice = full_inp[:, slice_idx, :, :]
                        in_slice_low = full_inp_low[:, 8, :, :]

                        num_vars = tar_slice.shape[0]
                        fig, axs = plt.subplots(5, num_vars, figsize=(5*num_vars, 12))

                        os.makedirs(f"{self.params.experiment_dir}/inference/plots", exist_ok=True)

                        # Compute shared color limits
                        vmin = tar_slice.min()
                        vmax = tar_slice.max()



                        num_vars = tar_slice.shape[0]
                        fig, axs = plt.subplots(4, num_vars, figsize=(4*num_vars, 12))

                        for i in range(num_vars):
                            vmin = tar_slice[i].min()
                            vmax = tar_slice[i].max()

                            name = channel_names[i]

                            im0 = axs[0, i].imshow(tar_slice[i], cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
                            axs[0, i].set_title(f"Target Var {i}")
                            fig.colorbar(im0, ax=axs[0, i], fraction=0.046, pad=0.04)

                            im1 = axs[1, i].imshow(out_slice[i], cmap='hot', origin='lower')
                            axs[1, i].set_title(f"{name} (Output)", fontsize=10)
                            fig.colorbar(im1, ax=axs[1, i], fraction=0.046, pad=0.04)

                            im2 = axs[2, i].imshow(inp_slice[i], cmap='hot', origin='lower')
                            axs[2, i].set_title(f"{name} (Interp Input-blocked)", fontsize=10)
                            fig.colorbar(im2, ax=axs[2, i], fraction=0.046, pad=0.04)

                            im3 = axs[3, i].imshow(in_slice_low[i], cmap='hot', origin='lower')
                            axs[3, i].set_title(f"Input low-res Var {i}")
                            fig.colorbar(im3, ax=axs[3, i], fraction=0.046, pad=0.04)

                        for ax_row in axs:
                            for ax in ax_row:
                                ax.axis('off')

                        plt.tight_layout()
                        plt.savefig(f'{self.params.experiment_dir}/inference/plots/target_vs_output_slice_{idx}.png', dpi=300)
                        plt.close()


                        full_out = torch.tensor(full_out, device=self.device)
                        full_tar = torch.tensor(full_tar, device=self.device)
                        full_inp = torch.tensor(full_inp, device=self.device)
                        full_inp_low = torch.tensor(full_inp_low, device=self.device)

                        output_rescaled = (full_out * std.view(1, 1, -1, 1, 1, 1) + mean.view(1, 1, -1, 1, 1, 1)).squeeze(0)
                        tar_rescaled = (full_tar * std.view(1, 1, -1, 1, 1, 1) + mean.view(1, 1, -1, 1, 1, 1)).squeeze(0)
                        inp_up_rescaled = (full_inp * std.view(1,1, -1, 1, 1, 1) + mean.view(1,1, -1, 1, 1, 1)).squeeze(0)
                        inp_low_rescaled = (full_inp_low * std.view(1,1, -1, 1, 1, 1) + mean.view(1,1, -1, 1, 1, 1)).squeeze(0)

                        if plot_spectra:
                            # Compute and store energy spectra
                            k_bins_out, E_spectrum_out = self.compute_energy_spectrum(output_rescaled)
                            k_bins_tar, E_spectrum_tar = self.compute_energy_spectrum(tar_rescaled)
                            k_bins_inp, E_spectrum_inp = self.compute_energy_spectrum(inp_up_rescaled)
                            k_bins_inp_low, E_spectrum_inp_low = self.compute_energy_spectrum(inp_low_rescaled)
                            spectra_list.append((k_bins_out, E_spectrum_out, k_bins_tar, E_spectrum_tar, k_bins_inp, E_spectrum_inp,
                                                k_bins_inp_low, E_spectrum_inp_low))

                        output_rescaled = remove_edges(output_rescaled)
                        tar_rescaled = remove_edges(tar_rescaled)
                        inp_up_rescaled = remove_edges(inp_up_rescaled)

                        ssimrho = self.SSIM(output_rescaled[:,0:1,:,:,:], tar_rescaled[:,0:1,:,:,:])
                        ssimux = self.SSIM(output_rescaled[:,1:2,:,:,:], tar_rescaled[:,1:2,:,:,:])
                        ssimuy = self.SSIM(output_rescaled[:,2:3,:,:,:], tar_rescaled[:,2:3,:,:,:])
                        ssimuz = self.SSIM(output_rescaled[:,3:4,:,:,:], tar_rescaled[:,3:4,:,:,:])
                        ssim_avg = (ssimrho + ssimux + ssimuy + ssimuz)/4.0
                        logs['valid_ssim'] += ssim_avg

                        mserho = F.mse_loss(output_rescaled[:,0:1,:,:,:], tar_rescaled[:,0:1,:,:,:])/mean[0]
                        mseux = F.mse_loss(output_rescaled[:,1:2,:,:,:], tar_rescaled[:,1:2,:,:,:])/mean[1]
                        mseuy = F.mse_loss(output_rescaled[:,2:3,:,:,:], tar_rescaled[:,2:3,:,:,:])/mean[2]
                        mseuz = F.mse_loss(output_rescaled[:,3:4,:,:,:], tar_rescaled[:,3:4,:,:,:])/mean[3]

                        nmse_loss = (mserho + mseux + mseuy + mseuz)/4.0
                        logs['valid_nmse'] += nmse_loss

                        ssimrho_interp = self.SSIM(inp_up_rescaled[:,0:1,:,:,:], tar_rescaled[:,0:1,:,:,:])
                        ssimux_interp = self.SSIM(inp_up_rescaled[:,1:2,:,:,:], tar_rescaled[:,1:2,:,:,:])
                        ssimuy_interp = self.SSIM(inp_up_rescaled[:,2:3,:,:,:], tar_rescaled[:,2:3,:,:,:])
                        ssimuz_interp = self.SSIM(inp_up_rescaled[:,3:4,:,:,:], tar_rescaled[:,3:4,:,:,:])

                        ssim_interp_avg = (ssimrho_interp + ssimux_interp + ssimuy_interp + ssimuz_interp)/4.0
                        logs['valid_interp_ssim'] += ssim_interp_avg

                        mserho_interp = F.mse_loss(inp_up_rescaled[:,0:1,:,:,:], tar_rescaled[:,0:1,:,:,:])/mean[0]
                        mseux_interp = F.mse_loss(inp_up_rescaled[:,1:2,:,:,:], tar_rescaled[:,1:2,:,:,:])/mean[1]
                        mseuy_interp = F.mse_loss(inp_up_rescaled[:,2:3,:,:,:], tar_rescaled[:,2:3,:,:,:])/mean[2]
                        mseuz_interp = F.mse_loss(inp_up_rescaled[:,3:4,:,:,:], tar_rescaled[:,3:4,:,:,:])/mean[3]

                        nmse_interp_loss = (mserho_interp + mseux_interp + mseuy_interp + mseuz_interp)/4.0
                        logs['valid_interp_nmse'] += nmse_interp_loss

                        print(f'Batch: {idx}, SSIM rho: {ssimrho}, SSIM ux: {ssimux}, SSIM uy: {ssimuy}, SSIM uz: {ssimuz}')
                        print(f'Batch: {idx}, SSIM rho interp: {ssimrho_interp}, SSIM ux interp: {ssimux_interp}, SSIM uy interp: {ssimuy_interp}, SSIM uz interp: {ssimuz_interp}')


                        residuals = output_rescaled - tar_rescaled
                        spatial_dims = tuple(range(full_out.ndim))[2:]
                        # Differentiate between log and accumulation losses
                        # raw_loss = residuals.pow(2).mean(spatial_dims)/(1e-7+ full_tar.pow(2).mean(spatial_dims))
                        # in BLASTNET paper they call it NRMSE but it's actually NMSE and they calculate it on the rescaled data!
                        # raw_loss = ((output_rescaled-tar_rescaled).pow(2).mean(spatial_dims)) / mean.view(1, -1, 1, 1, 1)
                        mse_loss = ((output_rescaled-tar_rescaled).pow(2).mean(spatial_dims)).mean()
                        nrmse_loss = (((output_rescaled - tar_rescaled).pow(2).mean(spatial_dims).sqrt()) / (1e-7 + tar_rescaled.pow(2).mean(spatial_dims).sqrt())).mean()
                        rmse_loss = ((output_rescaled-tar_rescaled).pow(2).mean(spatial_dims)).sqrt().mean()

                        # raw_loss = raw_loss.sqrt().mean()
                        # raw_loss = raw_loss.mean()
                        # interp_loss = (((inp_up_rescaled-tar_rescaled).pow(2).mean(spatial_dims))/ mean.view(1, -1, 1, 1, 1)).mean()
                        interp_mse_loss = ((inp_up_rescaled-tar_rescaled).pow(2).mean(spatial_dims)).mean()
                        interp_nrmse_loss = (((inp_up_rescaled-tar_rescaled).pow(2).mean(spatial_dims).sqrt())/ (1e-7 + tar_rescaled.pow(2).mean(spatial_dims).sqrt())).mean()
                        interp_rmse_loss = ((inp_up_rescaled-tar_rescaled).pow(2).mean(spatial_dims)).sqrt().mean()
                        # interp_loss = (((inp_up_rescaled - tar_rescaled).pow(2).mean(spatial_dims)) / (1e-7 + tar_rescaled.pow(2).mean(spatial_dims))).mean()
                        raw_l1_loss = F.l1_loss(full_out, full_tar)
                
                        # logs['valid_nmse'] += raw_loss
                        logs['valid_mse'] += mse_loss
                        logs['valid_nrmse'] += nrmse_loss
                        logs['valid_rmse']  += rmse_loss
                        logs['valid_l1']    += raw_l1_loss
                        # logs['valid_interp_nmse']  += interp_loss
                        logs['valid_interp_mse']  += interp_mse_loss
                        logs['valid_interp_nrmse']  += interp_nrmse_loss
                        logs['valid_interp_rmse']  += interp_rmse_loss

                        loss_nmse_dset_logs[dset_type]      += nmse_loss
                        loss_mse_dset_logs[dset_type]      += mse_loss
                        loss_nrmse_dset_logs[dset_type]      += nrmse_loss
                        loss_rmse_dset_logs[dset_type]      += rmse_loss
                        loss_l1_dset_logs[dset_type]   += raw_l1_loss
                        loss_interp_nmse_dset_logs[dset_type]   += nmse_interp_loss
                        loss_interp_mse_dset_logs[dset_type]   += interp_mse_loss
                        loss_interp_nrmse_dset_logs[dset_type]   += interp_nrmse_loss
                        loss_interp_rmse_dset_logs[dset_type]   += interp_rmse_loss
                        loss_ssim_dset_logs[dset_type] += ssim_avg
                        loss_interp_ssim_dset_logs[dset_type] += ssim_interp_avg

                    if self.global_rank == 0:
                        print(f"Epoch {self.epoch} Batch {idx} Rank 0: Valid Loss {nmse_loss.item()} Interp loss {nmse_interp_loss}")
                        
                        if plot_spectra:
                            # Loop over each snapshot in spectra_list
                            for snap_idx, s in enumerate(spectra_list):
                                k_bins_out, E_out, k_bins_tar, E_tar, k_bins_inp, E_inp, k_bins_inp_low, E_inp_low = s

                                # ---------- Plot spectra for this snapshot ----------
                                plt.figure(figsize=(10, 4))
                                plt.loglog(k_bins_tar[1:], E_tar[1:], 'k-', label='Ground truth')
                                plt.loglog(k_bins_inp[1:], E_inp[1:], 'b--', label='Interpolated')
                                plt.loglog(k_bins_out[1:], E_out[1:], 'r--', label='Super-resolved')
                                plt.loglog(k_bins_inp_low[1:], E_inp_low[1:], 'g--', label='Low-res')

                                plt.xlabel('Normalized Wavenumber k')
                                plt.ylabel('Normalized TKE E(k)')
                                plt.grid(True, which='both', ls='--')
                                plt.legend()
                                plt.tight_layout()

                                # Save with unique name per snapshot
                                plt.savefig(f'tke_spectrum_snapshot_mycode_{snap_idx:04d}.png', dpi=300)
                                plt.close()
                    #################################        
            self.check_memory("validate-end")
        if self.global_rank == 0:
            self.single_print('DONE VALIDATING - NOW SYNCING')
            logs = {k: v/steps for k, v in logs.items()}

            for key in loss_nmse_dset_logs.keys():
                logs[f'{key}/valid_nrmse'] = loss_nrmse_dset_logs[key]     / loss_dset_counts[key]
                logs[f'{key}/valid_mse'] = loss_mse_dset_logs[key]     / loss_dset_counts[key]
                logs[f'{key}/valid_rmse'] = loss_rmse_dset_logs[key]     / loss_dset_counts[key]
                logs[f'{key}/valid_nmse'] = loss_nmse_dset_logs[key]     / loss_dset_counts[key]
                logs[f'{key}/valid_l1']    = loss_l1_dset_logs[key]  / loss_dset_counts[key]
                logs[f'{key}/valid_interp_nrmse'] = loss_interp_nrmse_dset_logs[key]  / loss_dset_counts[key]
                logs[f'{key}/valid_interp_mse'] = loss_interp_mse_dset_logs[key]  / loss_dset_counts[key]
                logs[f'{key}/valid_interp_rmse'] = loss_interp_rmse_dset_logs[key]  / loss_dset_counts[key]
                logs[f'{key}/valid_interp_nmse'] = loss_interp_nmse_dset_logs[key]  / loss_dset_counts[key]
                logs[f'{key}/valid_rmse']  = loss_rmse_dset_logs[key]/ loss_dset_counts[key]
                logs[f'{key}/valid_ssim']  = loss_ssim_dset_logs[key]/ loss_dset_counts[key]
                logs[f'{key}/valid_interp_ssim']  = loss_interp_ssim_dset_logs[key]/ loss_dset_counts[key]
            self.single_print('DONE SYNCING - NOW LOGGING')


        return logs

    def infer(self):
        if self.global_rank == 0:
            summary(self.model)
        self.single_print("Starting Inference...")
        best_valid_loss = 1.e6
        if dist.is_initialized():
            self.val_sampler.set_epoch(0)
        start = self.timer.get_time()
        valid_start = self.timer.get_time()
        # Only do full validation set on last epoch - don't waste time
        with record_function_opt("validate_one_epoch", enabled=self.profiling):
            valid_logs = self.validate_one_epoch(full = True)

        post_start = self.timer.get_time()
        gc.collect()
        torch.cuda.empty_cache()

        cur_time = self.timer.get_time()
        self.single_print(f'Time valid: {post_start-valid_start}. For postprocessing:{cur_time-post_start}')
        self.single_print('Time taken for validation is {} sec'.format(self.timer.get_time()-start))
        self.single_print('Validation Metrics:')
        self.single_print('-------------------')

        # Print primary losses (global)
        self.single_print('Valid Losses:')
        self.single_print('- RMSE: {}'.format(valid_logs['valid_rmse'].item()))
        self.single_print('- NRMSE: {}'.format(valid_logs['valid_nrmse'].item()))
        self.single_print('- NMSE: {}'.format(valid_logs['valid_nmse'].item()))
        self.single_print('- MSE: {}'.format(valid_logs['valid_mse'].item()))
        self.single_print('- L1: {}'.format(valid_logs['valid_l1'].item()))

        # Print interpolated losses
        self.single_print('Interpolated Losses:')
        self.single_print('- Interp RMSE: {}'.format(valid_logs['valid_interp_rmse'].item()))
        self.single_print('- Interp NRMSE: {}'.format(valid_logs['valid_interp_nrmse'].item()))
        self.single_print('- Interp NMSE: {}'.format(valid_logs['valid_interp_nmse'].item()))
        self.single_print('- Interp MSE: {}'.format(valid_logs['valid_interp_mse'].item()))

        # Print SSIM metrics (Structural Similarity Index Metric)
        self.single_print('SSIM Metrics:')
        self.single_print('- SSIM: {}'.format(valid_logs['valid_ssim'].item()))
        self.single_print('- Interp SSIM: {}'.format(valid_logs['valid_interp_ssim'].item()))

