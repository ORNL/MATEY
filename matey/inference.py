# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
from collections import OrderedDict
from .data_utils.datasets import get_data_loader
from .models.avit import build_avit
from .models.svit import build_svit
from .models.vit import build_vit
from .models.turbt import build_turbt
from .utils.distributed_utils import get_sequence_parallel_group, determine_turt_levels
from .utils.forward_options import ForwardOptionsBase
from .trustworthiness.metrics import get_ssim
import json
from .utils.training_utils import autoregressive_rollout, update_loss_logs_inplace_eval
import copy

class Inferencer:
    def __init__(self, params, global_rank, local_rank, device):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        #define sequence parallel groups and local group info
        if hasattr(self.params, "sp_groupsize"):
            self.current_group, self.group_id, self.num_sequence_parallel_groups = get_sequence_parallel_group(sequence_parallel_groupsize=self.params.sp_groupsize)
        else:
            self.current_group, self.group_id, self.num_sequence_parallel_groups = get_sequence_parallel_group(num_sequence_parallel_groups=self.params.num_sequence_parallel_groups if hasattr(self.params, "num_sequence_parallel_groups") else self.world_size)

        self.group_rank = dist.get_rank(self.current_group)
        self.group_size = dist.get_world_size(self.current_group)

        self.initialize_data()
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
        if self.global_rank == 0:
            print(' '.join([str(t) for t in text]), flush =True)
    
   
    def initialize_data(self):
        #self.global_rank: global rank
        #self.group_size: number of ranks in each SP group
        #self.num_sequence_parallel_groups: number of SP groups
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
            num_replicas = self.num_sequence_parallel_groups
        else:
            in_rank = self.global_rank
            parallel_group_size=self.group_size
            group_rank=0
        print(f"Initializing data on rank {self.global_rank}; total {self.num_sequence_parallel_groups} SP groups with {self.group_size} ranks each", flush=True)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_data_paths,
                          dist.is_initialized(), split='train', train_offset=self.params.embedding_offset,
                          group_size= self.group_size, global_rank= self.global_rank, num_sp_groups=self.num_sequence_parallel_groups)
        self.valid_data_loader, self.valid_dataset, self.val_sampler = get_data_loader(self.params, self.params.valid_data_paths,
                          dist.is_initialized(), split='val',
                          group_size= self.group_size, global_rank= self.global_rank, num_sp_groups=self.num_sequence_parallel_groups)
        self.single_print("self.train_data_loader:",  len(self.train_data_loader), "valid_data_loader:", len(self.valid_data_loader))

    def initialize_model(self):
        if self.params.model_type == 'avit':
            self.model = build_avit(self.params).to(self.device)
        elif self.params.model_type == "svit":
            self.model = build_svit(self.params).to(self.device)
        elif self.params.model_type == "vit_all2all":
            self.model = build_vit(self.params).to(self.device)
        elif self.params.model_type == "turbt":
            self.model = build_turbt(self.params).to(self.device)

        if dist.is_initialized() and self.params.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                            output_device=[self.local_rank], find_unused_parameters=True)
           
        self.single_print(f'Model parameter count: {sum([p.numel() for p in self.model.parameters()])}')

    def restore_checkpoint(self, checkpoint_path):
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

    def model_forward(self, inp, field_labels, bcs, opts: ForwardOptionsBase, pushforward=True):
        # Handles a forward pass through the model, either normal or autoregressive rollout.
        autoregressive = getattr(self.params, "autoregressive", False)
        if not autoregressive:
            output = self.model(inp, field_labels, bcs, opts)
            return output, None
        else:
            # autoregressive rollout
            output, rollout_steps = autoregressive_rollout(self.model, inp, field_labels, bcs, opts, pushforward = pushforward)
            return output, rollout_steps

    def inference(self):
        self.model.eval()
        logs = {'valid_rmse':  torch.zeros(1).to(self.device),
                'valid_nrmse': torch.zeros(1).to(self.device),
                'valid_l1':    torch.zeros(1).to(self.device),
                'valid_ssim':  torch.zeros(1).to(self.device)}
        loss_dset_logs      = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_l1_dset_logs   = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_rmse_dset_logs = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}
        loss_dset_counts    = {dataset.type: torch.zeros(1, device=self.device) for dataset in self.valid_dataset.sub_dsets}

        self.single_print('val_loader_size', len(self.valid_data_loader), len(self.valid_dataset))
        steps = 0
        valid_iter = iter(self.valid_data_loader)

        for idx in range(len(self.valid_data_loader)):
            self.single_print("valid index:", idx, "of:", len(self.valid_data_loader))
            ##############################################################################################################
            data = next(valid_iter)
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
            ##############################################################################################################
            steps += 1
            loss_dset_counts[dset_type] += 1
            with torch.no_grad():
                tar = tar.to(self.device)
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
                print(f"Rank {self.global_rank} input shape {inp.shape if not isgraph else inp}, dset_type {dset_type}", flush=True)
                opts = ForwardOptionsBase(
                imod=imod, 
                imod_bottom=imod_bottom,
                tkhead_name=tkhead_name,
                sequence_parallel_group=seq_group,
                leadtime=leadtime,
                blockdict=copy.deepcopy(blockdict),
                cond_dict=copy.deepcopy(cond_dict),
                cond_input=cond_input,
                isgraph=isgraph,
                field_labels_out= field_labels_out
                )
                output, rollout_steps = self.model_forward(inp, field_labels, bcs, opts)
                if tar.ndim == 6: #B,T,C,D,H,W
                    if rollout_steps is None:
                        rollout_steps = leadtime.view(-1).long()
                    tar = tar[:, rollout_steps-1, :] # B,C,D,H,W
                update_loss_logs_inplace_eval(output, tar, graphdata if isgraph else None, logs, loss_dset_logs, loss_l1_dset_logs, loss_rmse_dset_logs, dset_type)
                if not isgraph and getattr(self.params, "log_ssim", False):
                        avg_ssim = get_ssim(output, tar, blockdict, self.global_rank, self.current_group, self.group_rank, self.group_size, self.device, self.valid_dataset, dset_index)
                        logs['valid_ssim'] += avg_ssim
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
    
        self.single_print('DONE SYNCING - Inference metrics')
        if self.global_rank==0:
            print(logs)

