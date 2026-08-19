# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
from collections import OrderedDict
from .data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT, CANONICAL_FIELDS, CANONICAL_COND_FIELDS
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
from torch_geometric.nn import global_mean_pool
from torchinfo import summary

#below needed for load xgc mesh and convert to pyg objects
import sys
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False 
from mpi4py import MPI 
assert not MPI.Is_initialized()
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "third_party", "XGC_reader"))
from xgc_reader import base as xgc_base
import adios2 as ad2
import re
from .data_utils.graph_datasets import GraphXGCDataset
from .data_utils.datasets import my_collate
import numpy as np
from torch_geometric.data import Data


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

        self.set_field_dictionary() 
        self.initialize_data()
        self.ckpt_n_states = self.set_n_states_from_checkpoint(self.params.checkpoint_path)
        
        #checking input_states value
        self.new_embedding = getattr(self.params, "new_embedding", False)
        labels_total=[self.train_dataset.subset_dict[dset] for dset in self.train_dataset.subset_dict]
        labels_total = [item  for sublist in labels_total for item in sublist]
        if self.params.n_states<max(labels_total)+1:
            use_ckpt = hasattr(self, 'ckpt_n_states') and not self.new_embedding
            new_n_states = self.ckpt_n_states if use_ckpt else max(labels_total) + 1
            msg_suffix = (
                f"using checkpoint value {self.ckpt_n_states}" if use_ckpt else
                f"expanding to {new_n_states}" + (" (fully new embedding)" if self.new_embedding else ""))
            print(f"Warning: reserved n_states {self.params.n_states} too small — {msg_suffix}.")
            self.params.n_states = new_n_states
        """    
        try:
            labels_total=[self.train_dataset.subset_dict[dset] for dset in self.train_dataset.subset_dict]
            labels_total = [item  for sublist in labels_total for item in sublist]
            if self.params.n_states<max(labels_total)+1:
                print(f"Warning, reserved n_states {self.params.n_states} is too small for datasets, set it to {max(labels_total)+1} instead")
                self.params.n_states = max(labels_total)+1
        except:
            self.params.n_states = 536
        """

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
        print(f"Initializing data on rank {self.global_rank}; total {self.num_sequence_parallel_groups} SP groups with {self.group_size} ranks each", flush=True)        
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_data_paths,
                          dist.is_initialized(), split='train', train_offset=self.params.embedding_offset,
                          group_size= self.group_size, global_rank= self.global_rank, num_sp_groups=self.num_sequence_parallel_groups, canonical_fields=self.canonical_fields, canonical_cond_fields=self.canonical_cond_fields)
        self.valid_data_loader, self.valid_dataset, self.val_sampler = get_data_loader(self.params, self.params.valid_data_paths,
                          dist.is_initialized(), split='val',
                          group_size= self.group_size, global_rank= self.global_rank, num_sp_groups=self.num_sequence_parallel_groups, canonical_fields=self.canonical_fields, canonical_cond_fields=self.canonical_cond_fields)
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
                            output_device=self.local_rank, find_unused_parameters=True)
           
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
    
    def set_n_states_from_checkpoint(self, checkpoint_path):
        """Load n_states_actual from checkpoint before full model loading, so that we can initialize the model with correct n_states and avoid unnecessary expansion."""
        n_states = None
        if self.global_rank == 0:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'n_states_actual' in ckpt:
                n_states = ckpt['n_states_actual']
                # print(f"Overriding n_states from checkpoint to {n_states}")
        if dist.is_initialized():
            result = [n_states]
            dist.broadcast_object_list(result, src=0)
            n_states = result[0]
        return n_states
    def alias_to_key_mapping(self, d):
        alias_to_key = {}
        for key, aliases in d.items():
            for alias in aliases:
                if alias in alias_to_key:
                    raise ValueError(f"invalid field dictionary: alias '{alias}' appears under multiple canonical fields.")
                alias_to_key[alias] = key
        return alias_to_key
    def check_field_dictionary_consistency(self, ckpt_fields, canon_fields):
        current_set = set(canon_fields)
        ckpt_set = set(ckpt_fields)
        ckpt_alias_map = self.alias_to_key_mapping(ckpt_fields)
        current_alias_map = self.alias_to_key_mapping(canon_fields)
        for alias, old_key in ckpt_alias_map.items():
            if alias in current_alias_map:
                new_key = current_alias_map[alias]
                if old_key != new_key:
                    raise ValueError(f"alias ownership changed! Alias '{alias}' was mapped to '{old_key}' in checkpoint, but now maps to '{new_key}'.\n")
        if ckpt_set == current_set:
            return ckpt_fields
        
        elif ckpt_set.issubset(current_set):
            new_fields = current_set - ckpt_set
            raise ValueError(f"New fields in current canonical fields compared to checkpoint: {sorted(new_fields)}! This changes model n_states and needs partial model loading, not implemented yet. ")
            # return canon_fields
        else:
            # missing_in_current = ckpt_set - current_set
            # print(f"Fields missing in current canonical fields compared to checkpoint: {sorted(missing_in_current)}")
            return ckpt_fields
    def set_field_dictionary(self):
        if self.params.resuming:
            checkpoint = torch.load(self.params.checkpoint_path, map_location='cuda:{}'.format(self.local_rank) if torch.cuda.is_available() else torch.device('cpu'), weights_only=False)
            ckpt_fields = checkpoint.get('canonical_fields', CANONICAL_FIELDS)
            ckpt_cond_fields = checkpoint.get('canonical_cond_fields', CANONICAL_COND_FIELDS)        
            self.canonical_fields = self.check_field_dictionary_consistency(ckpt_fields, CANONICAL_FIELDS)
            self.canonical_cond_fields= self.check_field_dictionary_consistency(ckpt_cond_fields, CANONICAL_COND_FIELDS)
        else:
            self.canonical_fields = CANONICAL_FIELDS
            self.canonical_cond_fields = CANONICAL_COND_FIELDS

    def model_forward(self, inp, field_labels, bcs, opts: ForwardOptionsBase, pushforward=True):
        # Handles a forward pass through the model, either normal or autoregressive rollout.
        autoregressive = getattr(self.params, "autoregressive", False)
        if not autoregressive:
            output = self.model(inp, field_labels, bcs, opts)
            return output, None
        else:
            if self.global_rank==0:
                print("Autoregressive rollout", opts.leadtime, flush=True)
            # autoregressive rollout
            output, rollout_steps = autoregressive_rollout(self.model, inp, field_labels, bcs, opts, pushforward = pushforward)
            return output, rollout_steps

    def inference(self):
        if self.global_rank == 0:
            summary(self.model)
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
                ghost_list = data["ghost_info"]# List[GhostInfo], one per sample
                ghost_info = ghost_list[0]
                if self.group_size > 1:
                    assert graphdata.batch.unique().numel() == 1, f"expect batch size 1 when split graph but got {graphdata.batch.unique()}"
                dset_index, field_labels, field_labels_out, bcs = map(lambda x: x.to(self.device), [data[varname] for varname in ["dset_idx", "field_labels", "field_labels_out", "bcs"]])
            else: 
                inp, dset_index, field_labels, bcs, tar, leadtime = map(lambda x: x.to(self.device), [data[varname] for varname in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
                field_labels_out = field_labels
            supportdata = True if hasattr(self.params, 'supportdata') else False
            if supportdata and hasattr(data, "cond_input"):
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
                field_labels_out= field_labels_out,
                ghost_info = ghost_info if isgraph else None,
                )
                output, rollout_steps = self.model_forward(inp, field_labels, bcs, opts)
                if tar.ndim == 6: #B,T,C,D,H,W
                    if rollout_steps is None:
                        rollout_steps = leadtime.view(-1).long()
                    tar = tar[:, rollout_steps-1, :] # B,C,D,H,W
                update_loss_logs_inplace_eval(output, tar, graphdata if isgraph else None, logs, loss_dset_logs, loss_l1_dset_logs, loss_rmse_dset_logs, dset_type, seq_group=seq_group, returnBatchloss=True)
                
                if not isgraph and getattr(self.params, "log_ssim", False):
                    avg_ssim = get_ssim(output, tar, blockdict, self.global_rank, self.current_group, self.group_rank, self.group_size, self.device, self.valid_dataset, dset_index)
                    logs['valid_ssim'] += avg_ssim

                print(f"Batch {idx} Rank {self.global_rank} Valid Loss {logs['batch_nrmse'].item()} {dset_type}")
                
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

    def inference_step(self, d3d_solps=None, d3d_xgc=None, leadtime=None):
        #demo examples of inferencing on d3d samples from solps and xgc
        #FIXME: 1) how to convert state varaibles between solps-d3d and solps-kstar; 2) couple between d3d_solps and d3d_xgc? 

        self.model.eval()
        ###contruct solps d3d data loader###
        self.d3d_solps_loader, self.d3d_solps_dataset, _ = get_data_loader(
            self.params, d3d_solps,
            dist.is_initialized(), split='val',
            group_size= self.group_size, global_rank= self.global_rank, num_sp_groups=self.num_sequence_parallel_groups)
        d3d_solps_iter = iter(self.d3d_solps_loader)
        ###contruct xgc d3d data loader###
        self.d3d_xgc_loader, self.d3d_xgc_dataset, _ = get_data_loader(self.params, d3d_xgc,
                          dist.is_initialized(), split='val',
                          group_size= self.group_size, global_rank= self.global_rank, num_sp_groups=self.num_sequence_parallel_groups)
        d3d_xgc_iter = iter(self.d3d_xgc_loader)
    
        isample=3 #a random sample id should be smaller than dataset size
        if leadtime is None:
            d3d_solps_sample = next(d3d_solps_iter) 
        else:
            d3d_solps_sample = self.d3d_solps_dataset[(isample, leadtime)] 
            d3d_solps_sample = self.d3d_solps_loader.collate_fn([d3d_solps_sample]) #in batch format
        if leadtime is None:
            d3d_xgc_sample = next(d3d_xgc_iter) 
        else:
            d3d_xgc_sample = self.d3d_xgc_dataset[(isample, leadtime)] 
            d3d_xgc_sample = self.d3d_xgc_loader.collate_fn([d3d_xgc_sample])
        for case, data in zip(["SOLPS", "XGC"],[d3d_solps_sample, d3d_xgc_sample]):
            if "graph" in data:
                graphdata = data["graph"].to(self.device)
                tar = graphdata.y #[nnodes, C_tar] 
                leadtime = graphdata.leadtime #[nnodes, 1]
                dset_index, field_labels, field_labels_out, bcs = map(lambda x: x.to(self.device), [data[varname] for varname in ["dset_idx", "field_labels", "field_labels_out", "bcs"]])
            else: 
                inp, dset_index, field_labels, bcs, tar, leadtime = map(lambda x: x.to(self.device), [data[varname] for varname in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
                field_labels_out = field_labels
            supportdata = True if hasattr(self.params, 'supportdata') else False
            #FIXME: will change bakc later!!!!!
            if False: #supportdata:
                cond_input = data["cond_input"].to(self.device)
            else:
                cond_input = None

            cond_dict = {}
            try:
                cond_dict["labels"] = data["cond_field_labels"].to(self.device)
                cond_dict["fields"] = rearrange(data["cond_fields"].to(self.device), 'b t c d h w -> t b c d h w')
            except:
                pass
            dataset = self.d3d_xgc_dataset if case=="XGC" else self.d3d_solps_dataset
            blockdict = getattr(dataset.sub_dsets[dset_index[0]], "blockdict", None) 
            dset_type = dataset.sub_dsets[dset_index[0]].type 
            tkhead_name = dataset.sub_dsets[dset_index[0]].tkhead_name
            ##############################################################################################################
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
                seq_group = self.current_group if dset_type in dataset.DP_dsets else None
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
                residuals = output - tar
                if output.ndim == 2:
                    #[nnodes, C_tar] 
                    # Differentiate between log and accumulation losses
                    raw_loss = global_mean_pool(residuals.pow(2), graphdata.batch)/global_mean_pool(1e-7 + tar.pow(2), graphdata.batch) #B,C
                    raw_loss = raw_loss.sqrt().mean()
                    raw_rmse_loss = residuals.pow(2).mean(dim=0).sqrt().mean()
                else:
                    ###full resolution###
                    spatial_dims = tuple(range(output.ndim))[2:]
                    # Differentiate between log and accumulation losses
                    raw_loss = residuals.pow(2).mean(spatial_dims)/(1e-7+ tar.pow(2).mean(spatial_dims))
                    raw_loss = raw_loss.sqrt().mean()
                    raw_rmse_loss = residuals.pow(2).mean(spatial_dims).sqrt().mean()
                print(f"Prediction of {case}-D3D, rmse_loss:{raw_rmse_loss}; nrmse_loss {raw_loss}", flush=True)
                torch.save({"inp":inp, "target": tar, "output":output}, f"matey_{case}_leadtime_{leadtime[0].item()}.pt")
               
    def construct_graph_from_xgc(self, case_path, leadtime=None):
        x=xgc_base.xgc1(path=case_path)
        x.setup_mesh()
        with ad2.FileReader(os.path.join(case_path, "xgc.mesh.bp")) as f:
            delta_phi = f.read("delta_phi")
            wedge_angle = f.read("wedge_angle")
        num_wedges = int(2.0 * np.pi / wedge_angle)
        num_planes = int(wedge_angle/delta_phi)
        assert num_wedges == round(2.0 * np.pi / wedge_angle)
        assert num_planes == round(wedge_angle/delta_phi)
        """    
        with open(os.path.join(case_path, "fort.input.used"), encoding="utf-8", errors="ignore") as f:
                text = f.read()
        nphi = re.search(r"SML_NPHI_TOTAL\s*=\s*(\d+)", text)
        wedge = re.search(r"SML_WEDGE_N\s*=\s*(\d+)", text)
        num_planes = int(nphi.group(1)) if nphi else None
        num_wedges = int(wedge.group(1)) if wedge else None
        delta_phi = 2.0 * np.pi / num_wedges / num_planes if num_planes else None
        """
        case_state = {
                "count": -1, #number of snapshots, filler for now
                "time_indices": [], 
                "xgc_files": [],
                "f3d_flag": True, #if xgc case is 3d or not
                "dt": 10, #case dependent
                "num_wedges": num_wedges,
                "num_planes": num_planes,
                "delta_phi": delta_phi,
            }
        helper = GraphXGCDataset.__new__(GraphXGCDataset)
        #pos: [N, 3], edge_index: [2, E], edge_attr: [E, 4]
        pos, edge_index, edge_attr, num_nodes = helper.get_nodepos_edgeindexattr(x, case_state, case_path)

        node_features = [pos[:, 0], pos[:, 1], pos[:, 2]]  # r, z, phi(phi=0 for 2d)
        """"
        for varname in self.field_names_out[:-1]:
            # ['e_den', 'e_T_perp', 'e_T_para', 'e_u_para', 'i_T_perp','i_T_para', 'i_u_para', 'dpot']
            variable = x.read_one_ad2_var(filename, varname)
            assert variable.ndim==2 and variable.shape==(num_planes, num_nodes), f"{variable.shape}"
            print(f"{varname}, {variable.shape}", flush=True)
            variable = variable.reshape(-1)
            node_features.append(torch.as_tensor(variable, dtype=torch.float32))
        #f"xgc.3d.{t:05d}.bp" for dpot
        print(filename, flush=True)
        dpot = x.read_one_ad2_var(filename.replace('.f3d.','.3d.'), "dpot")
        assert dpot.ndim==2 and dpot.shape==(num_planes, num_nodes), f"{dpot.shape}"
        print("dpot.shape", dpot.shape, flush=True)
        dpot = dpot.reshape(-1) #order "C" - last axis index changing fastest
        node_features.append(torch.as_tensor(dpot, dtype=torch.float32))
        """
        ###################################################################################################################################################
        # FIXME: Instead of loading data from xgc outputs (see above), we generate synthetic node features; 
        # @Michael, please replace those with the reconstructured solutions
        field_names_out = ['e_den', 'e_T_perp', 'e_T_para', 'e_u_para', 'i_T_perp','i_T_para', 'i_u_para', 'dpot']
        field_names = helper.field_names
        for varname in field_names_out:
            variable = torch.rand(num_nodes*num_planes, dtype=torch.float32) #synthetic features between [0,1)
            node_features.append(variable) 
        feature = torch.stack(node_features, dim=1)
        assert feature.shape==(num_nodes*num_planes, len(field_names_out)+3), f"{feature.shape,(num_nodes*num_planes, len(field_names_out)+3)}"
        #repeat node feature 3 times to represent the 3-step input
        x_seq = torch.stack([feature, feature, feature], dim=0).permute(1, 0, 2) #shape: [nsteps_input, N, C] --> [N, nsteps_input, C]
        indices_y=[field_names.index(x) for x in field_names_out]
        y_state = feature[:, indices_y] #output field_name_out only
        data = Data(x=x_seq, y=y_state, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        data.leadtime = torch.tensor([1]).to(torch.float32) if leadtime is None else leadtime #number of snapshots ahead
        data.group = case_path #need to check if consistent with original graph data class, but won't affect inference accuracy
        data.t = 0 #int(t)
        data.dt = 10 #case_state["dt"] 
        data.t0 = -1 #start time
        data.target_t = -1 #target time
        ###################################################################################################################################################
        field_labels = torch.tensor(self.train_dataset.subset_dict['graphxgc'])
        field_labels_out = field_labels[indices_y]
        ###################################################################################################################################################
        return my_collate([{"graph": data, "bcs": torch.as_tensor([0, 0]), "field_labels": field_labels, "field_labels_out":field_labels_out}]) #in batch format
    
    def get_minmax(self):
        self.train_dataset.sub_dsets[-1]._minmax_features()
        max_nodefeat = self.train_dataset.sub_dsets[-1].max_nodefeat["n565pe_PT_xgc1_d3d_adjust_flow2_for_C"]
        min_nodefeat = self.train_dataset.sub_dsets[-1].min_nodefeat["n565pe_PT_xgc1_d3d_adjust_flow2_for_C"]
        return min_nodefeat, max_nodefeat

         
    def inference_step_newxgcmesh(self, d3d_xgc_dir=None, leadtime=None):

        self.model.eval()

        case = "XGC"
        data = self.construct_graph_from_xgc(d3d_xgc_dir, leadtime=leadtime)
           
        graphdata = data["graph"].to(self.device)
        tar = graphdata.y #[nnodes, C_tar] 
        leadtime = graphdata.leadtime.reshape(-1,1)
        field_labels, field_labels_out, bcs = map(lambda x: x.to(self.device), [data[varname] for varname in ["field_labels", "field_labels_out", "bcs"]])
       
        supportdata = False
        cond_input = None
        cond_dict = {}
        blockdict =  None
        tkhead_name = "tk-graph"
        ##############################################################################################################
        with torch.no_grad():
            tar = tar.to(self.device)
            imod = self.params.hierarchical["nlevels"]-1 if hasattr(self.params, "hierarchical") else 0
            isgraph = True
            inp = graphdata
            imod_bottom = imod
           
            seq_group =  None
            print(f"Rank {self.global_rank} input shape {inp.shape if not isgraph else inp}, dset_type 'graphxgc'", flush=True)
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
            residuals = output - tar
            if output.ndim == 2:
                #[nnodes, C_tar] 
                # Differentiate between log and accumulation losses
                raw_loss = global_mean_pool(residuals.pow(2), graphdata.batch)/global_mean_pool(1e-7 + tar.pow(2), graphdata.batch) #B,C
                raw_loss = raw_loss.sqrt().mean()
                raw_rmse_loss = residuals.pow(2).mean(dim=0).sqrt().mean()
            else:
                ###full resolution###
                spatial_dims = tuple(range(output.ndim))[2:]
                # Differentiate between log and accumulation losses
                raw_loss = residuals.pow(2).mean(spatial_dims)/(1e-7+ tar.pow(2).mean(spatial_dims))
                raw_loss = raw_loss.sqrt().mean()
                raw_rmse_loss = residuals.pow(2).mean(spatial_dims).sqrt().mean()
            print(f"Prediction of {case}-D3D, rmse_loss:{raw_rmse_loss}; nrmse_loss {raw_loss}", flush=True)
            torch.save({"inp":inp, "target": tar, "output":output}, f"matey_{case}_leadtime_{leadtime[0].item()}.pt")
            
            min_nodefeat, max_nodefeat = self.get_minmax()
            #FIXME: @Michael
            print("Since we have no min/max features of the new case, use n565pe_PT_xgc1_d3d_adjust_flow2_for_C?", min_nodefeat, max_nodefeat, flush=True)
    

