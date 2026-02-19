# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import torch.nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import default_collate
import os
from .mixed_dset_batchsampler import MultisetBatchSampler
from .hdf5_datasets import *
from .netcdf_datasets import *
from .exodus_datasets import *
from .hdf5_3Ddatasets import *
from .blastnet_3Ddatasets import *
from .thewell_datasets import *
from .binary_3DSSTdatasets import *
from .graph_datasets import *
from .flow3d_datasets import *
import os
from torch_geometric.data import Data as GraphData, Batch
import warnings

broken_paths = []
# IF YOU ADD A NEW DSET MAKE SURE TO UPDATE THIS MAPPING SO MIXED DSET KNOWS HOW TO USE IT
DSET_NAME_TO_OBJECT = {
    ##PDEBench
    'swe': SWEDataset,
    'incompNS': IncompNSDataset,
    'diffre2d': DiffRe2DDataset,
    'compNS': CompNSDataset,
    'compNS128': CompNSDataset128,
    'compNS512': CompNSDataset512,
    ##Matt
    'thermalcollision2d': CollisionDataset,
    ##Doug
    'liquidMetalMHD': MHDDataset,
    ##SOLPS-ITER
    'SOLPS2D' :SOLPSDataset,
    ##JHU
    "isotropic1024fine": isotropic1024Dataset,
    #TaylorGreen
    "taylorgreen": TaylorGreen,
    ##BLASTNET
    "h2vitairli": H2vitairliDataset,
    "h2jetRe": H2jetDataset,
    "pass-fhit": FHITsnapshots,
    "hit-dns": HITDNSsnapshots,
    "SR": SR_Benchmark,
    ##thewell
    "acousticmaze": acoustic_scattering_maze,
    "acousticinclu":acoustic_scattering_inclusions,
    "acousticdiscont":acoustic_scattering_discontinuous,
    "activematter":active_matter,
    "convrsg":convective_envelope_rsg,
    "euleropen":euler_multi_quadrants_openBC,
    "eulerperiodic":euler_multi_quadrants_periodicBC,
    "helmholtzstaircase":helmholtz_staircase,
    "MHD64":MHD_64,
    "MHD256":MHD_256,
    "grayscottreactdiff":gray_scott_reaction_diffusion,
    "planetswe":planetswe,
    "postneutronstarmerger":post_neutron_star_merger,
    "rayleighbenard":rayleigh_benard,
    "rayleightaylor":rayleigh_taylor_instability,
    "shearflow":shear_flow,
    "supernova64":supernova_explosion_64,
    "supernova128":supernova_explosion_128,
    "turbgravcool":turbulence_gravity_cooling,
    "turbradlayer2D":turbulent_radiative_layer_2D,
    "turbradlayer3D":turbulent_radiative_layer_3D,
    "viscoelastic":viscoelastic_instability,
    ##SST
    "sstF4R32": sstF4R32Dataset,
    ##Flow3D
    "flow3d": Flow3D_Object,
    ##deepmindgraphnet
    "meshgraphnetairfoil": MeshGraphNetsAirfoilDataset,
    }

def get_data_loader(params, paths, distributed, split='train', global_rank=0, num_sp_groups=None, group_size=1, train_offset=0, multiepoch_loader=False):
    #global_rank: global_rank
    #num_sp_groups: number of SP groups
    #group_size: number of ranks in each group
    #paths, types, include_string = zip(*paths)

    leadtime_max=-1 #finetuning higher priority
    if hasattr(params, 'leadtime_max_finetuning'):
        leadtime_max = params.leadtime_max_finetuning
    elif hasattr(params, 'leadtime_max'):
        leadtime_max = params.leadtime_max

    dataset = MixedDataset(paths, n_steps=params.n_steps, train_val_test=getattr(params, 'train_val_test', None), split=split,
                            tie_fields=params.tie_fields, use_all_fields=params.use_all_fields, enforce_max_steps=params.enforce_max_steps,
                            train_offset=train_offset, tokenizer_heads=params.tokenizer_heads,
                            dt = getattr(params,'dt', 1),
                            leadtime_max=max(leadtime_max, 0),
                            SR_ratio=getattr(params, 'SR_ratio', None),
                            supportdata= getattr(params, "supportdata", None),
                            global_rank=global_rank, group_size=group_size, DP_dsets=getattr(params, "DP_dsets", ["isotropic1024fine", "taylorgreen"]))
    seed = torch.random.seed() if 'train'==split else 0
    if distributed:
        base_sampler = DistributedSampler
    else:
        base_sampler = RandomSampler
    sampler = MultisetBatchSampler(dataset, base_sampler, params.batch_size,
                               distributed=distributed, max_samples=params.epoch_size,
                               global_rank=global_rank, group_size=group_size, num_sp_groups=num_sp_groups, ordered_sampling=getattr(params, "ordered_sampling", True))
    # sampler = DistributedSampler(dataset) if distributed else None
    if multiepoch_loader:
        if split != 'train':
            warnings.warn("Warning: Using MultiEpochsDataLoader for validation can silently desynchronize " \
                  "sampler RNG state if the number of consumed samples differs from the number of yielded samples. Falling back to default DataLoader for valid.")
            loader = DataLoader
        else:
            loader = MultiEpochsDataLoader
    else:
        loader = DataLoader
    dataloader = loader(dataset,
                        num_workers=params.num_data_workers,
                        #prefetch_factor=2,
                        batch_sampler=sampler,
                        pin_memory=torch.cuda.is_available(), 
                        persistent_workers=True, #ask dataloaders not destroyed after each epoch
                        collate_fn=my_collate,
                        )
    return dataloader, dataset, sampler


class MixedDataset(Dataset):
    def __init__(self, path_list=[], n_steps=1, dt=1, leadtime_max=0, supportdata=None, train_val_test=(.8, .1, .1),
                  split='train', tie_fields=True, use_all_fields=True, extended_names=False,
                  enforce_max_steps=False, train_offset=0, tokenizer_heads=None, SR_ratio=None,
                  global_rank=0, group_size=1, DP_dsets=["isotropic1024fine", "taylorgreen"]):
        super().__init__()
        # Global dicts used by Mixed DSET.
        self.train_offset = train_offset
        try:
            self.path_list, self.type_list, self.include_string = zip(*path_list)
            self.tkhead_name=[tokenizer_heads[0]["head_name"] for _ in self.path_list]
            warnings.warn("Warning: no tkhead_type provided in config for datasets; we will use the first tokenizer_heads: %s"%(" ").join(self.tkhead_name))
        except:
            self.path_list, self.type_list, self.include_string, self.tkhead_name = zip(*path_list)

        self.tie_fields = tie_fields
        self.extended_names = extended_names
        self.split = split
        self.sub_dsets = []
        self.offsets = [0]
        self.train_val_test = train_val_test
        self.use_all_fields = use_all_fields

        self.DP_dsets= [subset for subset in DSET_NAME_TO_OBJECT.keys() if "graph" not in subset] if DP_dsets=="ALL" else DP_dsets #datasets that use distributed reading and each rank get a local subplit
        if len(self.DP_dsets)==0 and group_size>1:
            warnings.warn(
                "SP group is set, but no DP_dsets is defined. As a result, no SP will be used."
                "Please verify if this is intentional."
            )
        for dset, path, include_string, tkhead_name in zip(self.type_list, self.path_list, self.include_string, self.tkhead_name):
            if dset in self.DP_dsets:
                #For every group with group_size ranks, they read the subparts from the same sample
                group_id=global_rank//group_size #id of each group, e.g., 0,1,2,3 for 4 sp groups
                data_rank=global_rank%group_size #local rank inside each SP group, e.g., 0,1,2,...,7 if group_size=8 (assigning 8 GPUs to load the same sample)
                datagroupsize=group_size
            else:
                group_id=global_rank
                data_rank=0
                datagroupsize=1

            subdset = DSET_NAME_TO_OBJECT[dset](path, include_string, n_steps=n_steps,
                                                 dt=dt, leadtime_max = leadtime_max, supportdata = supportdata, train_val_test=train_val_test, split=split,
                                                 tokenizer_heads=tokenizer_heads, tkhead_name=tkhead_name, SR_ratio=SR_ratio,
                                                 group_id=group_id, group_rank=data_rank, group_size=datagroupsize)
            # Check to make sure our dataset actually exists with these settings
            try:
                len(subdset)
            except ValueError:
                raise ValueError(f'Dataset {path} is empty. Check that n_steps < trajectory_length in file.')
            self.sub_dsets.append(subdset)
            self.offsets.append(self.offsets[-1]+len(self.sub_dsets[-1]))
        self.offsets[0] = -1

        self.subset_dict = self._build_subset_dict()
        self.subset_cond_dict = self._build_subset_cond_dict()

    def get_state_names(self):
        name_list = []
        if self.use_all_fields:
            for name, dset in DSET_NAME_TO_OBJECT.items():
                field_names = dset.field_names
                name_list += field_names
            return name_list
        else:
            visited = set()
            for dset in self.sub_dsets:
                    name = dset.get_name() # Could use extended names here
                    if not name in visited:
                        visited.add(name)
                        name_list.append(dset.field_names)
        return [f for fl in name_list for f in fl] # Flatten the names

    def _build_subset_dict(self):
        # Maps fields to subsets of variables
        if self.tie_fields: # Hardcoded, but seems less effective anyway
            subset_dict = {
                        'swe': [3],
                        'incompNS': [0, 1, 2],
                        'compNS': [0, 1, 2, 3],
                        'diffre2d': [4, 5],
                        'thermalcollision2d':[0, 1, 2, 6],
                        }
        elif self.use_all_fields:
            cur_max = 0
            subset_dict = {}
            for name, dset in DSET_NAME_TO_OBJECT.items():
                field_names = dset.field_names
                subset_dict[name] = list(range(cur_max, cur_max + len(field_names)))
                cur_max += len(field_names)
        else:
            subset_dict = {}
            cur_max = self.train_offset
            for dset in self.sub_dsets:
                name = dset.get_name(self.extended_names)
                if not name in subset_dict:
                    subset_dict[name] = list(range(cur_max, cur_max + len(dset.field_names)))
                    cur_max += len(dset.field_names)
        return subset_dict

    def _build_subset_cond_dict(self):
        # Maps conditional fields to subsets of conditional variables
        if self.tie_fields: # Hardcoded, but seems less effective anyway
            raise ValueError("tie_fields is not implemented for _build_subset_cond_dict; use use_all_fields")
        elif self.use_all_fields:
            cur_max = 0
            subset_dict = {}
            for name, dset in DSET_NAME_TO_OBJECT.items():
                try:
                    cond_field_names = dset.cond_field_names
                    subset_dict[name] = list(range(cur_max, cur_max + len(cond_field_names)))
                    cur_max += len(cond_field_names)
                except: #no conditional fields for dset
                    continue
        else:
            raise ValueError("currently only support use_all_fields for _build_subset_cond_dict")
        return subset_dict

    def __getitem__(self, index):

        if hasattr(index, '__len__') and len(index)==2:
            leadtime = index[1]
            index = index[0]
            dset_idx = np.searchsorted(self.offsets, index, side='right')-1 #which dataset are we are on
            local_idx = index - max(self.offsets[dset_idx], 0) #which sample inside the dataset dset_idx
            local_idx = [local_idx, leadtime]
        else:
            dset_idx = np.searchsorted(self.offsets, index, side='right')-1 #which dataset are we are on
            local_idx = index - max(self.offsets[dset_idx], 0) #which sample inside the dataset dset_idx
            
        variables = self.sub_dsets[dset_idx][local_idx]
        #assuming variables in order: 
        #if cond_field_names and cond_input both defined (no such case at the moment):
        #   x, bcs, y, leadtime, cond_fields, cond_input
        #if cond_field_names defined:
        #   x, bcs, y, leadtime, cond_fields
        #if cond input defined:
        #   x, bcs, y, leadtime, cond_input
        #else:
        #   x, bcs, y, leadtime

        #OR
        #graph, bcs         

        datasamples={} 

        datasamples["field_labels"] = torch.tensor(self.subset_dict[self.sub_dsets[dset_idx].get_name()])
        datasamples["dset_idx"] = dset_idx

        if "graph" in variables:
            """
            # data.x = x_seq #[nsteps_input, N, F] 
            # data.y = y_state #[N, 3] -> (vx, vy, p)
            # data.leadtime = leadtime
            """
            datasamples["graph"] = variables["graph"]
            datasamples["bcs"] = variables["bcs"]
            datasamples["field_labels_out"] = datasamples["field_labels"][variables["field_labels_out"]]
            return datasamples

        else:
            datasamples["input"] = variables["x"]
            datasamples["label"] = variables["y"]
            datasamples["bcs"] = variables["bcs"]
            datasamples["leadtime"] = variables["leadtime"]

            if "cond_fields" in variables:
                assert hasattr(self.sub_dsets[dset_idx], "cond_field_names")
                datasamples["cond_field_labels"] = torch.tensor(self.subset_cond_dict[self.sub_dsets[dset_idx].get_name()])
                datasamples["cond_fields"] = variables["cond_fields"]

            if "cond_input" in variables:
                datasamples["cond_input"] = variables["cond_input"]
        
            return datasamples

    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])
    
class MultiEpochsDataLoader(torch.utils.data.DataLoader):
# Taken from: https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.len_samples = self.sampler.max_samples
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(self.len_samples):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def my_collate(batch):
    batch_new = {}
    for key in batch[0].keys():
        objs = [b[key] for b in batch]
        if key == "graph":
            batch_new[key] = Batch.from_data_list(objs)
        else:
            batch_new[key] = default_collate(objs)
    return batch_new
