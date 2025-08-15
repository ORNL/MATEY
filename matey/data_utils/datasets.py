import torch
import torch.nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import os
try:
    from mixed_dset_sampler import MultisetSampler
    from hdf5_datasets import *
    from netcdf_datasets import *
    from hdf5_3Ddatasets import *
    from blasnet_3Ddatasets import *
    from thewell_datasets import *
    from binary_3DSSTdatasets import *
except ImportError:
    from .mixed_dset_sampler import MultisetSampler
    from .hdf5_datasets import *
    from .netcdf_datasets import *
    from .exodus_datasets import *
    from .hdf5_3Ddatasets import *
    from .blasnet_3Ddatasets import *
    from .thewell_datasets import *
    from .binary_3DSSTdatasets import *
import os
import glob

broken_paths = []
# IF YOU ADD A NEW DSET MAKE SURE TO UPDATE THIS MAPPING SO MIXED DSET KNOWS HOW TO USE IT
DSET_NAME_TO_OBJECT = {
    ##PDEBench
    'swe': SWEDataset,
    'incompNS': IncompNSDataset,
    'diffre2d': DiffRe2DDataset,
    'compNS': CompNSDataset,
    ##Matt
    'thermalcollision2d': CollisionDataset,
    ##Doug
    'liquidMetalMHD': MHDDataset,
    ##JHU
    "isotropic1024fine": isotropic1024Dataset,
    #TaylorGreen
    "taylorgreen": TaylorGreen,
    ##BLASNET
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
    }

def get_data_loader(params, paths, distributed, split='train', rank=0, group_rank=0, group_size=1, train_offset=0, num_replicas=None):
    #rank: SP group ID, used for sample index
    #group_rank: local rank in the SP group
    # paths, types, include_string = zip(*paths)

    leadtime_max=1 #finetuning higher priority
    if hasattr(params, 'leadtime_max_finetuning'):
        leadtime_max = params.leadtime_max_finetuning
    elif hasattr(params, 'leadtime_max'):
        leadtime_max = params.leadtime_max

    dataset = MixedDataset(paths, n_steps=params.n_steps, train_val_test=params.train_val_test if hasattr(params, 'train_val_test')  else None, split=split,
                            tie_fields=params.tie_fields, use_all_fields=params.use_all_fields, enforce_max_steps=params.enforce_max_steps,
                            train_offset=train_offset, tokenizer_heads=params.tokenizer_heads,
                            dt = params.dt if hasattr(params,'dt') else 1,
                            leadtime_max=leadtime_max, #params.leadtime_max if hasattr(params, 'leadtime_max') else 1,
                            refine_ratio=params.refine_ratio if hasattr(params, 'refine_ratio')  else None,
                            gammaref=params.gammaref if hasattr(params, 'gammaref')  else None,
                            SR_ratio=params.SR_ratio if hasattr(params, 'SR_ratio') else None,
                            data_augmentation=params.data_augmentation if hasattr(params, 'data_augmentation') else False,
                            group_id=rank, group_rank=group_rank, group_size=group_size)
    seed = torch.random.seed() if 'train'==split else 0
    if distributed:
        base_sampler = DistributedSampler
    else:
        base_sampler = RandomSampler
    sampler = MultisetSampler(dataset, base_sampler, params.batch_size,
                               distributed=distributed, max_samples=params.epoch_size,
                               rank=rank, num_replicas=num_replicas)
    # sampler = DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            prefetch_factor=2,
                            shuffle=False, #(sampler is None),
                            sampler=sampler, # Since validation is on a subset, use a fixed random subset,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available(), 
                            persistent_workers=True, #ask dataloaders not destroyed after each epoch
                            )
    return dataloader, dataset, sampler


class MixedDataset(Dataset):
    def __init__(self, path_list=[], n_steps=1, dt=1, leadtime_max=1, train_val_test=(.8, .1, .1),
                  split='train', tie_fields=True, use_all_fields=True, extended_names=False,
                  enforce_max_steps=False, train_offset=0, tokenizer_heads=None, refine_ratio=None, gammaref=None, SR_ratio=None,
                  data_augmentation=None, group_id=0, group_rank=0, group_size=1):
        super().__init__()
        # Global dicts used by Mixed DSET.
        self.train_offset = train_offset
        try:
            self.path_list, self.type_list, self.include_string = zip(*path_list)
            self.tkhead_name=[tokenizer_heads[0]["head_name"] for _ in self.path_list]
            print("Warning: no tkhead_type provided in config for datasets; we will use the first tokenizer_heads: %s"%(" ").join(self.tkhead_name))
        except:
            self.path_list, self.type_list, self.include_string, self.tkhead_name = zip(*path_list)

        self.tie_fields = tie_fields
        self.extended_names = extended_names
        self.split = split
        self.sub_dsets = []
        self.offsets = [0]
        self.train_val_test = train_val_test
        self.use_all_fields = use_all_fields

        if refine_ratio is not None and gammaref is not None:
            print("Warning: both refine_ratio and gammaref: %.2f, %.2f are provided in config"%(refine_ratio, gammaref))
            print("We will use gammaref value for adaptivity")
            refine_ratio = None

        for dset, path, include_string, tkhead_name in zip(self.type_list, self.path_list, self.include_string, self.tkhead_name):
            subdset = DSET_NAME_TO_OBJECT[dset](path, include_string, n_steps=n_steps,
                                                 dt=dt, leadtime_max = leadtime_max, train_val_test=train_val_test, split=split,
                                                 tokenizer_heads=tokenizer_heads, refine_ratio=refine_ratio, gammaref=gammaref, tkhead_name=tkhead_name, SR_ratio=SR_ratio,
                                                 data_augmentation=data_augmentation,group_id=group_id, group_rank=group_rank, group_size=group_size)
            # Check to make sure our dataset actually exists with these settings
            try:
                len(subdset)
            except ValueError:
                raise ValueError(f'Dataset {path} is empty. Check that n_steps < trajectory_length in file.')
            self.sub_dsets.append(subdset)
            self.offsets.append(self.offsets[-1]+len(self.sub_dsets[-1]))
        self.offsets[0] = -1

        self.subset_dict = self._build_subset_dict()

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
        assert len(variables) == 4 or len(variables) == 5
        if len(variables)==4:
            x, bcs, y, leadtime = variables
            return x, dset_idx, torch.tensor(self.subset_dict[self.sub_dsets[dset_idx].get_name()]), bcs, y, leadtime
        elif len(variables)==5:
            x, bcs, y, refineind, leadtime = variables
            return x, dset_idx, torch.tensor(self.subset_dict[self.sub_dsets[dset_idx].get_name()]), bcs, y, refineind, leadtime

    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])
