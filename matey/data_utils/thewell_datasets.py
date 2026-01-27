# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import numpy as np
import os
from torch.utils.data import Dataset
import h5py
import glob
import yaml
from ..utils import getblocksplitstat

"""
WELL_DATASETS = [
    "acoustic_scattering_maze",
    "acoustic_scattering_inclusions",
    "acoustic_scattering_discontinuous",
    "active_matter",
    "convective_envelope_rsg",
    "euler_multi_quadrants_openBC",
    "euler_multi_quadrants_periodicBC",
    "helmholtz_staircase",
    "MHD_64",
    "MHD_256",
    "gray_scott_reaction_diffusion",
    "planetswe",
    "post_neutron_star_merger",
    "rayleigh_benard",
    "rayleigh_taylor_instability",
    "shear_flow",
    "supernova_explosion_64",
    "supernova_explosion_128",
    "turbulence_gravity_cooling",
    "turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_3D",
    "viscoelastic_instability",
]
"""
class TheWellDataset(Dataset):
    """
    from:https://github.com/PolymathicAI/the_well/tree/master
    Base class for data loaders. Returns data in T x C X D x H x W format.
    Args:
        path (str): Path to directory of HDF5 files
        include_string (str): Only include files with this string in name [Note: it is not used and 
                                kept here onlyto be consistent with other datasets]
        n_steps (int): Number of steps to include in the input of each sample
        dt (int): Time step between samples
        split (str): train/val/test split
        train_val_test (tuple): Percent of data to use for train/val/test
        split_level (str): 'sample' or 'file' - whether to split by samples within a file
                        (useful for data segmented by parameters) or file (mostly INS right now)
        leadtime_max: when >0, future solution solution prediction, tar is a solution at the lead time;
                      when =0, self-supervised learning and tar is None
    """
    def __init__(self, path, include_string='', n_steps=1, dt=1, leadtime_max=0, supportdata=None, split='train', 
                 train_val_test=None, extra_specific=False, tokenizer_heads=None, tkhead_name=None, SR_ratio=None,
                 group_id=0, group_rank=0, group_size=1):
        super().__init__()

        np.random.seed(2024)

        self.path = path
        self.split = split
        self.extra_specific = extra_specific # Whether to use parameters in name
        
        self.dt = 1
        self.leadtime_max = leadtime_max 
        self.nsteps_input = n_steps
        self.train_val_test = train_val_test
        self.partition = {'train': 0, 'val': 1, 'test': 2}[split]
        self.scalar_names, self.vector_names, self.tensor_names, self.type, self.cubsizes, self.spatial_dims, self.split_level = self._specifics()
        self.constantintimevar=[]
        self.constantintimetrajvar=[]
        if self.type in ['acousticdiscont','acousticinclu','acousticmaze']:
            self.constantintimevar=['density', 'speed_of_sound']
        if self.type in ['helmholtzstaircase']:
            self.constantintimetrajvar=['mask']
        self.time_skip=0 #inconsistent IC leads to training instability?
        self._get_directory_stats(path)
        self.title = self.type

        self.tokenizer_heads = tokenizer_heads
        self.tkhead_name=tkhead_name

        self.group_id=group_id
        self.group_rank=group_rank
        self.group_size=group_size

        if len(self.cubsizes)==3:
                H, W, D = self.cubsizes #x,y,z
        else:
            H, W= self.cubsizes #x,y
            D=1    
        self.blockdict =  getblocksplitstat(self.group_rank, self.group_size, D, H, W)
    
    def get_name(self):
        return self.type

    def get_name(self, full_name=False):
        if full_name:
            return self.subname + '_' + self.type
        else:
            return self.type

    def more_specific_title(self, type, path, include_string):
        """
        Override this to add more info to the dataset name
        """
        return type

    @staticmethod
    def _specifics():
        # Sets self.field_names, self.dataset_type
        raise NotImplementedError # Per dset

    def get_per_file_dsets(self):
        return [self]


    def _get_specific_stats(self, _f):
        trajectories = int(_f.attrs["n_trajectories"])
        steps = _f["dimensions"]["time"].shape[-1]
        return trajectories, steps

    def _get_specific_bcs(self, _f):
        raise NotImplementedError # Per dset

    def _reconstruct_sample(self, file, leadtime, time_idx, n_steps):
        raise NotImplementedError # Per dset - should be (x=(-history:local_idx+dt) so that get_item can split into x, y

    def _get_directory_stats(self, path):
        self.files_paths = glob.glob(path + "/*.h5") + glob.glob(path + "/*.hdf5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        self.file_lens = []
        self.file_steps = [] #total sample size for each file
        self.file_insteps = [] #input history length 
        self.file_samples = []
        self.split_offsets = []
        self.offsets = [0]
        file_paths = []
        for file in self.files_paths:
            file_paths.append(file)
            try:
                with h5py.File(file, 'r') as _f:
                    samples, steps = self._get_specific_stats(_f)
                steps -= self.time_skip
                if steps-self.nsteps_input-(self.dt-1) < 1:
                    raise ValueError('WARNING: File {} has {} steps, but n_steps is {}. Setting file steps = max allowable.'.format(file, steps, self.nsteps_input))
                file_insteps = self.nsteps_input
                self.file_lens.append(steps)
                self.file_insteps.append(file_insteps)
                self.file_steps.append(steps-file_insteps-(self.dt-1))
                self.file_samples.append(samples)
                self.offsets.append(self.offsets[-1]+self.file_steps[-1]*self.file_samples[-1])
            except:
                print('WARNING: Failed to open file {}. Continuing without it.'.format(file))
                raise RuntimeError('Failed to open file {}'.format(file))
        self.files_paths = file_paths
        self.offsets[0] = -1
        self.datasets = [None for _ in self.files_paths]
        #get split for partition
        """
        train/valid/test splits have been done, see directory structure
        ├── acoustic_scattering_discontinuous
        │   └── data
        │       ├── test
        │       │   ├── acoustic_scattering_discontinuous_chunk_18.hdf5
        │       │   └── acoustic_scattering_discontinuous_chunk_19.hdf5
        │       ├── train
        │       │   ├── acoustic_scattering_discontinuous_chunk_0.hdf5
        │       │   ├── ...
        │       └── valid
        │           ├── acoustic_scattering_discontinuous_chunk_16.hdf5
        │           └── acoustic_scattering_discontinuous_chunk_17.hdf5
        """
        self.split_offset = 0
        self.len = self.offsets[-1]
        #get mean and std for variables
        self._load_stats(os.path.join(path +"/../../", "stats.yaml"))

    def _loaddata_file(self, file_ind):
        self.datasets[file_ind] = h5py.File(self.files_paths[file_ind], 'r')

    def _load_stats(self, normalization_path):
        with open(normalization_path, mode="r") as f:
            stats = yaml.safe_load(f)
        self.means = {field: val for field, val in stats["mean"].items()}
        self.stds  = {field: val for field, val in stats["std"].items()}

    def __getitem__(self, index):
        if hasattr(index, '__len__') and len(index)==2:
            leadtime=torch.tensor([index[1]], dtype=torch.int)
            index = index[0]
        else:
            leadtime=None

        #go to the split (i.e., train, val, or test) collection
        index = index + self.split_offset
        file_idx = int(np.searchsorted(self.offsets, index, side='right')-1)

        local_idx  = index - max(self.offsets[file_idx], 0) # First offset is -1
        sample_idx = local_idx // self.file_steps[file_idx]
        time_idx   = local_idx % self.file_steps[file_idx]

        nsteps_input = self.file_insteps[file_idx]


        #open image file
        if self.datasets[file_idx] is None:
            self._loaddata_file(file_idx)

        if leadtime is None:
            #generate a random leadtime uniformly sampled from [1, self.leadtime_max]
            leadtime = torch.tensor([0])
            if self.leadtime_max>0:
                leadtime = torch.randint(1, min(self.leadtime_max+1, self.file_lens[file_idx]-time_idx-nsteps_input+1), (1,))
        else:
            leadtime = min(leadtime, self.file_lens[file_idx]-time_idx-nsteps_input)
        assert time_idx + nsteps_input + leadtime <= self.file_lens[file_idx]

        #try:
        if True:
            trajectory, leadtime = self._reconstruct_sample(self.datasets[file_idx], sample_idx, leadtime, time_idx, nsteps_input)
            bcs = self._get_specific_bcs(self.datasets[file_idx])
        #except:
        #    raise RuntimeError(f'Failed to reconstruct sample for file {self.files_paths[file_idx]} trajectory {sample_idx} time {time_idx}')

        #T,C,H,W ==> T,C,D(=1),H,W for compatibility with 3D
        if self.spatial_dims==2:
            trajectory=np.expand_dims(trajectory, axis=2)

        for tk in self.tokenizer_heads:
            if tk["head_name"] == self.tkhead_name:
                patch_size = tk["patch_size"]
                break
        #temporary treatment for non dividable res
        for ips, ps in enumerate(patch_size[-1]):
            nres_=trajectory.shape[2+ips]%ps
            if nres_>0:
                if torch.bernoulli(torch.Tensor([0.5]))==1:
                    trajectory=np.delete(trajectory, [-(ires+1) for ires in range(nres_)], axis=2+ips)
                else:
                    trajectory = np.append(trajectory, np.take(trajectory, [-(ires+1) for ires in range(ps-nres_)], axis=2+ips), axis=2+ips)

        return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1], leadtime

    def __len__(self):
        return self.len
 
    def _readdata(self, file, sample_idx, time_start, time_end):
        #get solution history
        comb_data=[]
        # scalar variables
        for varname in self.scalar_names: 
            if varname in self.constantintimetrajvar:
                var_ = file["t0_fields"][varname][...]
                var_ = np.repeat(var_[None, ...], time_end-time_start, axis=0)
            elif varname in self.constantintimevar:
                var_ = file["t0_fields"][varname][sample_idx,...]
                var_ = np.repeat(var_[None, ...], time_end-time_start, axis=0)
            else:
                var_ = file["t0_fields"][varname][sample_idx,time_start:time_end,:]
            if varname in self.means:
                var_ = (var_-self.means[varname])/max(self.stds[varname], 1e-4)
            comb_data.append(var_)
        # vector variables
        for varname in self.vector_names:  
            var_ = file["t1_fields"][varname][sample_idx,time_start:time_end,:] 
            for idim in range(self.spatial_dims):
                comb_data.append((var_[...,idim]-self.means[varname][idim])/max(self.stds[varname][idim], 1e-4))
                
        # tensor variables 
        for varname in self.tensor_names: 
            var_ = file["t2_fields"][varname][sample_idx,time_start:time_end,:] #tensor
            for idim in range(self.spatial_dims):
                for jdim in range(self.spatial_dims):
                    comb_data.append((var_[...,idim,jdim]-self.means[varname][idim][jdim])/max(self.stds[varname][idim][jdim], 1e-4))
        comb_data =  np.stack(comb_data, -1)#, dtype="float32")
        return comb_data.astype(np.float32)

    def _reconstruct_sample(self, file, sample_idx, leadtime, time_idx, nsteps_input):
        """
        #sample_idx: which trajectory in file, where the data is stored in the shape (n_trajectories, n_timesteps, x, y, z)
        #time_idx: the begining of input time index
        #return solutions in shape: [T, C, D, H, W] for 3D and [T, C, H, W] for 2D
        #leadtime: time index of a future solution 
        """
        time_idx += self.time_skip
        #get input history
        comb_x = self._readdata(file, sample_idx, time_idx, time_idx+nsteps_input)
        if self.type not in ["postneutronstarmerger"]: #temporary, reduced D from 66 to 64
            assert comb_x.shape==tuple([nsteps_input]+[dim for dim in self.cubsizes]+[len(self.field_names)]), f"{comb_x.shape}, {tuple([nsteps_input]+[dim for dim in self.cubsizes]+[len(self.field_names)])}, {file}"
        #get the label at time_idx-1+leadtime
        time_idx = time_idx+nsteps_input+leadtime.item()-1
        comb_y = self._readdata(file, sample_idx, time_idx, time_idx+1)
        if self.type not in ["postneutronstarmerger"]: #temporary, reduced D from 66 to 64
            assert comb_y.shape==tuple([1]+[dim for dim in self.cubsizes]+[len(self.field_names)])
        comb = np.concatenate((comb_x, comb_y), axis=0) #T, H, W, C or T, D, H, W, C

        #start index and end size of local split for current
        isz0, isx0, isy0    = self.blockdict["Ind_start"] # [idz, idx, idy]
        cbszz, cbszx, cbszy = self.blockdict["Ind_dim"] # [Dloc, Hloc, Wloc]

        if self.spatial_dims==2:#return: T,C,H,W
            return comb.transpose(0, 3, 1, 2)[:,:,isx0:isx0+cbszx, isy0:isy0+cbszy], leadtime.to(torch.float32)
        elif self.spatial_dims==3:#return: T,C,D,H,W
            return comb.transpose(0, 4, 1, 2, 3)[:,:,isz0:isz0+cbszz,isx0:isx0+cbszx, isy0:isy0+cbszy], leadtime.to(torch.float32)
        else:
            raise ValueError(f"unknown spatial dims {self.spatial_dims}")

class acoustic_scattering_maze(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'pressure', 'speed_of_sound']
        vector_names = ['velocity']
        tensor_names = []
        type = 'acousticmaze'
        cubsizes=[256, 256]
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]] 
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class acoustic_scattering_inclusions(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'pressure', 'speed_of_sound']
        vector_names = ['velocity']
        tensor_names = []
        type = 'acousticinclu'
        cubsizes=[256, 256] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class acoustic_scattering_discontinuous(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'pressure', 'speed_of_sound']
        vector_names = ['velocity']
        tensor_names = []
        type = 'acousticdiscont'
        cubsizes=[256, 256] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class active_matter(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['concentration']
        vector_names = ['velocity']
        tensor_names = ['D','E']
        type = 'activematter'
        cubsizes=[256, 256] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class convective_envelope_rsg(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'energy', 'pressure']
        vector_names = ['velocity']
        tensor_names = []
        type = 'convrsg'
        cubsizes=[256, 128, 256] 
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic
    
class euler_multi_quadrants_openBC(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'energy', 'pressure']
        vector_names = ['momentum']
        tensor_names = []        
        type = 'euleropen'
        cubsizes=[512, 512] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class euler_multi_quadrants_periodicBC(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'energy', 'pressure']
        vector_names = ['momentum']
        tensor_names = []  
        type = 'eulerperiodic'
        cubsizes=[512, 512] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class helmholtz_staircase(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['mask', 'pressure_im', 'pressure_re']
        vector_names = []
        tensor_names = []  
        type = 'helmholtzstaircase'
        cubsizes=[1024, 256] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic

class MHD_64(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density']
        vector_names = ['magnetic_field', 'velocity']
        tensor_names = []
        type = 'MHD64'
        cubsizes=[64, 64, 64] 
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic

class MHD_256(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density']
        vector_names = ['magnetic_field', 'velocity']
        tensor_names = []
        type = 'MHD256'
        cubsizes=[256, 256, 256] 
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic

class gray_scott_reaction_diffusion(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['A','B']
        vector_names = []
        tensor_names = []
        type = 'grayscottreactdiff'
        cubsizes=[128, 128] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic

class planetswe(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['height']
        vector_names = ['velocity']
        tensor_names = []
        type = 'planetswe'
        cubsizes=[256, 512] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class post_neutron_star_merger(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'electron_fraction', 'entropy', 'internal_energy', 'pressure', 'temperature']
        vector_names = ['magnetic_field', 'velocity']
        tensor_names = []
        type = 'postneutronstarmerger'
        #cubsizes=[192, 128, 66] 
        #FIXME: hardcoded now
        cubsizes=[192, 128, 64]
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic
    
class rayleigh_benard(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['buoyancy', 'pressure']
        vector_names = ['velocity']
        tensor_names = []
        type = 'rayleighbenard'
        cubsizes=[512, 128] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class rayleigh_taylor_instability(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density']
        vector_names = ['velocity']
        tensor_names = []
        type = 'rayleightaylor'
        cubsizes=[128, 128, 128] 
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic
    
class shear_flow(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['pressure', 'tracer']
        vector_names = ['velocity']
        tensor_names = []
        type = 'shearflow'
        cubsizes=[256, 512] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class supernova_explosion_64(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'pressure', 'temperature']
        vector_names = ['velocity']
        tensor_names = []
        type = 'supernova64'
        cubsizes=[64, 64, 64] 
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic
    
class supernova_explosion_128(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'pressure', 'temperature']
        vector_names = ['velocity']
        tensor_names = []
        type = 'supernova128'
        cubsizes=[128, 128, 128] 
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic
    
class turbulence_gravity_cooling(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'pressure', 'temperature']
        vector_names = ['velocity']
        tensor_names = []
        type = 'turbgravcool'
        cubsizes=[64, 64, 64] 
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic

class turbulent_radiative_layer_2D(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'pressure']
        vector_names = ['velocity']
        tensor_names = []
        type = 'turbradlayer2D'
        cubsizes=[128, 384] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
class turbulent_radiative_layer_3D(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['density', 'pressure']
        vector_names = ['velocity']
        tensor_names = []
        type = 'turbradlayer3D'
        cubsizes=[128, 128, 256] 
        spatial_dims = 3
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1,2]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1,2]for jdim in [0,1,2]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic

class viscoelastic_instability(TheWellDataset):
    @staticmethod
    def _specifics():
        scalar_names = ['c_zz', 'pressure']
        vector_names = ['velocity']
        tensor_names = ['C']
        type = 'viscoelastic'
        cubsizes=[512, 512] 
        spatial_dims = 2
        split_level="sample"  #pre-split in the well
        return scalar_names, vector_names, tensor_names, type, cubsizes, spatial_dims, split_level
    field_names = _specifics()[0]
    field_names += [varname+str(idim) for varname in  _specifics()[1] for idim in [0,1]]
    field_names += [varname+str(idim)+str(jdim) for varname in  _specifics()[2] for idim in [0,1]for jdim in [0,1]]
    def _get_specific_bcs(self, file):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
