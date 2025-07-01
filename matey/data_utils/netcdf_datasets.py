import torch
import torch.nn
import numpy as np
import os
from torch.utils.data import Dataset
import netCDF4
import glob
from .shared_utils import get_top_variance_patchids, plot_checking, plot_refinedtokens


class BasenetCDFDirectoryDataset(Dataset):
    """
    Base class for data loaders. Returns data in T x C x H x W format.
    Takes in path to directory of netCDF files to construct dset.
    Args:
        path (str): Path to directory of netCDF files
        include_string (str): Only include files with this string in name
        n_steps (int): Number of steps to include in each sample
        dt (int): Time step between samples
        lead_time: lead time for prediction output
        split (str): train/val/test split
        train_val_test (tuple): Percent of data to use for train/val/test
        subname (str): Name to use for dataset
        refine_ratio: pick int(refine_ratio*ntoken_coarse) tokens to refine
        gammaref: pick all tokens that with variances larger than gammaref*max_variance to refine
    """
    def __init__(self, path, include_string='', n_steps=1, dt=1, leadtime_max=1, split='train',
                 train_val_test=None, subname=None, tokenizer_heads=None, refine_ratio=None, gammaref=None, tkhead_name=None, SR_ratio=None,
                group_id=0, group_rank=0, group_size=1):
        super().__init__()
        self.path = path
        self.split = split
        if subname is None:
            self.subname = path.split('/')[-1]
        else:
            self.subname = subname
        self.dt = dt
        self.leadtime_max = leadtime_max
        self.n_steps = n_steps #history length
        self.include_string = include_string
        self.train_val_test = train_val_test
        self.partition = {'train': 0, 'val': 1, 'test': 2}[split]
        self.time_index, self.sample_index, self.field_names, self.type, self.split_level = self._specifics()
        self._get_directory_stats(path)
        self.title = self.type
        self.tokenizer_heads = tokenizer_heads
        self.tkhead_name=tkhead_name
        
        self.refine_ratio = refine_ratio
        self.gammaref = gammaref

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
        
    def _get_specific_stats(self, f):
        raise NotImplementedError # Per dset

    def _get_specific_bcs(self, f):
        raise NotImplementedError # Per dset

    def _reconstruct_sample(self, file, leadtime, time_idx, n_steps):
        raise NotImplementedError # Per dset - should be (x=(-history:local_idx+dt) so that get_item can split into x, y

    def _get_directory_stats(self, path):
        self.files_paths = glob.glob(path + "/*.nc")
        subfolders = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        if len(subfolders)>0:
             for subfolder in subfolders:
                 files_subset = glob.glob(subfolder + "/*.nc")
                 self.files_paths += files_subset
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        #print(f"necdf, {self.files_paths}, {self.n_files}", flush=True)
        self.file_lens = []
        self.file_steps = [] #total sample size for each file
        self.file_nsteps = [] #history length for each file
        self.file_samples = []
        self.split_offsets = []
        self.offsets = [0]
        for file in self.files_paths:
            if len(self.include_string) > 0 and self.include_string not in file:
                continue
            else:
                nc = netCDF4.Dataset(file, 'r')
                samples, steps = self._get_specific_stats(nc)
                if steps-self.n_steps-(self.dt-1) < 1:
                    raise ValueError('WARNING: File {} has {} steps, but n_steps/history is set to be {}.'.format(file, steps, self.n_steps))
                file_nsteps = self.n_steps
                self.file_lens.append(steps)
                self.file_nsteps.append(file_nsteps)
                self.file_steps.append(steps-file_nsteps-(self.dt-1))
                self.file_samples.append(samples)
                self.offsets.append(self.offsets[-1]+(steps-file_nsteps-(self.dt-1))*samples)
        self.offsets[0] = -1
        self.datasets = [None for _ in self.files_paths]
        self.len = self.offsets[-1]
        #get split for partition
        if self.train_val_test is None:
            print('WARNING: No train/val/test split specified. Using all data for training.')
            self.split_offset = 0
            self.len = self.offsets[-1]
        else:
            print('Using train/val/test split: {}'.format(self.train_val_test))
            total_samples = sum(self.file_samples)
            ideal_split_offsets = [int(self.train_val_test[i]*total_samples) for i in range(3)]
            end_ind = 0
            for i in range(self.partition+1):
                run_sum = 0
                start_ind = end_ind
                for samples, steps in zip(self.file_samples, self.file_steps):
                    run_sum += samples
                    if run_sum <= ideal_split_offsets[i]:
                        end_ind += samples * (steps)
                        if run_sum == ideal_split_offsets[i]:
                            break
                    else:
                        end_ind += np.abs((run_sum - samples) - ideal_split_offsets[i]) * (steps)
                        break
            self.split_offset = start_ind
            self.len = end_ind - start_ind


    def _loaddata_file(self, file_ind):
        self.datasets[file_ind] = netCDF4.Dataset(self.files_paths[file_ind], 'r')

    def __getitem__(self, index):
        if hasattr(index, '__len__') and len(index)==2:
            leadtime=torch.tensor([index[1]], dtype=torch.int)
            index = index[0]
        else:
            leadtime=None

        #go to the split (i.e., train, val, or test) collection
        index = index + self.split_offset

        file_idx = int(np.searchsorted(self.offsets, index, side='right')-1)
        nsteps = self.file_nsteps[file_idx]

        local_idx = index - max(self.offsets[file_idx], 0) # First offset is -1
        assert local_idx // self.file_steps[file_idx]==0

        time_idx = local_idx % self.file_steps[file_idx]


        #open image file
        if self.datasets[file_idx] is None:
            self._loaddata_file(file_idx)

        time_idx += nsteps

        if leadtime is None:
            #generate a random leadtime uniformly sampled from [1, self.leadtime_max]
            leadtime = torch.randint(1, min(self.leadtime_max+1, self.file_lens[file_idx]-time_idx+1), (1,))
        else:
            leadtime = min(leadtime, self.file_lens[file_idx]-time_idx)

        try:
            trajectory, leadtime = self._reconstruct_sample(self.datasets[file_idx], leadtime, time_idx, nsteps)
            bcs = self._get_specific_bcs(self.datasets[file_idx])
        except:
            raise RuntimeError(f'Failed to reconstruct sample for file {self.files_paths[file_idx]} time {time_idx}')

        #print("Pei debugging", trajectory.shape, index, flush=True)

        #T,C,H,W ==> T,C,D(=1),H,W for compatibility with 3D
        trajectory=np.expand_dims(trajectory, axis=2)
        for tk in self.tokenizer_heads:
            if tk["head_name"] == self.tkhead_name:
                patch_size = tk["patch_size"]
                break

        for tk in self.tokenizer_heads:
            if tk["head_name"] == self.tkhead_name:
                patch_size = tk["patch_size"]
                break
        if len(patch_size)==2 and (self.refine_ratio is not None or self.gammaref is not None):
            refineind = get_top_variance_patchids(patch_size, trajectory[:-1], self.gammaref, self.refine_ratio)

            return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1], refineind, leadtime

        return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1], leadtime

    def __len__(self):
        return self.len

class  CollisionDataset(BasenetCDFDirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ['dens', 'potentialtemperature', 'uwnd', 'wwnd']
        type = 'thermalcollision2d'
        split_level = None #not used, since one trajectory per file
        return time_index, sample_index, field_names, type, split_level
    field_names = _specifics()[2] #class attributes

    def get_min_max(self):
        #FIXME: the values have changed with varying configs
        self.densminmax = [0.40968536690797414, 1.2241610005988284]
        self.ptminmax = [278.1360863367233, 323.3630877579251]
        self.uminmax = [-48.099862293555944, 48.09986229355615]
        self.wminmax = [-58.27943937042431, 58.3000711270841]


    def _get_norm_data(self, data):
        self.get_min_max()
        data[:,:,:,0] = (data[:,:,:,0] - self.densminmax[0])/(self.densminmax[1] - self.densminmax[0])
        data[:,:,:,1] = (data[:,:,:,1] - self.ptminmax[0])  /(self.ptminmax[1]   - self.ptminmax[0])
        data[:,:,:,2] = (data[:,:,:,2] - self.uminmax[0])   /(self.uminmax[1]    - self.uminmax[0])
        data[:,:,:,3] = (data[:,:,:,3] - self.wminmax[0])   /(self.wminmax[1]    - self.wminmax[0])

        return data

    def _get_specific_stats(self, dat):
        #samples = list(f.keys())
        steps = dat.dimensions["t"].size
        return 1, steps

    def _get_specific_bcs(self, dat):
        #return [0, 0] # Non-periodic
        return [1, 0]

    def _get_temperature_pressure(self, rho_full, theta_full):
        C0 = 27.5629410929725921310572974482
        gamma = 1.40027894002789400278940027894
        R_d = 287.
        pressure = (C0*(rho_full*theta_full)**gamma)
        temp     = (pressure/rho_full/R_d)
        return temp, pressure

    def _reconstruct_sample(self, dat, leadtime, time_idx, n_steps):
        #get history of length n_steps
        rho_pert   = np.ma.getdata(dat.variables["dens" ][time_idx-n_steps:time_idx,:,:]) #nt, nz, nx
        uvel       = np.ma.getdata(dat.variables["uwnd" ][time_idx-n_steps:time_idx,:,:])
        wvel       = np.ma.getdata(dat.variables["wwnd" ][time_idx-n_steps:time_idx,:,:])
        theta_pert = np.ma.getdata(dat.variables["theta"][time_idx-n_steps:time_idx,:,:])
        hy_rho     = np.ma.getdata(dat.variables["hy_dens" ][:])
        hy_theta   = np.ma.getdata(dat.variables["hy_theta"][:])

        hy_rho = np.expand_dims(hy_rho, (0, 2))
        hy_theta = np.expand_dims(hy_theta, (0, 2))

        rho_full   = rho_pert   + hy_rho
        theta_full = theta_pert + hy_theta
        comb_x =  np.stack([rho_full, theta_full,  uvel, wvel], -1) #, dtype="float32")

        #get label at time_idx-1+leadtime
        rho_pert_y   = np.ma.getdata(dat.variables["dens" ][time_idx-1+leadtime,:,:]) #nt, nz, nx
        uvel_y       = np.ma.getdata(dat.variables["uwnd" ][time_idx-1+leadtime,:,:])
        wvel_y       = np.ma.getdata(dat.variables["wwnd" ][time_idx-1+leadtime,:,:])
        theta_pert_y = np.ma.getdata(dat.variables["theta"][time_idx-1+leadtime,:,:])
        rho_full_y   = rho_pert_y   + hy_rho
        theta_full_y = theta_pert_y + hy_theta
        comb_y =  np.stack([rho_full_y, theta_full_y,  uvel_y, wvel_y], -1) #, dtype="float32")

        comb = np.concatenate((comb_x, comb_y), axis=0)
        comb_norm =self._get_norm_data(comb)

        return comb_norm.transpose(0, 3, 1, 2).astype(np.float32), leadtime.to(torch.float32)

