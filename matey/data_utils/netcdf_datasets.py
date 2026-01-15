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
        supportdata (bool): Whether to include additional support data as input (e.g., control actuator)
        split (str): train/val/test split
        train_val_test (tuple): Percent of data to use for train/val/test
        subname (str): Name to use for dataset
        refine_ratio: pick int(refine_ratio*ntoken_coarse) tokens to refine
        gammaref: pick all tokens that with variances larger than gammaref*max_variance to refine
    """
    def __init__(self, path, include_string='', n_steps=1, dt=1, leadtime_max=1, supportdata=None, split='train',
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
        self.input_control_act = (isinstance(supportdata, list)
            and any("input_control_act" in d and d["input_control_act"] for d in supportdata))
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

    def _reconstruct_sample(self, file, leadtime, input_control, time_idx, n_steps):
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
        # FIXME:this should depend on autoregressive nature, not on the input_control_act flag
        # For autoregressive cases the rollout length is decided at runtime
        if leadtime is None and self.input_control_act in [None, False]:
            #generate a random leadtime uniformly sampled from [1, self.leadtime_max]
            leadtime = torch.randint(1, min(self.leadtime_max+1, self.file_lens[file_idx]-time_idx+1), (1,))
        elif self.input_control_act==True:
            leadtime = torch.tensor(min(self.leadtime_max, self.file_lens[file_idx]-time_idx+1))
        else:
            leadtime = min(leadtime, self.file_lens[file_idx]-time_idx)

        try:
            trajectory, leadtime, input_control = self._reconstruct_sample(self.datasets[file_idx], leadtime, time_idx, nsteps)
            bcs = self._get_specific_bcs(self.datasets[file_idx])
        except:
            raise RuntimeError(f'Failed to reconstruct sample for file {self.files_paths[file_idx]} time {time_idx}')

        #print("Pei debugging", trajectory.shape, index, flush=True)

        #T,C,H,W ==> T,C,D(=1),H,W for compatibility with 3D
        trajectory=np.expand_dims(trajectory, axis=2)

        if self.input_control_act==True:
            traj_out = trajectory[self.n_steps:]
            # pad trajectory so it's always the same size
            # sometimes leadtime can be smaller due to end of data
            if traj_out.shape[0] < self.leadtime_max:
                pad_len = self.leadtime_max - traj_out.shape[0]
                pad_shape = [pad_len] + list(traj_out.shape[1:])
                pad_tensor = np.zeros(pad_shape)
                traj_out = np.concatenate([traj_out, pad_tensor], axis=0)
                pad_tensor_control = np.zeros(pad_len).astype(np.float32)
                input_control=np.concatenate([input_control, pad_tensor_control],axis=0)

            return trajectory[0:self.n_steps], torch.as_tensor(bcs), traj_out.astype(np.float32), leadtime, input_control.astype(np.float32)
        else:
            return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1].astype(np.float32), leadtime

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

        return comb_norm.transpose(0, 3, 1, 2).astype(np.float32), leadtime.to(torch.float32), None

class  SOLPSDataset(BasenetCDFDirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ['density', 'temperature','radiated power'] #state field names, omit input field names for now
        type = 'SOLPS2D'
        split_level = None #not used, since one trajectory per file
        return time_index, sample_index, field_names, type, split_level
    field_names = _specifics()[2] #class attributes
    #FIXME: On NERSC and PSC we get an error, so use the __func__() way
    #field_names = _specifics.__func__()[2] #class attributes


    def get_min_max(self):
        #print(casename, rho.min(), rho.max(), temp.min(), temp.max(), in_actu.min(), in_actu.max(), timestep.min(), timestep.max())
        #solps-kstar_example-1 756380529461540.2 2.1904665841323272e+20 1e-05 220.66874753097187 5e+20 2.747249999999793e+21 0.0001 0.0908
        self.densminmax = [756380529461540.2, 2.1904665841323272e+20]
        self.tempminmax = [1e-05, 220.66874753097187]
        self.in_actuminmax=[5e+20, 2.747249999999793e+21]
        self.rad_powminmax=[3.9666444377903453e-19, 2163.635756123288]

    def _get_norm_data(self, data):
        self.get_min_max()
        data[:,:,:,0] = (data[:,:,:,0] - self.densminmax[0])/(self.densminmax[1] - self.densminmax[0])
        data[:,:,:,1] = (data[:,:,:,1] - self.tempminmax[0])  /(self.tempminmax[1]   - self.tempminmax[0])
        data[:,:,:,2] = (data[:,:,:,2] - self.rad_powminmax[0])   /(self.rad_powminmax[1]    - self.rad_powminmax[0])
        return data

    def _get_specific_stats(self, dat):
        #samples = list(f.keys())
        steps = dat.dimensions["nt"].size
        return 1, steps

    def _get_specific_bcs(self, dat):
        #not used
        return [0, 0]

    def _reconstruct_sample(self, dat, leadtime, time_idx, n_steps):
        """
        netcdf solps-kstar_example-1 {
        dimensions:
            nx = 98 ;
            ny = 38 ;
            nt = 908 ;
            dt = 1 ;
        variables:
            double density(nt, ny, nx) ;
            double temperature(nt, ny, nx) ;
            double input\ actuator(nt) ;
            double radiated\ power(nt, ny, nx) ;
            double timestep(nt) ;
        }
        """
        dt = dat.dimensions["dt"].size
        #get history of length n_steps
        rho     = dat.variables["density" ][time_idx-n_steps:time_idx,:,:] #nt, ny, nx
        temp    = dat.variables["temperature" ][time_idx-n_steps:time_idx,:,:] 
        if self.input_control_act == True:
            in_actu = dat.variables["input actuator" ][time_idx-n_steps:time_idx+leadtime] 

            # normalize input actuator
            self.get_min_max()
            in_actu = (in_actu - self.in_actuminmax[0])   /(self.in_actuminmax[1]    - self.in_actuminmax[0])

        rad_pow = dat.variables["radiated power" ][time_idx-n_steps:time_idx] 
 
        comb_x =  np.stack([rho, temp, rad_pow], -1).astype(np.float32)

        #get label at time_idx-1+leadtime
        rho_y      = dat.variables["density"       ][time_idx:time_idx+leadtime,:,:] #1, ny, nx
        temp_y     = dat.variables["temperature"   ][time_idx:time_idx+leadtime,:,:] 
        rad_pow_y  = dat.variables["radiated power"][time_idx:time_idx+leadtime,:,:]

        comb_y =  np.stack([rho_y, temp_y, rad_pow_y], -1).astype(np.float32)

        comb = np.concatenate((comb_x, comb_y), axis=0)
        comb_norm =self._get_norm_data(comb)
        if self.input_control_act == True:
            return comb_norm.transpose(0, 3, 1, 2), leadtime.to(torch.float32), in_actu.astype(np.float32)
        else:
            return comb_norm.transpose(0, 3, 1, 2), leadtime.to(torch.float32), None
