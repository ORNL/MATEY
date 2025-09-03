import torch
import torch.nn
import numpy as np
import os
from torch.utils.data import Dataset
import h5py
import glob
from .shared_utils import get_top_variance_patchids, plot_checking
import json
import csv
from .utils import closest_factors
from functools import reduce
from operator import mul

class BaseBLASNET3DDataset(Dataset):
    """
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
        refine_ratio: pick int(refine_ratio*ntoken_coarse) tokens to refine
        gammaref: pick all tokens that with variances larger than gammaref*max_variance to refine
        patch_size: list of patch sizes for converting from solution fields to patches/tokens
        leadtime_max: when >0, future solution solution prediction, tar is a solution at the lead time;
                      when =0, self-supervised learning and tar is None
        SR_ratio: superresolution ratio, used when input and output are at different resolutions, 
        currently only support this case: https://www.kaggle.com/datasets/waitongchung/blastnet-momentum-3d-sr-dataset/data
    """
    def __init__(self, path, include_string='', n_steps=1, dt=1, leadtime_max=0, split='train', 
                 train_val_test=None, extra_specific=False, tokenizer_heads=None, refine_ratio=None, 
                 gammaref=None, tkhead_name=None, SR_ratio=None, data_augmentation=False,
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
        if SR_ratio is None:
            SR_ratio=[1,1,1]
        self.SR_ratio=SR_ratio
        self.data_augmentation = data_augmentation
        self.time_index, self.sample_index, self.field_names, self.type, self.cubsizes, self.case_str, self.split_level = self._specifics()
        self._get_directory_stats(path)
        self.title = self.type

        self.tokenizer_heads = tokenizer_heads
        self.tkhead_name=tkhead_name
        self.refine_ratio = refine_ratio
        self.gammaref = gammaref
        self.group_id=group_id
        self.group_rank=group_rank
        self.group_size=group_size
        self.blockdict = self._getblocksplitstat()
        #make sure all GPUs in the same data group conduct the same augmentation operation
        self.rng = np.random.default_rng(seed=self.group_id)

    def get_name(self):
        return self.type

    @staticmethod
    def _specifics():
        # Sets self.field_names, self.dataset_type
        raise NotImplementedError # 

    def get_per_file_dsets(self):
        return [self]
    
    def _get_specific_bcs(self, f):
        raise NotImplementedError #

    def _reconstruct_sample(self, dictcase, time_idx, leadtime):
        raise NotImplementedError # Per dset - should be (x=(-history:local_idx+dt) so that get_item can split into x, y
    
    def _get_filesinfo(self, file_paths):
        raise NotImplementedError 
    
    def _getpathfromcsv(self, path):
        pass

    def _split_cases(self):
        if self.train_val_test is not None:
            cases_per_part = np.ceil(np.array(self.train_val_test)*self.total_cases).astype(int)
            cases_per_part[2] = max(self.total_cases- cases_per_part[0] - cases_per_part[1], 0)
            partition = self.partition
            ist=sum(cases_per_part[:partition])
            iend=sum(cases_per_part[:partition+1])
            casesids = np.arange(self.total_cases)
            np.random.shuffle(casesids) 
            return casesids[ist:iend]
        else:
            return np.arange(self.total_cases)
        
    def _get_directory_stats(self, path):
        self.files_paths = glob.glob(path + "/"+ self.case_str) 
        self.files_paths.sort()
        self.case_dict=self._get_filesinfo(self.files_paths)
        assert self.split_level=="case" or self.split_level=="snapshot"
        if self.split_level=="case":
            #split into train/val/test at case level
            self.total_cases = len(self.case_dict)
            self.casesids_split = self._split_cases()
            self.cases_split=[]
            self.file_nsteps=[]
            self.offset=[0]

            for icase in self.casesids_split:
                casedir=self.files_paths[icase]
                self.cases_split.append(casedir)
                dictcase = self.case_dict[casedir]
                # for a given solution with `dictcase["ntimes"]` steps and input length of self.nsteps_input, 
                # the number of time segments/samples for (input, next-step prediction) is 
                # when leadtime_max==0: self-supervised and hence number of samples equal to ntimes
                nsample_case=dictcase["ntimes"]-self.nsteps_input if self.leadtime_max>0 else dictcase["ntimes"]
                if nsample_case < self.leadtime_max:
                    raise RuntimeError('Error: Path {} has {} steps, but nsteps_input is {}. Please set file steps = max allowable.'.format(path, dictcase["ntimes"], self.nsteps_input))
                self.offset.append(nsample_case)
                self.file_nsteps.append(dictcase["ntimes"])
            self.offset=np.cumsum(self.offset)
            self.len=self.offset[-1] 
        elif self.split_level=="snapshot":
            assert self.leadtime_max==0 and self.nsteps_input==1
            #split into train/val/test at snapshot level
            self.total_cases = len(self.case_dict["solutions"])
            self.casesids_split = self._split_cases()
            self.len=len(self.casesids_split)
        else:
            raise ValueError(self.split_level)
    
    def _open_file(self, filepath):
        _file = h5py.File(filepath, 'r')
        self.data_files[filepath] = _file

    def __getitem__(self, index):
        if hasattr(index, '__len__') and len(index)==2:
            leadtime=torch.tensor([index[1]], dtype=torch.int)
            index = index[0]
        else:
            leadtime=None

        if self.split_level=="case":
            case_idx = np.searchsorted(self.offset, index, side='right', sorter=None)-1
            time_idx = index - self.offset[case_idx]

            nsteps = self.file_nsteps[case_idx] # Number of time steps per given file

            if leadtime is None:
                #generate a random leadtime uniformly sampled from [1, self.leadtime_max]
                leadtime = torch.tensor([0])
                if self.leadtime_max>0:
                    leadtime = torch.randint(1, min(self.leadtime_max+1, nsteps-time_idx-self.nsteps_input+1), (1,))
            else:
                leadtime = min(leadtime, nsteps-time_idx-self.nsteps_input)

            assert time_idx + self.nsteps_input + leadtime <= nsteps

            #solution info
            casedir = self.cases_split[case_idx]
            dictcase = self.case_dict[casedir]
            trajectory, leadtime = self._reconstruct_sample(dictcase, time_idx.item(), leadtime)
        elif self.split_level=="snapshot":
            case_idx = self.casesids_split[index]
            leadtime = torch.tensor([0])
            if self.type!="SR":
                dictcase = self.case_dict["solutions"][case_idx]
                nxyz = self.case_dict["Nxyz"][case_idx]
                trajectory, leadtime= self._reconstruct_sample(dictcase, nxyz, leadtime)
        else:
            raise ValueError("unknown %s"%self.split_level)
        ########################################
        
        #start index and end size of local split for current 
        isz0, isx0, isy0    = self.blockdict["Ind_start"] # [idz, idx, idy]
        cbszz, cbszx, cbszy = self.blockdict["Ind_dim"] # [Dloc, Hloc, Wloc]


        bcs = self._get_specific_bcs()
        if self.type!="SR":
            trajectory = trajectory[:,:,isz0:isz0+cbszz,isx0:isx0+cbszx, isy0:isy0+cbszy]#T,C,Dloc,Hloc,Wloc
            if self.leadtime_max>0:
                inp=trajectory[:-1]
                tar=trajectory[-1]
            else: #self-supervised
                inp=trajectory
                tar=inp[-1]
        else:
            inp, tar, dzdxdy = self._reconstruct_sample(case_idx)
            #blockdict has dims for full resoluton output; need to convert to LR inputs

            inp = inp[:,:,isz0//self.SR_ratio[0]:(isz0+cbszz)//self.SR_ratio[0],isx0//self.SR_ratio[1]:(isx0+cbszx)//self.SR_ratio[1], isy0//self.SR_ratio[2]:(isy0+cbszy)//self.SR_ratio[2]]#T,C,Dloc,Hloc,Wloc
            tar = tar[:,:,isz0:isz0+cbszz,isx0:isx0+cbszx, isy0:isy0+cbszy]#T,C,Dloc,Hloc,Wloc



        for tk in self.tokenizer_heads:
            if tk["head_name"] == self.tkhead_name:
                patch_size = tk["patch_size"]
                break
        if len(patch_size)==2 and (self.refine_ratio is not None or self.gammaref is not None):
            refineind = get_top_variance_patchids(patch_size, inp, self.gammaref, self.refine_ratio)

            return inp, torch.as_tensor(bcs), tar, refineind, leadtime
        return inp, torch.as_tensor(bcs), tar, leadtime

    def __len__(self):
        return self.len
    
    def _getblocksplitstat(self):
        H, W, D = self.cubsizes #x,y,z
        sequence_parallel_size=self.group_size
        Lz, Lx, Ly = 1.0, 1.0, 1.0
        Lz_start, Lx_start, Ly_start = 0.0, 0.0, 0.0
        ##############################################################
        #based on sequence_parallel_size, split the data in D, H, W direciton
        if sequence_parallel_size>1:
            nproc_blocks = closest_factors(sequence_parallel_size, 3)
        else:
            nproc_blocks = [1,1,1]
        assert reduce(mul, nproc_blocks)==sequence_parallel_size
        ##############################################################
        #split a sample by space into nprocz blocks for z-dim, nprocx blocks for x-dim, and nprocy blocks for y-dim
        Dloc = D//nproc_blocks[0]
        Hloc = H//nproc_blocks[1]
        Wloc = W//nproc_blocks[2]
        #keep track of each block/split ID
        iz, ix, iy = torch.meshgrid(torch.arange(nproc_blocks[0]), 
                                    torch.arange(nproc_blocks[1]),  
                                    torch.arange(nproc_blocks[2]), indexing="ij")
        blockIDs = torch.stack([iz.flatten(), ix.flatten(), iy.flatten()], dim=-1) #[sequence_parallel_size, 3]

        blockdict={}
        blockdict["Lzxy"] = [Lz/nproc_blocks[0], Lx/nproc_blocks[1], Ly/nproc_blocks[2]]
        blockdict["nproc_blocks"] = nproc_blocks
        blockdict["Ind_dim"] = [Dloc, Hloc, Wloc]
        #######################
        idz, idx, idy = blockIDs[self.group_rank,:]
        blockdict["Ind_start"] = [idz*Dloc, idx*Hloc, idy*Wloc]
        Lz_loc, Lx_loc, Ly_loc = blockdict["Lzxy"]
        blockdict["zxy_start"]=[Lz_start+idz*Lz_loc, Lx_start+idx*Lx_loc, Ly_start+idy*Ly_loc]
        return blockdict

class H2vitairliDataset(BaseBLASNET3DDataset):
    @staticmethod
    def _specifics():
        """22 cases differing turbulence intensity, inflow uin, and integral lenght scale
        each with multiple independent(?) snapshots
        """
        case_str="free-propagating-h2-vit-air-li-case-*"
        time_index = 1
        sample_index = 0
        field_names = ["UX_ms-1", "UY_ms-1", "UZ_ms-1", "P_Pa", "T_K", "RHO_kgm-3", "YH2O", "YO2", "YH", "YOH", "YH2", "YHO2", "YO", "YH2O2", "YN2"]
        type = 'h2vitairli'
        cubsizes=[256, 128, 128]
        split_level="case" 
        return time_index, sample_index, field_names, type, cubsizes, case_str, split_level
    field_names = _specifics.__func__()[2] #class attributes

    def _get_filesinfo(self, file_paths):
        case_dict={}
        for datacasedir in file_paths:
            f = open(os.path.join(datacasedir, 'info.json'))
            metadata = json.load(f) 
            variables = metadata["global"]["variables"]
            ntimes = metadata["global"]["snapshots"]
            assert ntimes==len(metadata['local'])
            nvar = len(variables)
            case_dict[datacasedir]={}
            case_dict[datacasedir]['basedir']=datacasedir
            case_dict[datacasedir]['x']=os.path.join(datacasedir, metadata['global']['grid']['x'])
            case_dict[datacasedir]['y']=os.path.join(datacasedir, metadata['global']['grid']['y'])
            case_dict[datacasedir]['z']=os.path.join(datacasedir, metadata['global']['grid']['z'])
            case_dict[datacasedir]['ntimes']=ntimes
            case_dict[datacasedir]['nxnynz']=metadata["global"]['Nxyz']
            case_dict[datacasedir]['local']=metadata['local']
        return case_dict

    def _reconstruct_sample(self, dictcase, time_idx, leadtime):
        #dictcase: dictionary contains the global and local info of the picked case
        #time_idx: the begining of input time index
        #return solutions in shape: [T, C, D, H, W]
        #leadtime: time index of a future solution 
        solutions= dictcase['local']
        nx,ny,nz = dictcase['nxnynz']
        datacasedir=dictcase['basedir']
        nvar=len(self.field_names)
        #FIXME: the sol['id'] is not consecutive integers for the solutions; need to check what solutions are saved
        #FIXME: temporary nx_coarse to fit into hip
        sol_fields = np.empty((self.cubsizes[0],self.cubsizes[1],self.cubsizes[2],nvar,self.nsteps_input+1*self.leadtime_max), dtype=np.float32)
        #print([(isol,sol['id']) for isol, sol in enumerate(solutions)], flush=True)
        skipx=nx//self.cubsizes[0]
        skipy=ny//self.cubsizes[1]
        skipz=nz//self.cubsizes[2]
        nx_end=self.cubsizes[0]*skipx
        ny_end=self.cubsizes[1]*skipy
        nz_end=self.cubsizes[2]*skipz
        #get input history
        for isol, sol in enumerate(solutions[time_idx:time_idx+self.nsteps_input]):
            #assert isol+time_idx==sol['id']
            for ivar, var in enumerate(self.field_names):
                varfile=os.path.join(datacasedir, sol[var+" filename"])
                sol_fields[:,:,:,ivar,isol] = np.fromfile(varfile,dtype='<f4').reshape(nx,ny,nz)[:nx_end:skipx,:ny_end:skipy,:nz_end:skipz] 
        #get the label
        if self.leadtime_max>0:
            time_idx = time_idx+self.nsteps_input+leadtime.item()-1
            sol = solutions[time_idx]
            #assert time_idx==sol['id']
            for ivar, var in enumerate(self.field_names):
                varfile=os.path.join(datacasedir, sol[var+" filename"])
                sol_fields[:,:,:,ivar,-1] = np.fromfile(varfile,dtype='<f4').reshape(nx,ny,nz)[:nx_end:skipx,:ny_end:skipy,:nz_end:skipz] 
        #return: T,C,D,H,W
        return sol_fields.transpose((4, 3, 2, 0, 1)), leadtime.to(torch.float32)

    def _get_specific_bcs(self):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic

class H2jetDataset(BaseBLASNET3DDataset):
    @staticmethod
    def _specifics():
        case_str="hydrogen-jet-*00"
        time_index = 1
        sample_index = 0
        field_names = ["RHO_kgm-3","UX_ms-1","UY_ms-1","P_Pa","T_K","YH","YH2","YO","YO2","YOH","YH2O","YHO2","YH2O2"]
        type = 'h2jetRe'
        cubsizes=[1536, 1920, 1] #[1600, 2000, 1] 
        #this is a 2D dataset; axial and streamwise (https://blastnet.github.io/sharma2024.html) 
        split_level="case" 
        return time_index, sample_index, field_names, type, cubsizes, case_str, split_level
    field_names = _specifics.__func__()[2] #class attributes

    def _get_filesinfo(self, file_paths):
        case_dict={}
        for datacasedir in file_paths:
            f = open(os.path.join(datacasedir, 'info.json'))
            metadata = json.load(f) 
            variables = metadata["global"]["variables"]
            ntimes = metadata["global"]["snapshots"]
            assert ntimes==len(metadata['local'])
            nvar = len(variables)
            case_dict[datacasedir]={}
            case_dict[datacasedir]['basedir']=datacasedir
            case_dict[datacasedir]['x']=os.path.join(datacasedir, metadata['global']['grid']['x'])
            case_dict[datacasedir]['y']=os.path.join(datacasedir, metadata['global']['grid']['y'])
            case_dict[datacasedir]['nxnynz']=metadata["global"]['Nxyz']
            local_lists=[]
            for isol, sol in enumerate(metadata['local']):
                varfile=os.path.join(datacasedir, sol[self.field_names[0]+" filename"])
                if not os.path.exists(varfile):
                    continue
                local_lists.append(sol)
            case_dict[datacasedir]['local']=local_lists
            if len(local_lists)!=ntimes:
                print("warning: snapshots from info.json not consistent with actual files for ", datacasedir, flush=True)
                ntimes=len(local_lists)
            case_dict[datacasedir]['ntimes']=ntimes
        return case_dict

    def _reconstruct_sample(self, dictcase, time_idx, leadtime):
        #dictcase: dictionary contains the global and local info of the picked case
        #time_idx: the begining of input time index
        #return solutions in shape: [T, C, D, H, W]
        #leadtime: time index of a future solution 
        solutions= dictcase['local']
        nx,ny = dictcase['nxnynz']
        nz = 1
        datacasedir=dictcase['basedir']
        nvar=len(self.field_names)
        #FIXME: the sol['id'] is not consecutive integers for the solutions; need to check what solutions are saved
        #FIXME: temporary nx_coarse to fit into hip
        sol_fields = np.empty((self.cubsizes[0],self.cubsizes[1],self.cubsizes[2],nvar,self.nsteps_input+1*self.leadtime_max), dtype=np.float32)
        #print([(isol,sol['id']) for isol, sol in enumerate(solutions)], flush=True)
        skipx=nx//self.cubsizes[0]
        skipy=ny//self.cubsizes[1]
        skipz=nz//self.cubsizes[2]
        nx_end=self.cubsizes[0]*skipx
        ny_end=self.cubsizes[1]*skipy
        nz_end=self.cubsizes[2]*skipz
        #get input history
        for isol, sol in enumerate(solutions[time_idx:time_idx+self.nsteps_input]):
            #assert isol+time_idx==sol['id']
            for ivar, var in enumerate(self.field_names):
                varfile=os.path.join(datacasedir, sol[var+" filename"])
                sol_fields[:,:,:,ivar,isol] = np.fromfile(varfile,dtype='<f4').reshape(nx,ny,nz)[:nx_end:skipx,:ny_end:skipy,:nz_end:skipz] 
        #get the label
        if self.leadtime_max>0:
            time_idx = time_idx+self.nsteps_input+leadtime.item()-1
            sol = solutions[time_idx]
            #assert time_idx==sol['id']
            for ivar, var in enumerate(self.field_names):
                varfile=os.path.join(datacasedir, sol[var+" filename"])
                sol_fields[:,:,:,ivar,-1] = np.fromfile(varfile,dtype='<f4').reshape(nx,ny,nz)[:nx_end:skipx,:ny_end:skipy,:nz_end:skipz] 
        #return: T,C,D,H,W
        return sol_fields.transpose((4, 3, 2, 0, 1)), leadtime.to(torch.float32)

    def _get_specific_bcs(self):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic

class FHITsnapshots(BaseBLASNET3DDataset):
    @staticmethod
    def _specifics():
        # "description": "Passive scalar transport in forced homogeneous isotropic turbulence DNS"
        case_str="passive-fhit-dns-r*[0-9]"
        time_index = 1
        sample_index = 0
        field_names = ["UX_ms-1","UY_ms-1","UZ_ms-1","Y"]
        type = 'pass-fhit'
        cubsizes=[256, 256, 256] 
        split_level="snapshot"
        return time_index, sample_index, field_names, type, cubsizes, case_str, split_level
    field_names = _specifics.__func__()[2] #class attributes

    def _get_filesinfo(self, file_paths):
        skip="passive-fhit-dns-r1/./data/UX_ms-1_id000.dat" #damaged file
        case_dict={}
        sols_lists=[]
        nxyz_lists=[]
        for datacasedir in file_paths:
            f = open(os.path.join(datacasedir, 'info.json'))
            metadata = json.load(f) 
            variables = metadata["global"]["variables"]
            ntimes = metadata["global"]["snapshots"]
            assert ntimes==len(metadata['local'])
            nvar = len(variables)
            case_dict[datacasedir]={}
            case_dict[datacasedir]['basedir']=datacasedir
            try:
                case_dict[datacasedir]['x']=os.path.join(datacasedir, metadata['global']['grid']['x'])
                case_dict[datacasedir]['y']=os.path.join(datacasedir, metadata['global']['grid']['y'])
                case_dict[datacasedir]['z']=os.path.join(datacasedir, metadata['global']['grid']['z'])
            except:
                #two cases have no grid info: passive-fhit-dns-r3 and passive-fhit-dns-r4
                case_dict[datacasedir]['x']=None
                case_dict[datacasedir]['y']=None
                case_dict[datacasedir]['z']=None
            case_dict[datacasedir]['ntimes']=ntimes
            case_dict[datacasedir]['nxnynz']=metadata["global"]['Nxyz']
            for sol in metadata['local']:
                varfile=os.path.join(datacasedir, sol[self.field_names[0]+" filename"])
                if not os.path.exists(varfile) or skip in varfile:
                    continue
                for var in self.field_names:
                    sol[var+" filename"]=os.path.join(datacasedir, sol[var+" filename"])
                sols_lists.append(sol)
                nxyz_lists.append(metadata["global"]['Nxyz'])
        case_dict['solutions']=sols_lists
        case_dict["Nxyz"]=nxyz_lists
        return case_dict

    def _reconstruct_sample(self, dictcase, nxyz, leadtime=0.0):
        #dictcase: solution dictionary
        #return solutions in shape: [T=1, C, D, H, W]
        nx,ny,nz = nxyz
        nvar=len(self.field_names)
        sol=dictcase
        sol_fields = np.empty((self.cubsizes[0],self.cubsizes[1],self.cubsizes[2],nvar,self.nsteps_input), dtype=np.float32)
        skipx=nx//self.cubsizes[0]
        skipy=ny//self.cubsizes[1]
        skipz=nz//self.cubsizes[2]
        nx_end=self.cubsizes[0]*skipx
        ny_end=self.cubsizes[1]*skipy
        nz_end=self.cubsizes[2]*skipz
        #assert isol+time_idx==sol['id']
        for ivar, var in enumerate(self.field_names):
            varfile=sol[var+" filename"]
            try:
                sol_fields[:,:,:,ivar,0] = np.fromfile(varfile,dtype='<f4').reshape(nx,ny,nz)[:nx_end:skipx,:ny_end:skipy,:nz_end:skipz] 
            except:
                raise ValueError("cannot read from %s"%varfile)
        #return: T,C,D,H,W
        return sol_fields.transpose((4, 3, 2, 0, 1)), leadtime

    def _get_specific_bcs(self):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic


class HITDNSsnapshots(BaseBLASNET3DDataset):
    @staticmethod
    def _specifics():
        # "Decaying Homogeneous Isotropic Turbulence DNS"
        case_str="canonical-hit-dns-*pressure"
        time_index = 1
        sample_index = 0
        field_names = ["P_Pa",  "UX_ms-1","UY_ms-1","UZ_ms-1"]
        type = 'hit-dns'
        cubsizes=[256, 256, 256] #[2040,2040,2048] 
        split_level="case"
        return time_index, sample_index, field_names, type, cubsizes, case_str,split_level
    field_names = _specifics.__func__()[2] #class attributes

    def _get_filesinfo(self, file_paths):
        case_dict={}
        for datacasedir in file_paths:
            f_p = open(os.path.join(datacasedir, 'info.json'))
            metadata_p = json.load(f_p) 
            f_vec = open(os.path.join(datacasedir.replace('pressure','velocity'), 'info.json'))
            metadata_vec = json.load(f_vec) 
            variables = metadata_p["global"]["variables"]+metadata_vec["global"]["variables"]
            assert variables==self.field_names
            ntimes = metadata_p["global"]["snapshots"]
            assert ntimes==len(metadata_p['local'])
            assert ntimes==metadata_vec["global"]["snapshots"]
            case_dict[datacasedir]={}
            case_dict[datacasedir]['basedir_p']=datacasedir
            case_dict[datacasedir]['ntimes']=ntimes
            case_dict[datacasedir]['nxnynz']=metadata_p["global"]['Nxyz']
            case_dict[datacasedir]['local_p']=metadata_p['local']
            case_dict[datacasedir]['local_vec']=metadata_vec['local']
        return case_dict

    def _reconstruct_sample(self, dictcase, time_idx,  leadtime=0.0):
        #dictcase: dictionary contains the global and local info of the picked case
        #time_idx: the begining of input time index
        #return solutions in shape: [T, C, D, H, W]
        #FIXME: leadtime is not used, as the snapshots are independent (double check later); and hence not for spatiotemporal predictions
        assert leadtime==0.0
        solutions_p= dictcase['local_p']
        solutions_v= dictcase['local_vec']
        nx,ny,nz = dictcase['nxnynz']
        datacasedir=dictcase['basedir_p']
        nvar=len(self.field_names)
        sol_fields = np.empty((self.cubsizes[0],self.cubsizes[1],self.cubsizes[2],nvar,self.nsteps_input), dtype=np.float32)
        skipx=nx//self.cubsizes[0]
        skipy=ny//self.cubsizes[1]
        skipz=nz//self.cubsizes[2]
        nx_end=self.cubsizes[0]*skipx
        ny_end=self.cubsizes[1]*skipy
        nz_end=self.cubsizes[2]*skipz
        #get input history
        for isol, (sol_p, sol_v) in enumerate(zip(solutions_p[time_idx:time_idx+self.nsteps_input], solutions_v[time_idx:time_idx+self.nsteps_input])):
            for ivar, var in enumerate(self.field_names):
                if ivar==0:
                    varfile=os.path.join(datacasedir, sol_p[var+" filename"])
                else:
                    varfile=os.path.join(datacasedir.replace('pressure','velocity'), sol_v[var+" filename"])
                sol_fields[:,:,:,ivar,isol] = np.fromfile(varfile,dtype='<f4').reshape(nx,ny,nz)[:nx_end:skipx,:ny_end:skipy,:nz_end:skipz] 
        #return: T,C,D,H,W
        return sol_fields.transpose((4, 3, 2, 0, 1)), leadtime

    def _get_specific_bcs(self):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic

######SR task example from https://www.kaggle.com/code/waitongchung/momentum128-readandinfer/notebook########
class SR_Benchmark(BaseBLASNET3DDataset):
    @staticmethod
    def _specifics():
        # "description": "Passive scalar transport in forced homogeneous isotropic turbulence DNS"
        case_str=None
        time_index = 1
        sample_index = 0
        field_names = ['RHO_kgm-3_id','UX_ms-1_id','UY_ms-1_id','UZ_ms-1_id']
        type = 'SR'
        cubsizes=[128, 128, 128] 
        split_level="snapshot"
        return time_index, sample_index, field_names, type, cubsizes, case_str, split_level
    field_names = _specifics.__func__()[2] #class attributes

    def _getpathfromcsv(self, path):
        #convert csv to dict
        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            datadict= {col: [] for col in reader.fieldnames}
            for i, row in enumerate(reader):
                for col in reader.fieldnames:
                    datadict[col].append(row[col])
        return datadict
    
    def _get_directory_stats(self, path):
        if self.case_str is None:
            assert self.type=="SR", "only support SR, while got %s"%self.type
            self.datadict = self._getpathfromcsv(path)
        assert self.split_level=="snapshot"
        assert self.leadtime_max==0 and self.nsteps_input==1
        #split into train/val/test at snapshot level
        self.total_cases = len(self.datadict['hash'])
        self.casesids_split = self._split_cases()
        self.len=len(self.casesids_split)
        self.datapath=os.path.dirname(path)
        assert all(element == self.SR_ratio[0] for element in self.SR_ratio)
        self.outputbase = self.datapath+'/dataset/HR/'+self.split+'/'
        if self.SR_ratio[0] in [8, 16, 32]:
            self.upscale = self.SR_ratio[0]  
            self.inputbase = self.datapath+'/dataset/LR_'+str(self.upscale)+'x/'+self.split+'/'
        else:
            self.upscale = 1
            self.inputbase = self.outputbase
        if self.split=="train":
            self.mean, self.std = self.get_mean_std()
        else:#val or test
            self.mean, self.std = self.get_mean_std_test()

    #adapted from https://www.kaggle.com/code/waitongchung/momentum128-readandinfer/notebook
    #get mean/std for train/val sets
    def get_mean_std(self):
        #evaluated thorugh mean of train data in rho and (u,v,w lumped together)
        my_mean = np.array([0.24,28.0, 28.0, 28.0])
        my_std = np.array([0.068,48.0, 48.0, 48.0])

        return my_mean, my_std

    #get mean/std for test sets
    def get_mean_std_test(self):
        #evaluated thorugh mean of train data in rho and (u,v,w lumped together)
        my_mean = np.array([0.24,29.0, 29.0, 29.0])
        my_std = np.array([0.068,48.0, 48.0, 48.0])

        return my_mean, my_std

    # get mean/std for paramvar set
    def get_mean_std_paramvar(self):
        #evaluated thorugh mean of train data in rho and (u,v,w lumped together)
        my_mean = np.array([0.23,34.0, 34.0, 34.0])
        my_std = np.array([0.059,55.0, 55.0, 55.0])

        return my_mean, my_std

    # get mean/std for forcedhit set
    def get_mean_std_forcedhit(self):
        #evaluated thorugh mean of train data in rho and (u,v,w lumped together)
        my_mean = np.array([11,-0.051, -0.051, -0.051])
        my_std = np.array([4.6,1.4, 1.4, 1.4])

        return my_mean, my_std

    #normalize by mean and std
    def normalize(self,sample, mean, std):
        return (sample - mean)/std

    def rot90_3D(self,input,k,dims):
        #input is C,H,W,D
        #C is rho,u,v,w
        #dims is 0,1,2 for x,y,z
        #preserves the divergence
        assert len(dims) == 2
        for i in range(k):
            #plus 1 because of channel
            swp = (dims[0]+1,dims[1]+1)
            input = torch.rot90(input, k=1, dims=swp)
            
            #plus 1 because of rho
            input[[swp[0],swp[1]]] = input[[swp[1],swp[0]]]
            
            input[swp[0]] = -input[swp[0]]
        return input

    def rot_dx(self,dx_list,k,dims):
        assert len(dims) == 2
        for i in range(k):
            dx_list[[dims[0],dims[1]]] = dx_list[[dims[1],dims[0]]]
        return dx_list[0],dx_list[1],dx_list[2]

    def flip_3D(self,input,dims):
        #input is C,H,W,D
        #dims is 0,1,2 for x,y,z
        #plus 1 because of channel
        #preserves the divergence
        assert len(dims) == 1
        swp = [dims[0]+1]
        input = torch.flip(input, dims=swp)
        input[swp[0]] = -input[swp[0]]
        return input

    #data augmentation
    def random_rot90_3D(self,X,Y,dx,dy,dz):
        dx_list = torch.tensor([dx,dy,dz]).reshape(3,1)
        #k = np.random.randint(0,4)
        #axis = np.random.choice([0,1,2],2,replace=False)
        k = self.rng.integers(0,4)
        axis = self.rng.choice([0,1,2], size=2, replace=False)
        X=self.rot90_3D(X,k,axis)
        Y=self.rot90_3D(Y,k,axis)
        dx,dy,dz = self.rot_dx(dx_list,k,axis)
        return X,Y,dx,dy,dz

    #data augmentation
    def random_flip_3D(self,X,Y):
        for axis in range(3):
            #p = np.random.rand()
            p = self.rng.random() 
            if p > 0.5:
                X=self.flip_3D(X,[axis])
                Y=self.flip_3D(Y,[axis])
        return X,Y

    def _reconstruct_sample(self,idx):    
        hash_id = self.datadict['hash'][idx]
        scalars = self.field_names
        X = []
        upscale = self.upscale
        for ivar, scalar in enumerate(scalars):
            xpath = self.inputbase+scalar+hash_id+'.dat' 
            var = np.fromfile(xpath,dtype=np.float32).reshape(128//upscale,128//upscale,128//upscale)
            X.append(self.normalize(var, self.mean[ivar], self.std[ivar]))
        X = np.stack(X,axis=0)
        Y = []
        for ivar, scalar in enumerate(scalars):
            ypath = self.outputbase+scalar+hash_id+'.dat'
            var = np.fromfile(ypath,dtype=np.float32).reshape(128,128,128)
            Y.append(self.normalize(var, self.mean[ivar], self.std[ivar]))
        Y = np.stack(Y,axis=0)

        dx = torch.tensor(np.float32(self.datadict['dx [m]'][idx]))
        if self.datadict['dy [m]'][idx] != '':
            dy = torch.tensor(np.float32(self.datadict['dy [m]'][idx]))
        else:
            dy = dx
        if self.datadict['dz [m]'][idx] != '':
            dz = torch.tensor(np.float32(self.datadict['dz [m]'][idx]))
        else:
            dz = dx
        #manually transfrom with rotations and flips
        if self.split=="train" and self.data_augmentation:
            X,Y,dx,dy,dz = self.random_rot90_3D(torch.from_numpy(X),torch.from_numpy(Y),dx,dy,dz)
            X,Y = self.random_flip_3D(X,Y)
            X,Y = X.numpy(), Y.numpy()
        #return: T,C,D,H,W
        return X[np.newaxis,...].transpose((0, 1, 4, 2, 3)),Y[np.newaxis,...].transpose((0, 1, 4, 2, 3)), [dz, dx, dy]

    def _get_specific_bcs(self):
        #FIXME: not used for now
        return [0, 0, 0] # Non-periodic
