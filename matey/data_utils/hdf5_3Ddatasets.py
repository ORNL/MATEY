import torch
import torch.nn
import numpy as np
import os
from torch.utils.data import Dataset
import h5py
import glob
from .shared_utils import get_top_variance_patchids, plot_checking
import gc, psutil
from .utils import closest_factors
from functools import reduce
from operator import mul
from einops import rearrange

class BaseHDF53DDataset(Dataset):
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
    """
    def __init__(self, path, include_string='', n_steps=1, dt=1, leadtime_max=1, split='train', 
                 train_val_test=None, extra_specific=False, tokenizer_heads=None, refine_ratio=None, gammaref=None, tkhead_name=None, SR_ratio=None,
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
        self.time_index, self.sample_index, self.field_names, self.type, self.cubsizes = self._specifics()
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

    def _reconstruct_sample(self, file,  leadtime, sample_idx, time_idx, n_steps):
        raise NotImplementedError # Per dset - should be (x=(-history:local_idx+dt) so that get_item can split into x, y
    
    def _get_filesinfo(self, file_paths):
        raise NotImplementedError 

    def _get_directory_stats(self, path):
        self.files_paths = glob.glob(path + "/*.h5") + glob.glob(path + "/*.hdf5")
        self.files_paths.sort()
        #self.n_files = len(self.files_paths)
        self.time_dict=self._get_filesinfo(self.files_paths)
        self.timesteps = list(self.time_dict.keys())
        self.timesteps.sort()
        self.time_start=self.timesteps[0]

        # for a given set of len(timesteps) solutions and input length of self.nsteps_input, 
        # the number of segments for (input, next-step prediction) is
        self.ntimesegs = len(self.timesteps)-self.nsteps_input
        if self.ntimesegs < 1:
            raise RuntimeError('Error: Path {} has {} steps, but nsteps_input is {}. Please set file steps = max allowable.'.format(path, len(self.timesteps), self.nsteps_input))

        self.data_files = {}
        self.file_samples_perstep=[]
        #get subsamples (i.e., number of blocks at size self.cubsizes) in each data file
        for it in self.timesteps:
            dict_t=self.time_dict[it]
            xmax=-1; ymax=-1; zmax=-1
            samples_perfile = []
            for file, xrange, yrange, zrange in zip(dict_t["filename"], dict_t["xrange"], dict_t["yrange"], dict_t["zrange"]):
                try:
                    h5py.File(file, 'r').keys()
                    self.data_files[file]=None
                except:
                    raise RuntimeError(f'Failed to open file {file}')
                xmax = max(xmax, xrange[1])
                ymax = max(ymax, yrange[1])
                zmax = max(zmax, zrange[1])
            self.time_dict[it]["xmax"]=xmax
            self.time_dict[it]["ymax"]=ymax
            self.time_dict[it]["zmax"]=zmax
            samples_perstep = (xmax//self.cubsizes[0])*(ymax//self.cubsizes[1])*(zmax//self.cubsizes[2])
            self.file_samples_perstep.append(samples_perstep)
        #FIXME: assuming all data files contain same grid points
        assert len(list(set(self.file_samples_perstep)))==1, list(set(self.file_samples_perstep))
        #number of all samples
        self.total_samples=self.ntimesegs*self.file_samples_perstep[0]
        print(f"Samples stats: total samples, {self.total_samples}, number of times, {self.ntimesegs}, samples per step, {self.file_samples_perstep[0]}")

        sampleids = np.arange(self.total_samples)
       # sampleids= sampleids[:len(sampleids) // 10]
        #print(sampleids)

        self.sample_info={} #saving the info of the very first block of samples
      
        file_samples_perstep=self.file_samples_perstep[0]
        #keep track of file_path, block index (ix, iy, iz), and time index, of leading block in each sample with ID: isample
        for isample in sampleids:
            self.sample_info[isample] = {}
            itime = isample//file_samples_perstep
            dict_t=self.time_dict[itime+self.time_start]
            self.sample_info[isample]["time"] = itime
            ###ispace: the ispace-th sample at itime-th step
            ispace = isample%file_samples_perstep
            ix, iy, iz = -1, -1, -1
            filepath=None
            self.sample_info[isample]["xblock"]=[]
            self.sample_info[isample]["yblock"]=[]
            self.sample_info[isample]["zblock"]=[]
            self.sample_info[isample]["xcube"]=[]
            self.sample_info[isample]["ycube"]=[]
            self.sample_info[isample]["zcube"]=[]
            self.sample_info[isample]["xrange"]=[]
            self.sample_info[isample]["yrange"]=[]
            self.sample_info[isample]["zrange"]=[]
            self.sample_info[isample]["filepath"] = []
            nx = self.time_dict[it]["xmax"]//self.cubsizes[0]
            ny = self.time_dict[it]["ymax"]//self.cubsizes[1]
            nz = self.time_dict[it]["zmax"]//self.cubsizes[2]
            #nx, ny, nz samples in x, y, and z directions, respecitvely
            iz = ispace//(nx*ny) #the iz-th sample
            iy = ispace%(nx*ny)//nx #the iy-th sample
            ix = ispace%(nx*ny)%nx #the ix-th sample
            #so sample index 
            xl = ix*self.cubsizes[0]; xu = xl + self.cubsizes[0]
            yl = iy*self.cubsizes[1]; yu = yl + self.cubsizes[1]
            zl = iz*self.cubsizes[2]; zu = zl + self.cubsizes[2]
            for file, xrange, yrange, zrange in zip(dict_t["filename"], dict_t["xrange"], dict_t["yrange"], dict_t["zrange"]):
                # Check overlap in the y dimension
                overlap_x = (xrange[0] < xu) and (xl < xrange[1])
                # Check overlap in the y dimension
                overlap_y = (yrange[0] < yu) and (yl < yrange[1])
                # Check overlap in the z dimension
                overlap_z = (zrange[0] < zu) and (zl < zrange[1])
                if overlap_x and overlap_y and overlap_z:
                    #overlapping index for xrange, yrange, zrange for data extraction from file
                    ix = [max(0, xl-xrange[0]), min(xrange[1]-xrange[0], xu-xrange[0])]
                    iy = [max(0, yl-yrange[0]), min(yrange[1]-yrange[0], yu-yrange[0])]
                    iz = [max(0, zl-zrange[0]), min(zrange[1]-zrange[0], zu-zrange[0])]
                    #overlapping index for cubicle, icx, icy, icz for data storing inside samples
                    icx = [max(0, xrange[0]-xl), min(xu-xl, xrange[1]-xl)]
                    icy = [max(0, yrange[0]-yl), min(yu-yl, yrange[1]-yl)]
                    icz = [max(0, zrange[0]-zl), min(zu-zl, zrange[1]-zl)]
                    filepath=file
                    assert filepath is not None
                    assert icx[1]-icx[0]==ix[1]-ix[0], f"{icx}, {ix}, {xrange}, {xl}, {xu}"
                    assert icy[1]-icy[0]==iy[1]-iy[0], f"{icy}, {iy}, {yrange}, {yl}, {yu}"
                    assert icz[1]-icz[0]==iz[1]-iz[0], f"{icz}, {iz}, {zrange}, {zl}, {zu}"
                    self.sample_info[isample]["xblock"].append(ix)
                    self.sample_info[isample]["yblock"].append(iy)
                    self.sample_info[isample]["zblock"].append(iz)
                    self.sample_info[isample]["filepath"].append(filepath)
                    self.sample_info[isample]["xcube"].append(icx)
                    self.sample_info[isample]["ycube"].append(icy)
                    self.sample_info[isample]["zcube"].append(icz)
                    self.sample_info[isample]["xrange"].append(xrange)
                    self.sample_info[isample]["yrange"].append(yrange)
                    self.sample_info[isample]["zrange"].append(zrange)
                    #print("Pei debugging:", isample, xrange, yrange, zrange, xl, yl, zl, xu, yu, zu, ix, iy, iz,icx,icy,icz, flush=True)
            #print("Pei debugging", isample, self.sample_info[isample], self.cubsizes, nx,ny,nz, flush=True)
        if self.train_val_test is None:
            print(f'WARNING: No train/val/test split specified. Using all data for {self.split}.')
            self.sampleid_split=sampleids
        else:
            #split sample indices into train/val/test    
            sample_per_part = np.ceil(np.array(self.train_val_test)*self.total_samples).astype(int)
            sample_per_part[2] = max(self.total_samples - sample_per_part[0] - sample_per_part[1], 0)
            partition = self.partition
            ist=sum(sample_per_part[:partition])
            iend=sum(sample_per_part[:partition+1])
            #np.random.shuffle(sampleids) 
            self.sampleid_split=sampleids[ist:iend]
        self.len=len(self.sampleid_split)

    def _open_files(self, filename_0, time_idx, leadtime):
        raise NotImplementedError # 

    
    def _open_file(self, filepath):
        _file = h5py.File(filepath, 'r')
        self.data_files[filepath] = _file

    def __getitem__(self, index):
        if hasattr(index, '__len__') and len(index)==2:
            leadtime=torch.tensor([index[1]], dtype=torch.int)
            index = index[0]
        else:
            leadtime=None

        sample_idx = self.sampleid_split[index]

        filepaths = self.sample_info[sample_idx]["filepath"]
        time_idx = self.sample_info[sample_idx]["time"]
        ix       = self.sample_info[sample_idx]["xblock"]
        iy       = self.sample_info[sample_idx]["yblock"]
        iz       = self.sample_info[sample_idx]["zblock"]
        icx       = self.sample_info[sample_idx]["xcube"]
        icy       = self.sample_info[sample_idx]["ycube"]
        icz       = self.sample_info[sample_idx]["zcube"]
        if leadtime is None:
            #generate a random leadtime uniformly sampled from [1, self.leadtime_max]
            leadtime = torch.randint(1, min(self.leadtime_max+1, len(self.timesteps)-time_idx-self.nsteps_input+1), (1,))
        else:
            leadtime = min(leadtime, len(self.timesteps)-time_idx-self.nsteps_input)
        
        assert time_idx + self.nsteps_input + leadtime <= len(self.timesteps)

        #open image files
        file_pointers=[self._open_files(filepath, time_idx.item(), leadtime.item()) for filepath in filepaths]
        #print("Pei debugging", filepaths, file_pointers, sample_idx, leadtime, self.sample_info[sample_idx], flush=True)
        ########################################
        trajectory, leadtime = self._reconstruct_sample(file_pointers, time_idx.item(), ix, iy, iz, icx, icy, icz, leadtime)
        bcs = self._get_specific_bcs()
        #########################################
        #print("hdf5_3Ddatasets:", self.path, trajectory.min(), trajectory.max(), ix, iy, iz, icx, icy, icz, leadtime, flush=True)
        for tk in self.tokenizer_heads:
            if tk["head_name"] == self.tkhead_name:
                patch_size = tk["patch_size"]
                break
        #print(f"Memory summary: Data {trajectory.shape}, CUDA {torch.cuda.memory_allocated()/ 1024**3} GB; RAM {psutil.virtual_memory().used/1024**3} GB, {psutil.virtual_memory().percent} percentage", flush=True)
        if len(patch_size)==2 and (self.refine_ratio is not None or self.gammaref is not None):
            refineind =  get_top_variance_patchids(patch_size, trajectory[:-1], self.gammaref, self.refine_ratio)
            return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1], refineind, leadtime
        return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1], leadtime

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
    
    def _getoverlapblocks(self, icz, icy, icx, iblockstart, iblockend):
        """
        #(icx, icy, icz): contains [start, end] index in each sample from a single file
        #iblockstart/iblockend: contains start/end index (z,y,x) in each sample that a local rank should read
        #return:  [in order] z-y-x
        # output0: rz0:rz1, ry0:ry1, rx0:rx1: local [start, end] index for each rank
        # output1: bz0:bz1, by0:by1, bx0:bx1: the corresponding index inside each file
        """     
        isz0, isy0, isx0 = iblockstart
        isz1, isy1, isx1 = iblockend
         # intersect with local split (still global index)
        z0_i, z1_i = max(icz[0], isz0), min(icz[1], isz1) #start, end
        y0_i, y1_i = max(icy[0], isy0), min(icy[1], isy1)
        x0_i, x1_i = max(icx[0], isx0), min(icx[1], isx1)
        # Skip blocks with no overlap
        if z1_i <= z0_i or y1_i <= y0_i or x1_i <= x0_i:
            return None, None
        #shift to local coordinate
        rz0, rz1 = z0_i - isz0, z1_i - isz0
        ry0, ry1 = y0_i - isy0, y1_i - isy0
        rx0, rx1 = x0_i - isx0, x1_i - isx0
        #relative coordinate into local for uvec and p
        bz0, bz1 = z0_i - icz[0], z1_i - icz[0]
        by0, by1 = y0_i - icy[0], y1_i - icy[0]
        bx0, bx1 = x0_i - icx[0], x1_i - icx[0]
        #print("Pei debugging0,",self.group_rank, z0_i, z1_i, y0_i, y1_i, x0_i, x1_i, flush=True)
        #print("Pei debugging1,",self.group_rank, self.blockdict["Ind_start"], isz1, isx1, isy1, icz, icx, icy, flush=True)
        #print("Pei debugging2", [rz0, rz1, ry0, ry1, rx0, rx1], [bz0, bz1, by0, by1, bx0, bx1], flush=True)
        return [rz0, rz1, ry0, ry1, rx0, rx1], [bz0, bz1, by0, by1, bx0, bx1]

class isotropic1024Dataset(BaseHDF53DDataset):
    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ['Vx', 'Vy', 'Vw', 'pressure']
        type = 'isotropic1024fine'
        #cubsizes=[64, 64, 64]
        cubsizes=[128,128,128]
        #cubsizes=[192,192,192]
        #cubsizes=[256, 256, 256]
        #cubsizes=[512, 512, 512]
        #cubsizes=[1024,1024,512] #x,y,z
        #cubsizes=[1024,1024,1024]
        return time_index, sample_index, field_names, type, cubsizes
    field_names = _specifics()[2] #class attributes

    def _get_filesinfo(self, file_paths):
        time_dict={}
        for filename in file_paths:
            filenamestr=os.path.basename(filename)
            ###isotropic1024fine_t_30_30_x_1_1024_y_1_1024_z_513_1024.h5
            strsegs=filenamestr.split("_")
            its=int(strsegs[2])
            ite=int(strsegs[3])
            ixs=int(strsegs[5])
            ixe=int(strsegs[6])
            iys=int(strsegs[8])
            iye=int(strsegs[9])
            izs=int(strsegs[11])
            ize=int(strsegs[12][:-3])
            assert its==ite
            if its not in time_dict:
                time_dict[its]={}
                time_dict[its]["filename"]=[]
                time_dict[its]["xrange"]=[]
                time_dict[its]["yrange"]=[]
                time_dict[its]["zrange"]=[]
            time_dict[its]["filename"].append(filename)
            #shift the index to follow Python 0-index and right most exclusive
            time_dict[its]["xrange"].append([ixs-1, ixe])
            time_dict[its]["yrange"].append([iys-1, iye])
            time_dict[its]["zrange"].append([izs-1, ize])
        #print(time_dict.keys())
        return time_dict
    
    def _open_files(self, filename_0, time_idx, leadtime):
        #return a list of file points where the first nsteps_input are for input files and the last one for label/output/target
        file_pointers=[]
        for it in range(self.nsteps_input):
            ####isotropic1024fine_t_30_30_x_1_1024_y_1_1024_z_513_1024.h5
            filepath=filename_0.replace(f"_t_{time_idx+self.time_start}_{time_idx+self.time_start}_",f"_t_{time_idx+self.time_start+it}_{time_idx+self.time_start+it}_")
            if self.data_files[filepath] is None:
                self._open_file(filepath)
            file_pointers.append(self.data_files[filepath])
        it=self.nsteps_input+leadtime-1
        filepath=filename_0.replace(f"_t_{time_idx+self.time_start}_{time_idx+self.time_start}_",f"_t_{time_idx+self.time_start+it}_{time_idx+self.time_start+it}_")
        if self.data_files[filepath] is None:
            self._open_file(filepath)
        file_pointers.append(self.data_files[filepath])
        return file_pointers
    
    def _reconstruct_sample(self, file_pointerslist, time_idx, blk_ixs, blk_iys, blk_izs, icxs, icys, iczs, leadtime):
        #file_pointerslist: 
        # a list of list of file points where the first level of list corresponding to different files and the second levels contains time steps
        # in the second level: the first nsteps_input are for input files and the last one for label/output/target
        #(blk_ix, blk_iy, blk_iz): contains [start, end] index inside each file
        #(icx, icy, icz): contains [start, end] index for each sample
        #return solutions in shape: [T, C, Dloc, Hloc, Wloc] for current rank, self.group_rank

        cbszz, cbszx, cbszy = self.blockdict["Ind_dim"] # [Dloc, Hloc, Wloc]
        #start and end index of local split for current 
        isz0, isx0, isy0    = self.blockdict["Ind_start"] # [idz, idx, idy]
        iblockstart = [isz0, isy0, isx0 ]
        iblockend = [isz0+cbszz, isy0+cbszy, isx0+cbszx] 
        comb_xy = np.empty((len(file_pointerslist[0]), cbszz, cbszy, cbszx, 4), dtype='float32') # T, D, W, H, C
        #get input history
        time_idx = time_idx+self.time_start
        for file_pointers, blk_ix, blk_iy, blk_iz, icx, icy, icz in zip(file_pointerslist,blk_ixs, blk_iys, blk_izs, icxs, icys, iczs):
            output_index, datafile_index=self._getoverlapblocks(icz, icy, icx, iblockstart, iblockend)
            if output_index is None:
                continue
            else:
                rz0, rz1, ry0, ry1, rx0, rx1 = output_index
                bz0, bz1, by0, by1, bx0, bx1 = datafile_index
            for it, file in enumerate(file_pointers[:-1]):
                #x=file["xcoor"] #nx
                #y=file["ycoor"] #ny
                #z=file["zcoor"] #nz
                #p=file["Pressure_%04d"%(time_idx+it)] #in shape [nz, ny, nx, 1]
                #uvec =file["Velocity_%04d"%(time_idx+it)]#in shape [nz, ny, nx, 3]
                p    = file["Pressure_%04d"%(time_idx+it)][blk_iz[0]:blk_iz[1],blk_iy[0]:blk_iy[1],blk_ix[0]:blk_ix[1],:]
                uvec = file["Velocity_%04d"%(time_idx+it)][blk_iz[0]:blk_iz[1],blk_iy[0]:blk_iy[1],blk_ix[0]:blk_ix[1],:]
                comb_xy[it, rz0:rz1, ry0:ry1, rx0:rx1, 0:3] = uvec[bz0:bz1, by0:by1, bx0:bx1,:]
                comb_xy[it, rz0:rz1, ry0:ry1, rx0:rx1, 3:4] = p   [bz0:bz1, by0:by1, bx0:bx1,:]

            #get label at leadtime
            it = len(file_pointerslist[0][:-1])+leadtime.item()-1
            file=file_pointers[-1]

            p    = file["Pressure_%04d"%(time_idx+it)][blk_iz[0]:blk_iz[1],blk_iy[0]:blk_iy[1],blk_ix[0]:blk_ix[1],:]
            uvec = file["Velocity_%04d"%(time_idx+it)][blk_iz[0]:blk_iz[1],blk_iy[0]:blk_iy[1],blk_ix[0]:blk_ix[1],:]
            comb_xy[-1, rz0:rz1, ry0:ry1, rx0:rx1, 0:3] = uvec[bz0:bz1, by0:by1, bx0:bx1,:]
            comb_xy[-1, rz0:rz1, ry0:ry1, rx0:rx1, 3:4] = p   [bz0:bz1, by0:by1, bx0:bx1,:]

        #return: T,C,D,H,W
        return comb_xy.transpose((0, 4, 1, 3, 2)), leadtime.to(torch.float32)
    def _get_specific_bcs(self):
        return [0, 0] # Non-periodic

class TaylorGreen(BaseHDF53DDataset):
    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ["r", "u", "v", "w"] #density (r), velocity in 3 directions (u,v,w)
        type = 'taylorgreen'
        #cubsizes=[256, 256, 256] #nx, ny,nz
        cubsizes=[-1,-1,-1] #[512, 512, 256]; [768, 768, 384]; [1024, 1024, 512]
        return time_index, sample_index, field_names, type, cubsizes
    field_names = _specifics()[2] #class attributes

    def _get_filesinfo(self, file_paths):
        time_dict={}
        self.dtscale=0.040 #dt=0.002; nt=20
        for filename in file_paths:
            ###ruvw_nt14840.h5
            with h5py.File(filename, "r") as hf:
                time = hf["time"][0]
                [nz,ny,nx]=list(hf["nznynx"][:])
            filenamestr=os.path.basename(filename)
            its = int(filenamestr[len("ruvw_nt"):-3])//20
            if its not in time_dict:
                time_dict[its]={}
                time_dict[its]["filename"]=[]
                time_dict[its]["xrange"]=[]
                time_dict[its]["yrange"]=[]
                time_dict[its]["zrange"]=[]
            time_dict[its]["filename"].append(filename)
            #shift the index to follow Python 0-index and right most exclusive
            time_dict[its]["xrange"].append([0, nx])
            time_dict[its]["yrange"].append([0, ny])
            time_dict[its]["zrange"].append([0, nz])
            if self.cubsizes==[-1,-1,-1]:
                self.cubsizes=[nx,ny,nz]
                #self.cubsizes=[nz,nz,nz]
        return time_dict
    
    def _open_files(self, filename_0, time_idx, leadtime):
        #return a list of file points where the first nsteps_input are for input files and the last one for label/output/target
        file_pointers=[]
        for it in range(self.nsteps_input):
            ####ruvw_nt260.h5
            filepath=filename_0.replace(f"_nt{(time_idx+self.time_start)*20}.h5",f"_nt{(time_idx+self.time_start+it)*20}.h5")
            if self.data_files[filepath] is None:
                self._open_file(filepath)
            file_pointers.append(self.data_files[filepath])
        it=self.nsteps_input+leadtime-1
        filepath=filename_0.replace(f"_nt{(time_idx+self.time_start)*20}.h5",f"_nt{(time_idx+self.time_start+it)*20}.h5")
        if self.data_files[filepath] is None:
            self._open_file(filepath)
        file_pointers.append(self.data_files[filepath])
        return file_pointers
    
    def _reconstruct_sample(self, file_pointerslist, time_idx, blk_ixs, blk_iys, blk_izs, icxs, icys, iczs, leadtime):
        #file_pointerslist: 
        # a list of list of file points where the first level of list corresponding to different files and the second levels contains time steps
        # in the second level: the first nsteps_input are for input files and the last one for label/output/target
        #(blk_ix, blk_iy, blk_iz): contains [start, end] index inside each file
        #(icx, icy, icz): contains [start, end] index for each sample
        #return solutions in shape: [T, C, Dloc, Hloc, Wloc] for current rank, self.group_rank

        cbszz, cbszx, cbszy = self.blockdict["Ind_dim"] # [Dloc, Hloc, Wloc]
        #start and end index of local split for current 
        isz0, isx0, isy0    = self.blockdict["Ind_start"] # [idz, idx, idy]
        iblockstart = [isz0, isy0 , isx0]
        iblockend = [isz0+cbszz, isy0+cbszy, isx0+cbszx] 

        comb_xy = np.empty((len(file_pointerslist[0]), cbszz, cbszy, cbszx, 4), dtype='float32') # T, D, W, H, C
        #get input history
        time_idx = time_idx+self.time_start
        for file_pointers, blk_ix, blk_iy, blk_iz, icx, icy, icz in zip(file_pointerslist,blk_ixs, blk_iys, blk_izs, icxs, icys, iczs):
            output_index, datafile_index=self._getoverlapblocks(icz, icy, icx, iblockstart, iblockend)
            if output_index is None:
                continue
            else:
                rz0, rz1, ry0, ry1, rx0, rx1 = output_index
                bz0, bz1, by0, by1, bx0, bx1 = datafile_index
            for it, file in enumerate(file_pointers[:-1]):
                for ivar, var in enumerate(self.field_names):
                    datavar = file[var][:,:,:] # in shape: nz, ny, nx
                    varcomp = datavar[blk_iz[0]:blk_iz[1],blk_iy[0]:blk_iy[1],blk_ix[0]:blk_ix[1]]
                    comb_xy[it,rz0:rz1, ry0:ry1, rx0:rx1,ivar] = varcomp[bz0:bz1, by0:by1, bx0:bx1]

            #get label at leadtime
            file=file_pointers[-1]
            for ivar, var in enumerate(self.field_names):
                datavar = file[var][:,:,:] # in shape: nz, ny, nx
                varcomp = datavar[blk_iz[0]:blk_iz[1],blk_iy[0]:blk_iy[1],blk_ix[0]:blk_ix[1]]
                comb_xy[-1,rz0:rz1, ry0:ry1, rx0:rx1,ivar] = varcomp[bz0:bz1, by0:by1, bx0:bx1]
        #return: T,C,D,H,W
        #print("reconstruct:", comb_xy.min(), comb_xy.max(),  blk_ixs, blk_iys, blk_izs, icxs, icys, iczs, time_idx, file_pointerslist, flush=True)
        return comb_xy.transpose((0, 4, 1, 3, 2)), leadtime.to(torch.float32)
    
    def _get_specific_bcs(self):
        return [0, 0] # Non-periodic
