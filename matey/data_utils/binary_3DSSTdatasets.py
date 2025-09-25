"""
Remember to parameterize the file paths eventually
"""
import torch
import torch.nn
import numpy as np
import os
from torch.utils.data import Dataset
import h5py
import glob
from .shared_utils import get_top_variance_patchids, plot_checking
import re
from .utils import closest_factors
from functools import reduce
from operator import mul

np.random.seed(2024) 

class BaseBinary3DSSTDataset(Dataset):
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
        subname (str): Name to use for dataset
        split_level (str): 'sample' or 'file' - whether to split by samples within a file
                        (useful for data segmented by parameters) or file (mostly INS right now)
        gammaref: pick all tokens that with variances larger than gammaref*max_variance to refine
    """
    def __init__(self, path, include_string='', n_steps=1, dt=1, leadtime_max=1, split='train', 
                 train_val_test=None, extra_specific=False, tokenizer_heads=None, tkhead_name=None, SR_ratio=None,
                 group_id=0, group_rank=0, group_size=1):
        super().__init__()
        self.path = path
        self.split = split
        self.extra_specific = extra_specific # Whether to use parameters in name
        self.subname = path.split('/')[-1]
        
        self.leadtime_max = leadtime_max
        self.nsteps_input = n_steps
        self.train_val_test = train_val_test
        self.partition = {'train': 0, 'val': 1, 'test': 2}[split]
        self.time_index, self.sample_index, self.field_names, self.type, self.cubsizes, self.dt, self.tscale, self.gridsizes = self._specifics()
        self._get_directory_stats(path)
        self.title = self.type

        self.tokenizer_heads = tokenizer_heads
        self.tkhead_name=tkhead_name

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

    def _reconstruct_sample(self, file_pointers, time_idx, blk_ix, blk_iy, blk_iz, leadtime):
        """
        Function to ectract input and lable data for given time index.
        
        Args:
            file_pointers: a structure of file names. Each index has all field_names. 
                            The first nsteps_input are for input files and the 
                            last one for label/output/target
            time_idx: the begining of input time index
            blk_ix, blk_iy, blk_iz: IDs of the block at size self.cubsizes
            leadtime: time index of a future solution
        
        Returns:
            comb_xy: inputs and solutions in shape [T, C, H, W, D]
            leadtime: time index of a future solution
        """

        comb_xy = []
        
        # get input history
        time_idx = time_idx*self.dt + self.time_start
        time_list = sorted(file_pointers.keys())[:]
        input_len = len(time_list) - 1 if self.leadtime_max>0 else len(time_list)
        for it, file in enumerate(time_list[:input_len]):
            uvec_list = []
            for varid, var in enumerate(self.field_names):
                fname = file_pointers[file][varid]
                p = self._get_data_memmap(fname, blk_ix, blk_iy, blk_iz)
                uvec_list.append(p)
            comb_xy.append(np.array(uvec_list))  # Convert list to numpy array (C, H, W, D)

        # get the label
        if self.leadtime_max>0:
            # get label at leadtime
            it = len(time_list[:-1]) + leadtime - 1 # last one: label/output/target
            next_idx = time_list[0] + (it*self.dt)
            file = next_idx.numpy()[0]
            uvec_list = []
            for varid, var in enumerate(self.field_names):
                fname = file_pointers[file][varid]
                p = self._get_data_memmap(fname, blk_ix, blk_iy, blk_iz)
                uvec_list.append(p)
            comb_xy.append(np.array(uvec_list))  # Convert list to numpy array (C, H, W, D)
        else:
            # self-reconstruction case
            comb_xy.append(np.array(uvec_list))
        
        # stack all snapshots
        comb_xy = np.stack(comb_xy, axis=0) # T, C, H, W, D

        return comb_xy, leadtime.to(torch.float32)  # TODO: H, W, D -> nz (non-homogeneous), ny, nx
        
    def _extract_times(self, file_names):
        """
        Function to extract unique times from all filenames
        
        Args:
            file_names: a list of all file names
        
        Returns:
            unique_times: soreted list of unique time stamps from the filenames
        """
        pattern = r'_([0-9]+\.[0-9]+)$'
        time_pattern = re.compile(pattern)
        unique_times = set()
        for name in file_names:
            match = time_pattern.search(name)
            if match:
                unique_times.add(float(match.group(1)))

        return sorted(unique_times)
        
    def _get_filesinfo(self, file_paths):
        """
        Function to extract info about all the unique time files: 
        time step, data shapes, coordinate range
        
        Args:
            file_paths: a list of all file paths
        
        Returns:
            time_dict: dictionary of info for each unique time stamp
        """
        # print('files:', sorted(file_paths))
        t_labels = self._extract_times(file_paths)
        # print('t_labels:', t_labels)
        num_timesteps = len(t_labels)
        
        time_dict={}
        for time in t_labels:
            its = int(round(time/self.tscale))  # convert float time of filename to integer
            if its not in time_dict:
                time_dict[its] = {}
                time_dict[its]["filename"]=[]
                time_dict[its]["xrange"]=[]
                time_dict[its]["yrange"]=[]
                time_dict[its]["zrange"]=[]
            time_dict[its]["xrange"].append(self.gridsizes[0] - 2) 
            time_dict[its]["yrange"].append(self.gridsizes[1])     
            time_dict[its]["zrange"].append(self.gridsizes[2])     
            for var in self.field_names:
                filename = f"{var}_{time:0.6f}"
                time_dict[its]["filename"].append(filename)
        return time_dict
        
    def _get_directory_stats(self, path):
        """
        Function to extract file info, identify unique time stamps, 
        and total # subdomains/subsamples for traning and testing.

        Args:
            path: path to directory of the dataset
        """
        self.path = os.path.dirname(path)
        file_filter='*_*' ####p_10.040000
        self.files_paths = glob.glob(os.path.join(self.path, file_filter))
        self.files_paths = [os.path.basename(f) for f in self.files_paths]
        self.files_paths.sort()

        self.time_dict = self._get_filesinfo(self.files_paths)
        self.timesteps = list(self.time_dict.keys())
        self.timesteps.sort()
        self.time_start = self.timesteps[0]

        # for a given set of len(timesteps) solutions and input length of self.nsteps_input, 
        # the number of segments for (input, next-step prediction) is
        self.ntimesegs = len(self.timesteps)-self.nsteps_input
        if self.ntimesegs < 1:
            raise RuntimeError('Error: Path {} has {} steps, but nsteps_input is {}. Please set file steps = max allowable.'.format(path, len(self.timesteps), self.nsteps_input))

        self.file_samples_perstep=[]
        # get subsamples (i.e., number of blocks at size self.cubsizes) in each data file
        for it in self.timesteps:
            dict_t = self.time_dict[it]
            for xrange, yrange, zrange in zip(dict_t["xrange"], dict_t["yrange"], dict_t["zrange"]):
                samples_perstep = (xrange//self.cubsizes[0])*(yrange//self.cubsizes[1])*(zrange//self.cubsizes[2])
                self.file_samples_perstep.append(samples_perstep)
        assert len(list(set(self.file_samples_perstep)))==1
        # number of all samples
        self.total_samples = self.ntimesegs*self.file_samples_perstep[0]
        sampleids = np.arange(self.total_samples)
        self.sample_info={} #saving the info of the very first block of samples
        file_samples_perstep=self.file_samples_perstep[0]
        
        # keep track of file_path, block index (ix, iy, iz), and time index, of leading block in each sample with ID: isample
        for isample in sampleids:
            self.sample_info[isample] = {}
            itime = isample//file_samples_perstep
            ispace = isample%file_samples_perstep
            dict_t = self.time_dict[itime*self.dt + self.time_start]
            ix, iy, iz = -1, -1, -1
            filepath = None
            for file, xrange, yrange, zrange in zip(dict_t["filename"], dict_t["xrange"], dict_t["yrange"], dict_t["zrange"]):
                nx = (xrange)//self.cubsizes[0]
                ny = (yrange)//self.cubsizes[1]
                nz = (zrange)//self.cubsizes[2]
                if ispace>nx*ny*nz:
                    continue
                else:
                    iz = ispace//(nx*ny)
                    iy = ispace%(nx*ny)//nx
                    ix = ispace%(nx*ny)%nx
                    filepath = file
                    break
            assert filepath is not None
            self.sample_info[isample]["time"] = itime
            self.sample_info[isample]["xblock"] = ix
            self.sample_info[isample]["yblock"] = iy
            self.sample_info[isample]["zblock"] = iz
            self.sample_info[isample]["filepath"] = filepath  # This will store only the first var/file from the field_names

        # split sample indices into train/val/test
        np.random.shuffle(sampleids) 
        if self.train_val_test is not None:
            sample_per_part = np.ceil(np.array(self.train_val_test)*self.total_samples).astype(int)
            sample_per_part[2] = max(self.total_samples - sample_per_part[0] - sample_per_part[1], 0)
            partition = self.partition
            ist = sum(sample_per_part[:partition])
            iend = sum(sample_per_part[:partition+1])
            self.sampleid_split = sampleids[ist:iend]
        else:
            self.sampleid_split = sampleids
        self.len=len(self.sampleid_split)
            
    def _open_files(self, filename_0, time_idx, leadtime):
        """
        Function to store all file info for a given time index and leadtime.
        
        Args:
            filename_0: first var/file from the field_names for the time_idx
            time_idx: the begining of input time index
            leadtime: time index of a future solution
        
        Returns:
            file_pointers: a structure of file names. Each index has all field_names. 
                            The first nsteps_input are for input files and the 
                            last one for label/output/target
        """
        file_pointers = {}
        for it in range(self.nsteps_input): # first nsteps_input: input sequence
            ####p_10.040000
            next_idx = (time_idx+it) * self.dt + self.time_start
            file_pointers[next_idx] = []
            for var in self.field_names:
                old = f"{self.field_names[0]}_{(time_idx*self.dt + self.time_start)*self.tscale:.6f}"
                new = f"{var}_{next_idx*self.tscale:.6f}"
                filepath = filename_0.replace(old, new)
                filepath = f'{self.path}/{filepath}'
                try:
                    with open(filepath, 'rb') as temp_file:
                        file_pointers[next_idx].append(filepath)
                except Exception as e:
                    raise RuntimeError(f'Failed to open file {filepath}. An error occured: {e}')
        it = self.nsteps_input + leadtime - 1 # last one: label/output/target
        next_idx = (time_idx+it) * self.dt  + self.time_start
        file_pointers[next_idx] = []
        for var in self.field_names:
            old = f"{self.field_names[0]}_{(time_idx*self.dt + self.time_start)*self.tscale:.6f}"
            new = f"{var}_{next_idx*self.tscale:.6f}"
            filepath = filename_0.replace(old, new)
            filepath = f'{self.path}/{filepath}'
            try:
                with open(filepath, 'rb') as temp_file:
                    file_pointers[next_idx].append(filepath)
            except Exception as e:
                raise RuntimeError(f'Failed to open file {filepath}. An error occured: {e}')
        return file_pointers # should have all times as keys and each index will have all field_names
        
    def _get_data_fromfile(self, filepath, blk_ix, blk_iy, blk_iz):
        """
        Get data using numpy fromfile function. 
        NOTE: fromfile reads the `count` # of items (of type `dtype`) 
        in a continous manner of memory storage.
        
        Args:
            filepath: Path to raw data
            blk_ix, blk_iy, blk_iz: IDs of the block at size self.cubsizes
        
        Returns:
            datacube: 3D subdomain from raw data for given block index
        """
        # original domain resolution
        nz, ny, nx = self.gridsizes[2], self.gridsizes[1], self.gridsizes[0]
        # resolution of ROI/subcube size
        nxsl, nysl, nzsl = self.cubsizes
        # offset in grid point based on block IDs
        nxoffset = blk_ix*nxsl
        nyoffset = blk_iy*nysl
        nzoffset = blk_iz*nzsl
        # number of bytes of each data point
        nbyte = 4
        # subsampling - skip these many samples in each direction; set =1 for original resolution
        nxskip, nyskip, nzskip = 1, 1, 1
        
        nxoff = nxoffset * (nbyte)
        nyoff = nyoffset * (nx*nbyte)
        nzoff = nzoffset * (nx*ny*nbyte)
        # initial corner of the cuboid
        init = nzoff + nyoff + nxoff
        datacube = np.zeros((nxsl, nysl, nzsl), dtype=np.float32)
        # t = time.time()
        nyshift = 0   # to shift a slice of nx*ny
        for k in range(0,nzsl):
          nxshift = 0  # to shift a row of nx. Reset to 0 after 1 slice of nx*ny
          for j in range(0,nysl):
              pos = init + nxshift + nyshift
              datacube[:, j, k] = np.fromfile(filepath, dtype=np.float32, count=nxsl*nxskip, offset=pos)[::nxskip]
              nxshift += (nx*nyskip)*nbyte  # shift 1 row of nx for every ny
          nyshift += ((nx*ny)*nzskip)*nbyte  # shift 1 slice of nx*ny for every nz
        # elpsdt = time.time() - t
        # print(f'Time elapsed for loading datacube: {int(elpsdt/60)} min {elpsdt%60:.2f} sec')
        datacube = datacube.transpose(2,1,0) # transpose data to be [z, y, x]
        return datacube

    def _get_data_memmap(self, filepath, blk_ix, blk_iy, blk_iz):
        """
        Get data using numpy memmap function.
        NOTE: memmap outputs a file pointer accessing the binary data 
        in the shape provided. Need to copy data to local memory.
        
        Args:
            filepath: Path to raw data
            blk_ix, blk_iy, blk_iz: IDs of the block at size self.cubsizes
        
        Returns:
            datacube: 3D subdomain from raw data for given block index
        """
        nz, ny, nx = self.gridsizes[2], self.gridsizes[1], self.gridsizes[0]
        cbszx, cbszy, cbszz = self.cubsizes
        # Get memmap filepointer
        data_memmap = np.memmap(filepath, dtype=np.float32, mode='r', shape=(nz, ny, nx)) # NOTE: data is stored [z, y, x]
        # Extract the sub-cube
        sub_cube = data_memmap[ blk_iz*cbszz:(blk_iz+1)*cbszz,
                                blk_iy*cbszy:(blk_iy+1)*cbszy,
                                blk_ix*cbszx:(blk_ix+1)*cbszx]
        # Copy the sub-cube to a new array to avoid memory-mapping issues when processing
        datacube = sub_cube.copy()
        # Close and delete filepointer
        data_memmap._mmap.close()
        del data_memmap, sub_cube
        return datacube 
        
    def __getitem__(self, index):
        if hasattr(index, '__len__') and len(index)==2:
            leadtime=torch.tensor([index[1]], dtype=torch.int)
            index = index[0]
        else:
            leadtime=None

        sample_idx = self.sampleid_split[index]

        filepath = self.sample_info[sample_idx]["filepath"]
        time_idx = self.sample_info[sample_idx]["time"]
        ix       = self.sample_info[sample_idx]["xblock"]
        iy       = self.sample_info[sample_idx]["yblock"]
        iz       = self.sample_info[sample_idx]["zblock"]
        # print(f"getting data for filepath: {filepath}")

        if leadtime is None:
            #generate a random leadtime uniformly sampled from [1, self.leadtime_max]
            leadtime = torch.tensor([0]) # self-reconstruction case
            if self.leadtime_max>0: # not self-reconstruction case
                leadtime = torch.randint(1, min(self.leadtime_max+1, len(self.timesteps)-time_idx-self.nsteps_input+1), (1,))
        else:
            leadtime = min(leadtime, len(self.timesteps)-time_idx-self.nsteps_input)
   
        assert time_idx + self.nsteps_input + leadtime <= len(self.timesteps)

        #open image files
        file_pointers = self._open_files(filepath, time_idx.item(), leadtime.item())

        ########################################
        trajectory, leadtime = self._reconstruct_sample(file_pointers, time_idx.item(), ix, iy, iz, leadtime)
        bcs = self._get_specific_bcs()

        #start index and end size of local split for current 
        isz0, isx0, isy0    = self.blockdict["Ind_start"] # [idz, idx, idy]
        cbszz, cbszx, cbszy = self.blockdict["Ind_dim"] # [Dloc, Hloc, Wloc]
        trajectory = trajectory[:,:,isz0:isz0+cbszz,isx0:isx0+cbszx, isy0:isy0+cbszy]#T,C,Dloc,Hloc,Wloc

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

class sstF4R32Dataset(BaseBinary3DSSTDataset):
    @staticmethod
    def _specifics():
        time_index = 1 # DO NOT DELETE
        sample_index = 0 # DO NOT DELETE
        field_names = ['u', 'v', 'w', 'r'] # ['u', 'v', 'w', 'r', 'p']
        type = 'sstF4R32'
        cubsizes= [128, 128, 128] #[64, 64, 64] #[256, 256, 128] #[128, 128, 64] # [64, 64, 64]
        dt = 0.04 # dt of snapshots
        tscale = 0.002  # time-scale factor for converting to integer iterations
        dt = int(dt/tscale)  # re-scale to integer iteration value
        gridsizes = [514, 512, 256] # [nx+2, ny, nz]
        return time_index, sample_index, field_names, type, cubsizes, dt, tscale, gridsizes  # Do not change order of first 3
    field_names = _specifics()[2] #class attributes

    def _get_specific_bcs(self):
        return [0, 0] # Non-periodic
        
class sstPiF050Gn0050Dataset(BaseBinary3DSSTDataset):
    @staticmethod
    def _specifics():
        time_index = 1 # DO NOT DELETE
        sample_index = 0 # DO NOT DELETE
        field_names = ['u', 'v', 'w', 'r'] # ['u', 'v', 'w', 'r', 'p']
        type = 'sstPiF050Gn0050'
        cubsizes= [128, 128, 128] #[64, 64, 64] #[256, 32, 256] #[128, 16, 128] # [64, 64, 64]
        dt = 0.000054 # dt of snapshots
        tscale = 0.0000027  # time-scale factor for converting to integer iterations
        dt = int(dt/tscale)  # re-scale to integer iteration value
        gridsizes = [37634, 4704, 37632] # [nx+2, ny, nz]
        #assert self.leadtime_max==0 and self.nsteps_input==1, \
        #f"Assertion failed: For {type} data, only discrete snapshots are available. So leadtime should be 0. But, `n_steps`={self.nsteps_input} and `leadtime_max`={self.leadtime_max}."
        return time_index, sample_index, field_names, type, cubsizes, dt, tscale, gridsizes  # Do not change order of first 3
    field_names = _specifics()[2] #class attributes

    def _get_specific_bcs(self):
        return [0, 0] # Non-periodic