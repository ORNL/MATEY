import os
import glob
from dataclasses import dataclass
from typing import  List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import re, pickle
from .utils import unwrap_leadtime_config

broken_paths = ['']

@dataclass
class TrajectoryIndex:
    """Metadata for one trajectory folder."""
    traj_path: str
    files: List[str]          # ordered snapshot files
    n_steps_total: int        # total number of snapshot files
    n_samples: int            # number of valid training samples (sliding windows)
    minmax: Tuple[np.ndarray, np.ndarray]  # (global_min[C], global_max[C])

class BaseTrajectoryDataset(Dataset):
    """
    Dataset for directory-structured trajectories:
    root/
    prefix/ #prefix={case}_{irun}
        prefix_itime0.pk   # snapshot at t=0, shape [D,H,W,C]
        prefix_itime1.pk   # snapshot at t=0, shape [D,H,W,C]
        ...
    prefix'/ #prefix={case}_{irun+1}
        ...

    Returns samples in T x C x D x H x W,
    and splits train/val/test by *trajectory folders*.

    Notes / TODOs:
    - Parameterize file paths & file extensions.
    - Add normalization if needed.
    - If your snapshots are .npz/.pt/.h5, override _load_snapshot().
    ################
    Base class for datasets organized as many trajectory subfolders.
    Returns:
        x:        (T_in,  C, D, H, W)  where T_in == n_steps
        bcs:      torch.Tensor (e.g. [0,0] or [1,1]) you can override
        y:        (T_out, C, D, H, W) #we return [t_curret:t_current+leadtime_max] for now, as autoregressive training require all samples inside each minibatch to have the same leadtime
        leadtime: torch.float32 tensor scalar
    Args:
        path: directory containing trajectory subfolders
        traj_glob: pattern to match trajectory folders (default "*")
        file_glob: pattern to match snapshot files inside each trajectory (default "*.npy")
        n_steps: number of history steps in input
        dt: stride between history steps (kept for style; currently used)
        leadtime_max: max leadtime (>=1) if leadtime is sampled randomly
        leadtime_fixed: if set, always use this leadtime
        split: "train"|"val"|"test"
        train_val_test (Tuple[float,float,float]): fractions, e.g. (0.8,0.1,0.1)
        seed: for deterministic trajectory split
        subname: name label
        extra_specific: for naming, consistent with your style
    """
    def __init__(self, path, include_string, traj_glob= "*", file_glob = "*.pk", n_steps=1, dt=1, leadtime_config={}, supportdata=None, split="train", 
                 train_val_test=None, seed = 0, subname= None, extra_specific=False, tokenizer_heads=None, tkhead_name=None, SR_ratio=None, group_id=0, group_rank=0, group_size=1):
        super().__init__()
        self.path= path
        self.traj_glob = traj_glob
        self.file_glob = file_glob
        self.n_steps = int(n_steps)
        self.dt = int(dt)
        assert self.dt==1, f"not tested self.dt={self.dt}"
        self.leadtime_max, self.leadtime_fixed, self.leadtime_returnfull = unwrap_leadtime_config(leadtime_config)
        self.split = split
        self.train_val_test = train_val_test
        self.seed = int(seed)
        self.include_string = include_string
        self.extra_specific = extra_specific

        self.subname = subname if subname is not None else os.path.basename(os.path.abspath(self.path))

        self.partition = {"train": 0, "val": 1, "test": 2}[split]

        # Dataset-specific bits
        self.scalar_names, self.vector_names, self.BiMax_names, self.spatial_dims, self.DG_num, self.type, self.split_level, self.cubsizes = self._specifics()
  
        if self.extra_specific:
            self.title = self.more_specific_title(self.type, self.path, include_string)
        else:
            self.title = self.type

        self._get_directory_stats()

        self.tokenizer_heads = tokenizer_heads
        self.tkhead_name=tkhead_name
        self.__class__.field_names = self.field_names #class attributes

    def get_name(self, full_name=False) :
        return f"{self.subname}_{self.type}" if full_name else self.type

    def more_specific_title(self, type, path, include_string):
        return type

    @staticmethod
    def _specifics():
        """
        Return:scalar_names, vector_names, tensor_names, spatial_dims, DG_num, type, split_level, cubsizes
        """
        raise NotImplementedError

    def _list_trajectory_dirs(self):
        trajs = glob.glob(os.path.join(self.path, self.traj_glob))
        trajs = [t for t in trajs if os.path.isdir(t)]
        trajs.sort()
        if self.include_string:
            trajs = [t for t in trajs if self.include_string in t]
        trajs = [t for t in trajs if t not in broken_paths]
        return trajs

    def get_itime(self, filename):
        m = re.search(r'_itime(\d+).pk', filename)
        return int(m.group(1)) if m else float('inf') 

    def _list_snapshot_files(self, traj_path):
        #file is in format: tcv_miller_scan_big_00000_itime242.pk
        files = glob.glob(os.path.join(traj_path, self.file_glob))
        files.sort(key=self.get_itime)
        assert self.get_itime(files[0])==0 and self.get_itime(files[-1])==len(files)-1, f"check {files[0], files[-1]} and the file list {files}"
        return files

    def _get_snapshot_minmax(self, traj_path):
        #file is in format: minmax_overtime.pz
        ###    np.savez(f"./{prefix}/minmax_overtime.npz",global_min=global_min,global_max=global_max)
        d = np.load(f"{traj_path}/minmax_overtime.npz")
        gmin = d["global_min"]
        gmax = d["global_max"]
        return (gmin, gmax)

    def _split_trajectories(self, traj_paths):
        if self.train_val_test is None:
            return traj_paths

        tvt = np.array(self.train_val_test, dtype=np.float64)
        tvt = tvt / tvt.sum()

        random.seed(self.seed)
        random.shuffle(traj_paths)

        n = len(traj_paths)
        sample_per_part = np.ceil(np.array(self.train_val_test)*n).astype(int)
        sample_per_part[2] = max(n - sample_per_part[0] - sample_per_part[1], 0)

        ist = sum(sample_per_part[:self.partition])
        iend = sum(sample_per_part[:self.partition+1])

        return traj_paths[ist:iend]

    def _get_directory_stats(self):
        traj_paths_all = self._list_trajectory_dirs()
        traj_paths = self._split_trajectories(traj_paths_all)

        self.trajs = []
        self.offsets = [0] #cumulative sample counts
        for tp in traj_paths:
            files = self._list_snapshot_files(tp)
            if len(files) == 0:
                continue
            
            #Number of valid sample start positions:
            #We need n_steps history (with dt) ending at time_idx (exclusive),
            #and one future label at time_idx + leadtime - 1.
            T = len(files) #files should be at time=0,1.,...,T-1
            input_len = self.n_steps * self.dt

            if self.leadtime_fixed is not None:
                #1st sample: 0, 1,...,inp_len-1, so inp_end0=inp_len-1 
                #last sample: inp_end + leadtime_fixed = T-1 -> inp_end1=T-1-leadtime_fixed
                #total samples = inp_end1-inp_end0+1 =T-(inp_len+leadtime_fixed)+1
                n_samples=T-(input_len+self.leadtime_fixed)+1
            else:
                n_samp_min=T-(input_len+self.leadtime_max)+1 #miminum would be the cases generate samples with leadtime_max
                n_samples = n_samp_min

            if n_samples <= 0:
                print(f"WARNING: not enought snapshots in {tp} for a given input_len and leadtime")
                continue

            self.trajs.append(
                TrajectoryIndex(traj_path=tp, files=files, n_steps_total=T, n_samples=n_samples, minmax=self._get_snapshot_minmax(tp))
            )
            self.offsets.append(self.offsets[-1] + n_samples)

        self.len = self.offsets[-1]

        if self.len == 0:
            raise RuntimeError(
                f"No valid samples found under {self.path}. "
                f"Check traj_glob='{self.traj_glob}', file_glob='{self.file_glob}', "
                f"n_steps={self.n_steps}, dt={self.dt}, leadtime_max={self.leadtime_max}."
            )

    def __len__(self) -> int:
        return self.len

    def _load_snapshot(self, file_path):
        raise NotImplementedError

    def _ensure_dhwc(self, arr):
        """
        Accept [D,H,W,C] or [H,W,C]. Return [D,H,W,C].
        """
        if arr.ndim == 3:
            #[H,W,C] -> [1,H,W,C]
            return arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Expected snapshot with 3 or 4 dims, got shape {arr.shape}")
        assert arr.shape[-1]==len(self.field_names), f"check data dimension {arr.shape, self.field_names}"
        return arr

    def _get_specific_bcs(self):
        return [0, 0]

    def _reconstruct_sample(self, traj, leadtime,time_idx, n_steps):
        """
        Return:
            comb: [T_in+1, C, D, H, W] as numpy
            leadtime: torch.float32 tensor
        """
        raise NotImplementedError

    def __getitem__(self, index):
        if hasattr(index, "__len__") and len(index) == 2:
            index = index[0]
            leadtime = index[1]
        else:
            leadtime = None

        traj_idx = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local_idx = index - self.offsets[traj_idx] #marking the beginning (inclusive) of input snapeshots
        traj = self.trajs[traj_idx]

        #each sample input consists of self.n_steps * self.dt snapeshots (dt is 1; not tested)
        time_idx = (self.n_steps * self.dt) + local_idx #marking the end (exclusive) of input snapeshots
        
        assert time_idx < traj.n_steps_total, f"{index, local_idx, time_idx, traj.n_steps_total, self.offsets}"

        comb, leadtime_out = self._reconstruct_sample(traj=traj, leadtime=leadtime, time_idx=time_idx, n_steps=self.n_steps)

        
        bcs = torch.as_tensor(self._get_specific_bcs(), dtype=torch.float32)
        
        #[T, C, D, H, W]
        x = comb[:self.n_steps]
        y = comb[self.n_steps:]
        return {"x": x, "bcs": bcs, "y": y, "leadtime": torch.tensor([leadtime_out]).to(torch.float32)}



class GkeyllTrajDataset(BaseTrajectoryDataset):
    """
    - Each trajectory is a subfolder
    - Each timestep is a separate file storing snapshot [D,H,W,C]
    - Input: history sequence length n_steps
    - Output: future snapshot at one leadtime
    Field names: you can supply or keep generic.
    """
    @staticmethod
    def _specifics():
        scalar_names=["field", 
          "elc_M0","elc_M1","elc_M2","elc_M2par","elc_M2perp","elc_M3par","elc_M3perp",
          "ion_M0","ion_M1","ion_M2","ion_M2par","ion_M2perp","ion_M3par","ion_M3perp"]
        vector_names=["elc_HamiltonianMoments","elc_source_HamiltonianMoments",
          "ion_HamiltonianMoments","ion_source_HamiltonianMoments",
          "mapc2p"]
        #FIXME: check what's the ion_BiMaxwellianMoments
        BiMax_names=["elc_BiMaxwellianMoments", "ion_BiMaxwellianMoments"]
        spatial_dims=3
        DG_num=8 #number of DG coefficients
        type = 'gkeylltcv'
        split_level = "traj"  #splitting by trajectory
        cubsizes = [36, 24, 16] #number of cells in each spatial dimension
        return scalar_names, vector_names, BiMax_names, spatial_dims, DG_num, type, split_level, cubsizes
    scname = _specifics()[0]; vtname=_specifics()[1]; bimname=_specifics()[2] 
    field_names = [varname+f"_DG{idg}" for varname in scname for idg in range(8)]
    field_names += [varname+str(idim)+f"_DG{idg}" for varname in vtname for idim in  range(3) for idg in range(8)]
    field_names += [varname+str(icom)+f"_DG{idg}" for varname in  bimname for icom in range(4) for idg in range(8)] 
    """
    ### keep the same other
    varnames=["field", 
          "elc_M0","elc_M1","elc_M2","elc_M2par","elc_M2perp","elc_M3par","elc_M3perp",
          "ion_M0","ion_M1","ion_M2","ion_M2par","ion_M2perp","ion_M3par","ion_M3perp",
          "elc_HamiltonianMoments","elc_BiMaxwellianMoments","elc_source_HamiltonianMoments",
          "ion_HamiltonianMoments","ion_BiMaxwellianMoments","ion_source_HamiltonianMoments"] 
    geometry_varnames=["mapc2p"]
    """
    def _get_normalize(self, minmax, data):
        #data: #[T_in+T_out, D,H,W,C]
        minvals, maxvals = minmax
        scname = self.scalar_names
        vtname= self.vector_names
        bimname=self.BiMax_names
        field_names = self.field_names
        for var in scname:
            idxs = [i for i, s in enumerate(field_names) if var+"_DG" in s]
            assert len(idxs)==self.DG_num, f"{len(idxs), self.DG_num, var, [field_names[idx] for idx in idxs]}"
            var_min=minvals[idxs].min()
            var_max=maxvals[idxs].max()
            data[:,:,:,:,idxs] = (data[:,:,:,:,idxs]-var_min)/(var_max-var_min)

        
        for var in vtname: 
            idxs = [i for i, s in enumerate(field_names) if var in s]
            assert len(idxs)==self.DG_num*self.spatial_dims, f"{len(idxs), self.DG_num*self.spatial_dims}"
            var_min=minvals[idxs].min()
            var_max=maxvals[idxs].max()
            data[:,:,:,:,idxs] = (data[:,:,:,:,idxs]-var_min)/(var_max-var_min)
        
        for var in bimname:
            idxs = [i for i, s in enumerate(field_names) if var in s]
            assert len(idxs)==self.DG_num*4, f"{len(idxs), self.DG_num*4}"
            var_min=minvals[idxs].min()
            var_max=maxvals[idxs].max()
            if var_max==var_min:
                print(f"Warning: constant field {var, var_min, var_max}", flush=True)
                continue
            data[:,:,:,:,idxs] = (data[:,:,:,:,idxs]-var_min)/(var_max-var_min)
        return data
    
    def _get_specific_bcs(self):
        #FIXME: not used for now
        return [0, 0] # Non-periodic
    
    def _load_snapshot(self, file_path, itime):
        try:
            with open(file_path, "rb") as f:
                plasma=pickle.load(f)
        except Exception as e:
            raise ValueError(f"error in loading file {file_path}: {e!r}")

    
        stateDGcoeffs=plasma["stateDGcoeffs"]
        tidx=plasma["itime"]
        datashape=plasma["nshape"] #saved as D,H,W,C,T(=1)
        grid=plasma["grid"]
        statenames=plasma["statenames"]
        assert stateDGcoeffs.shape==datashape, f"{stateDGcoeffs.shape, datashape}"
        assert itime==tidx, f"{itime, tidx}"
        assert self.scalar_names==statenames[:len(self.scalar_names)], f"{self.scalar_names, statenames[:len(self.scname)]}"
        bim_exclude=[state for state in statenames[len(self.scalar_names):] if state not in self.BiMax_names]
        assert self.vector_names==bim_exclude, f"{self.vector_names, bim_exclude}"

        return stateDGcoeffs.squeeze(-1) #squeeze in time

    def _reconstruct_sample(self, traj, leadtime, time_idx, n_steps):
        T_total = traj.n_steps_total
        #Choose leadtime
        if self.leadtime_fixed is not None:
            #lt = int(self.leadtime_fixed)
            lt_max=self.leadtime_fixed
        elif leadtime is None:
            #Uniform random leadtime in [1, leadtime_max], capped by what's available
            max_lt = min(self.leadtime_max, T_total - time_idx)
            #lt = int(torch.randint(1, max_lt + 1, (1,)).item())
            lt_max=max_lt #self.leadtime_max
        else:
            #Provided leadtime; clamp to feasible
            lt = int(leadtime.item())
            lt = min(lt, T_total - time_idx)

        #Input indices: time_idx - n_steps*dt ... time_idx-1 stepping by dt
        hist_indices = list(range(time_idx - n_steps * self.dt, time_idx, self.dt))
        #output index: time_idx + lt - 1
        #y_index = time_idx + lt - 1
        #FIXME: return all information for autoregressive training; will come back if any better methods
        y_index = list(range(time_idx, time_idx+lt_max, self.dt))

        hist_frames = []
        for ti in hist_indices:
            arr = self._ensure_dhwc(self._load_snapshot(traj.files[ti], ti))
            hist_frames.append(arr)
        #Load output frame

        #y_arr = self._ensure_dhwc(self._load_snapshot(traj.files[y_index], y_index))
        y_frames = []
        for ti in y_index:
            arr = self._ensure_dhwc(self._load_snapshot(traj.files[ti], ti))
            y_frames.append(arr)

        # Stack: [T_in, D,H,W,C] and [T_out, D,H,W,C] -> concat on time
        hist_stack = np.stack(hist_frames, axis=0)
        y_stack = np.stack(y_frames, axis=0)
        comb_dhwc = np.concatenate([hist_stack, y_stack], axis=0) #[T_in+T_out, D,H,W,C]

        comb_dhwc=self._get_normalize(traj.minmax, comb_dhwc)
        #print("Pei debugging", comb_dhwc.min(), comb_dhwc.max(), flush=True)

        # Convert to [T, C, D, H, W]
        comb = np.transpose(comb_dhwc, (0, 4, 1, 2, 3)).astype(np.float32)
        return comb, lt_max


# ---- Example usage ----
if __name__ == "__main__":
    dset = GkeyllTrajDataset(
        path="/lustre/orion/lrn037/proj-shared/gkeyll",n_steps=4,dt=1,leadtime_max=8,split="train",train_val_test=(0.8, 0.1, 0.1),seed=123)

    x, bcs, y, lt = dset[0]
    print("x:", x.shape, "bcs:", bcs, "y:", y.shape, "leadtime:", lt)
    # x: (n_steps, C, D, H, W)
    # y: (1, C, D, H, W)
