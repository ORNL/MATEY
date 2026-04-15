# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils import getblocksplitstat
from .utils import unwrap_leadtime_config, decompress_zstd, locate_leaf_chunk_file, load_zarr_metadata, list_timestep_indices

class BaseZarrDataset(Dataset):
    """Base class for data loaders for Zarr v3 chunked stores.

    This loader uses raw zstd decompression by default for the chunk files.
    In future we could change this to use xarray/zarr directly if we want to support more complex Zarr features
    """

    def __init__(self, path, include_string="", n_steps=1, dt=1, leadtime_config={}, supportdata=None, split="train", train_val_test=None, subname=None,
        tokenizer_heads=None, tkhead_name=None, SR_ratio=None, group_id=0, group_rank=0, group_size=1,):
        super().__init__()
        self.path = path
        self.split = split
        if subname is None:
            self.subname = path.split('/')[-1]
        else:
            self.subname = subname
        self.dt = dt
        self.leadtime_max, self.leadtime_fixed, self.leadtime_returnfull = unwrap_leadtime_config(leadtime_config)
        self.n_steps = n_steps
        self.include_string = include_string
        self.train_val_test = train_val_test
        self.partition = {"train": 0, "val": 1, "test": 2}[split]
        self.tokenizer_heads = tokenizer_heads
        self.tkhead_name = tkhead_name
        self.SR_ratio = SR_ratio
        self.group_id = group_id
        self.group_rank = group_rank
        self.group_size = group_size
        self.input_control_act = False

        self.time_index, self.sample_index, self.field_names, self.type, self.cubsizes = self._specifics()
        self.title = self.type
        self.blockdict = getblocksplitstat(self.group_rank, self.group_size, self.cubsizes[0], self.cubsizes[1], self.cubsizes[2])

        self.zarr_meta = load_zarr_metadata(self.path)
        self.chunks_dir = os.path.join(self.path, "c") #zarr default chunk directory is "c", but we can make this more flexible in the future if needed
        self.timestep_indices = list_timestep_indices(self.chunks_dir)
        self.n_timesteps = len(self.timestep_indices)

        _, _, _, _, self.C = self.zarr_meta["chunk_grid"]["configuration"]["chunk_shape"]

        self._get_directory_stats()
    

    def read_timestep_array(self, timestep):
        """
        Read and return the timestep array with shape (D, H, W, C).
        """
        leaf = locate_leaf_chunk_file(self.chunks_dir, timestep)
        raw = decompress_zstd(leaf)
        frame = np.frombuffer(raw, dtype="<f4").reshape(*self.cubsizes, self.C)
        return frame

    @staticmethod
    def _specifics():
        # Sets self.field_names, self.dataset_type
        raise NotImplementedError

    def get_name(self, full_name=False):
        if full_name:
            return self.subname + '_' + self.type
        return self.type

    def _get_specific_bcs(self):
        return [0, 0]

    def _reconstruct_sample(self, leadtime, time_idx, n_steps):
        raise NotImplementedError

    def _get_directory_stats(self):
        if self.n_timesteps - self.n_steps - self.leadtime_max + 1 < 1:
            raise ValueError(
                f"Dataset has {self.n_timesteps} timesteps, but n_steps={self.n_steps} and leadtime_max={self.leadtime_max} require more history than available."
            )

        self.file_lens = [self.n_timesteps]
        self.file_steps = [self.n_timesteps - self.n_steps - self.leadtime_max + 1]
        self.file_nsteps = [self.n_steps]
        self.file_samples = [1]
        self.offsets = [0, self.file_steps[0]]
        self.offsets[0] = -1
        self.datasets = [None]
        self.len = self.offsets[-1]

        if self.train_val_test is None:
            self.split_offset = 0
            self.len = self.offsets[-1]
        else:
            total = self.file_steps[0]
            train_end = int(self.train_val_test[0] * total)
            val_end = train_end + int(self.train_val_test[1] * total)
            if self.partition == 0:  # train
                self.split_offset = 0
                self.len = train_end
            elif self.partition == 1:  # val
                self.split_offset = train_end
                self.len = val_end - train_end
            else:  # test
                self.split_offset = val_end
                self.len = total - val_end

    def __len__(self):
        return self.len

    def _read_timestep(self, timestep: int) -> np.ndarray:
        return self.read_timestep_array(timestep)

    def __getitem__(self, index):
        if hasattr(index, "__len__") and len(index) == 2:
            leadtime = index[1]
            index = index[0]
        else:
            leadtime = None

        index = index + self.split_offset
        if index < 0 or index >= self.file_steps[0]:
            raise IndexError(f"Global index {index} out of range [0, {self.file_steps[0]})")

        local_idx = index - max(self.offsets[0], 0)
        assert local_idx // self.file_steps[0] == 0
        time_idx = local_idx % self.file_steps[0]

        time_idx += self.n_steps
        if leadtime is None and not self.input_control_act:
            if self.leadtime_fixed:
                leadtime = self.leadtime_max
            else:
                if self.leadtime_max > 0:
                    max_lead = min(self.leadtime_max + 1, self.n_timesteps - time_idx + 1)
                    leadtime = torch.randint(1, max_lead, (1,)).item()
                else:
                    leadtime = 0
        elif self.input_control_act:
            if leadtime is None:
                leadtime = min(self.leadtime_max, self.n_timesteps - time_idx)
            else:
                leadtime = min(leadtime, self.n_timesteps - time_idx)
        else:
            leadtime = min(leadtime, self.n_timesteps - time_idx)

        trajectory, leadtime = self._reconstruct_sample(leadtime, time_idx, self.n_steps)

        isz0, isx0, isy0 = self.blockdict["Ind_start"]
        cbszz, cbszx, cbszy = self.blockdict["Ind_dim"]
        trajectory = trajectory[:, :, isz0:isz0 + cbszz, isx0:isx0 + cbszx, isy0:isy0 + cbszy]

        if leadtime == 0:
            assert self.n_steps == 1, f"n_steps must be 1 when leadtime is 0, got {self.n_steps}"
        else:
            assert trajectory.shape[0] == self.n_steps + leadtime, f"Check data shape {trajectory.shape, self.n_steps, leadtime}"

        data_obj = {
            "x": trajectory[: self.n_steps],
            "bcs": torch.as_tensor(self._get_specific_bcs()),
            "y": trajectory[-leadtime:] if self.leadtime_returnfull else trajectory[-1:],
            "leadtime": torch.tensor([leadtime]).to(torch.float32),
        }
        return data_obj


class JHTDB_ChannelDataset(BaseZarrDataset):
    """Dataset for JHTDB channel flow stored as a Zarr v3 chunked store.

    This loader uses raw zstd decompression by default for the chunk files.
    """

    @staticmethod
    def _specifics():
        field_names = ["u", "v", "w", "p"]
        type = "jhtdbchannelflow"
        cubsizes = [192, 64, 256]  # D, H, W
        time_index = 0
        sample_index = None
        return time_index, sample_index, field_names, type, cubsizes
    field_names = _specifics()[2] #class attributes

    def _reconstruct_sample(self, leadtime, time_idx, n_steps):
        frames = []
        for t in range(time_idx - n_steps, time_idx + leadtime):
            arr = self._read_timestep(self.timestep_indices[t])
            frames.append(arr)

        trajectory = np.stack(frames, axis=0).astype(np.float32)
        trajectory = trajectory.transpose(0, 4, 1, 2, 3)  # T, C, D, H, W

        if len(self.cubsizes) == 2:
            trajectory = np.expand_dims(trajectory, axis=2)
        else:
            assert list(trajectory.shape[-3:]) == self.cubsizes, f"shape mismatch, {trajectory.shape[-3:], self.cubsizes}"

        return trajectory, leadtime