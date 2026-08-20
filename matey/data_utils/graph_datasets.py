# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.
import json
import os
from dataclasses import dataclass, asdict
import glob
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.utils import coalesce
import torch.nn.functional as F
try:
    import tensorflow as tf
    tf_exist = True
except ImportError:
    tf = None
    tf_exist = False
import random
import sys

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False 
from mpi4py import MPI 
assert not MPI.Is_initialized()

here = os.path.dirname(os.path.abspath(__file__))
#NOTE: To load this dataset, we'll need XGC_reader, which should be downloaded from 
# "https://github.com/seunghoeku/XGC_reader" to "./third_party/XGC_reader/" folder.
sys.path.insert(0, os.path.join(here, "..","..", "third_party", "XGC_reader"))
from xgc_reader import base as xgc_base
from tqdm import tqdm
import adios2 as ad2
import re
import torch.distributed as dist
from torch_geometric.utils import to_undirected
from matey.data_utils.utils import unwrap_leadtime_config, partition_graph, build_local_subgraph, GhostInfo

@dataclass(frozen=True)
class SampleId:
    """A single training sample (usually a (case, time) pair)."""
    group: str          # trajectory/case identifier for splitting
    item: str           # unique string (filename + timestep)
    path: str          # where to load raw data (case file or case directory)
    t: Optional[int] = None  # time index if applicable

class BaseCFDGraphDataset(Dataset):
    def __init__(self, path, include_string='', n_steps=1, dt=1, leadtime_config={}, supportdata=None, split='train', 
        train_val_test=None, extra_specific=False, tokenizer_heads=None, tkhead_name=None, SR_ratio=None,
        group_id=0, group_rank=0, group_size=1, use_dist=False, Norm=True, comm=None, partition_method = "metis"):

        super().__init__()
        np.random.seed(2024)

        self.path = path
        self.split = split
        self.train_val_test = train_val_test
        self.extra_specific = extra_specific
        self.include_string = include_string if len(include_string)>0 else split
        self.dt = dt
        assert self.dt==1, f"currently only support dt=1 but got {dt}"
        self.leadtime_max, self.leadtime_fixed, self.leadtime_returnfull = unwrap_leadtime_config(leadtime_config)
        #if leadtime_fixed == True, set leadtime for all samples to be constant leadtime_max
        self.nsteps_input = n_steps
        self.partition = {'train': 0, 'val': 1, 'test': 2}[split]

        self.tokenizer_heads = tokenizer_heads
        self.tkhead_name=tkhead_name
        self.group_id=group_id
        self.group_rank=group_rank
        self.group_size=group_size
        self.use_dist=bool(use_dist or (self.group_size>1))
        self.normalize=Norm
        self.comm=comm

        if self.train_val_test is None:
            self.processed_dir = self.path+f"/{split}/processed"
        else:
            self.processed_dir = self.path+f"/processed"
        self.processed_index = self.processed_dir + "/index.json"
        if self.use_dist:
            assert dist.is_available() and dist.is_initialized(), (
                "torch.distributed must be initialized when use_dist=True "
                "or group_size > 1"
            )

        self.partition_method = partition_method
        if self.train_val_test is None:
            self.partition_root = self.path + f"/{split}/partitioned_{self.group_size}"
        else:
            self.partition_root = self.path + f"/partitioned_{self.group_size}"
        os.makedirs(self.processed_dir, exist_ok=True)

        self.field_names_out, self.type, self.time_steps, self.num_node_types = self._specifics()
        self.title = self.type

        self.get_stat()
        if not os.path.exists(self.processed_index) or (self.group_size>1 and not os.path.exists(self.partition_root)):
            self.process()
        print(f"{self.group_id}, complete graph converting!", flush=True)

        self.samples = self.discover_samples()
        self.splits= self.create_splits(self.samples)
        self.active_indices = list(self.splits[self.split])

        self._minmax_features()
        
    def get_name(self):
        return self.type

    def _np(self, x):
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def _to_tensor(self, x, dtype):
        return torch.as_tensor(x, dtype=dtype)

    def _pairs_from_simplex(self, k):
        """
        Returns vertex-index pairs to connect within an element.
        - triangles (k=3): (0-1,1-2,2-0)
        - tets (k=4): all 6 edges
        - otherwise: ring
        """
        if k == 3:
            return [(0, 1), (1, 2), (2, 0)]
        if k == 4:
            return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        #generic polygon: ring
        return [(i, (i + 1) % k) for i in range(k)]

    def cells_to_edge_index(self, cells, num_nodes, undirected = True):
        """
        Build edge_index from cell connectivity.
        cells: [num_cells, k] integer node indices (triangles/tets/polygons)
        """
        cells = self._np(cells).astype(np.int64)
        assert cells.ndim == 2, f"cells must be [C,k], got shape {cells.shape}"

        C, k = cells.shape
        pairs = self._pairs_from_simplex(k)

        edges = []
        for (a, b) in pairs:
            src = cells[:, a]
            dst = cells[:, b]
            edges.append(np.stack([src, dst], axis=0))  #[2, C]
            if undirected:
                edges.append(np.stack([dst, src], axis=0))

        edge_index = np.concatenate(edges, axis=1)  #[2, E]
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

        #Drop invalid edges just in case (corrupt meshes)
        mask = (edge_index[0] >= 0) & (edge_index[0] < num_nodes) & (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        edge_index = edge_index[:, mask]

        #Coalesce (sort + unique)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        return edge_index
    @staticmethod
    def mesh_edge_attr(pos, edge_index):
        #edge features: [dx, dy ,dz, |d|]
        src, dst = edge_index
        disp = pos[dst] - pos[src] #[E, dim]
        dist = torch.linalg.norm(disp, dim=-1, keepdim=True) #[E,1]
        return torch.cat([disp, dist], dim=-1)

    def one_hot_node_type(self, node_type, num_types):
        node_type = node_type.view(-1).long()
        return F.one_hot(node_type, num_classes=num_types).to(torch.float32)

    def create_splits(self, samples):
        #split at the group/traj/config level

        groups = [s.group for s in samples]
        if self.train_val_test is not None:
            assert abs(sum(self.train_val_test) - 1.0) < 1e-6
            rng = np.random.default_rng(2024)
            unique = np.array(sorted(set(groups)))
            rng.shuffle(unique)

            n = len(unique)
            n_train = int(round(self.train_val_test[0] * n))
            n_val = int(round(self.train_val_test[1] * n))
            train_g = set(unique[:n_train])
            val_g = set(unique[n_train:n_train + n_val])

            splits = {"train": [], "val": [], "test": []}
            for i, g in enumerate(groups):
                if g in train_g:
                    splits["train"].append(i)
                elif g in val_g:
                    splits["val"].append(i)
                else:
                    splits["test"].append(i)
        else:
            #all go to self.split
            splits = {"train": [], "val": [], "test": []}
            for i, g in enumerate(groups):
                splits[self.split].append(i)

        return splits

    def __len__(self):
        return len(self.active_indices)

    def get_stat(self):
        pass

    @staticmethod
    def _specifics():
        raise NotImplementedError #

    def discover_samples(self):
        """Return all samples across all splits (split happens later)."""
        raise NotImplementedError

    def build_graph(self, sample: SampleId):
        """Convert one raw sample into a PyG Data graph."""
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def _get_specific_bcs(self):
        #FIXME: not used for now
        return torch.as_tensor([0, 0])

class MeshGraphNetsAirfoilDataset(BaseCFDGraphDataset):
    """
    PyG Dataset that:
      - reads MeshGraphNets airfoil TFRecords once with TensorFlow,
      - converts each trajectory into many PyG Data graphs (t -> t+leadtime),
      - caches them under root/processed,
      - provides train/val/test splits by trajectory.
    
    April 2026: add Distributed with help of Claude Sonnet 4.6
     - Each rank holds only a *shard* of each graph snapshot. The node 
        partition is computed once (METIS or random), stored alongside the
        processed .pt files, and reused on every subsequent run.
     - self.group_size: Number of graph partitions == number of ranks that will share one graph. 
        Each rank should load a specific shard is given by `group_rank`
     - new parameter        
        partition_method : str  "metis" (default, recommended) or "random".
    """
    @staticmethod
    def _specifics():
        """
        MeshGraphNets airfoil/cylinder_flow CFD TFRecord features:
        - 'cells: raw int32 bytes, shape [1, C, 3]
        - 'mesh_pos' : raw float32 bytes, shape [1, N, 2]
        - 'node_type': raw int32 bytes, shape [1, N, 1]
        - 'velocity' : raw float32 bytes, shape [T, N, 2]
        - 'pressure' : raw float32 bytes, shape [T, N, 1]
        """
        field_names_out = ['velocityx', 'velocityy', 'pressure']
        type = 'meshgraphnetairfoil'
        time_steps=601
        num_node_types = 5
        return field_names_out, type, time_steps, num_node_types
    field_names = ["pos_x", "pos_y"] + [f"nodetype{iht}" for iht in range(_specifics()[-1])] + _specifics()[0]

    @staticmethod
    def partition_dataset(src_processed_dir, dst_partition_root, num_parts, method = "metis", overwrite = False, rank=0, world_size=1, comm=None):
        """
        Pre-compute node partitions for every trajectory in src_processed_dir.
        Saves, for each trajectory <traj_id>:
          <dst_partition_root>/<traj_id>/node_assignment.pt   – [N] long
          <dst_partition_root>/<traj_id>/grouprank_<r>/graphdata_<t:05d>.pt
                                                               – local Data + GhostInfo
        This is a *single-process* utility; run it once before distributed
        training.  For very large datasets you can parallelise over trajectories
        by calling it from multiple processes with disjoint traj lists.
        """

        os.makedirs(dst_partition_root, exist_ok=True)
        traj_dirs = [
            d for d in sorted(os.listdir(src_processed_dir))
            if os.path.isdir(os.path.join(src_processed_dir, d))
        ]

        # stride-assign trajectories to ranks
        my_trajs = [t for i, t in enumerate(traj_dirs) if i % world_size == rank]
        if rank == 0:
            print(f"[partition_dataset] {len(traj_dirs)} trajectories total, {world_size} ranks, {len(my_trajs)} trajectories each rank, {num_parts} graph parts, method={method}",
                flush=True)
        print(f"[partition_dataset] {len(traj_dirs)} trajectories, {num_parts} parts, method={method}")

        for traj_id in my_trajs:
            traj_src = os.path.join(src_processed_dir, traj_id)
            traj_dst = os.path.join(dst_partition_root, traj_id)

            # ── compute partition from t=0 snapshot (topology is static) ──────
            assignment_path = os.path.join(traj_dst, "node_assignment.pt")
            if not overwrite and os.path.exists(assignment_path):
                node_assignment = torch.load(assignment_path, weights_only=False, map_location="cpu")
            else:
                d0 = torch.load(os.path.join(traj_src, "graphdata_00000.pt"), weights_only=False,map_location="cpu")
                node_assignment = partition_graph(d0.edge_index, d0.num_nodes, num_parts, method=method)
                os.makedirs(traj_dst, exist_ok=True)
                torch.save(node_assignment, assignment_path)

            # ── create per-rank shard dirs ─────────────────────────────────────
            for r in range(num_parts):
                os.makedirs(os.path.join(traj_dst, f"grouprank_{r}"), exist_ok=True)

            # ── process every time step ────────────────────────────────────────
            pt_files = sorted(f for f in os.listdir(traj_src) if f.startswith("graphdata_") and f.endswith(".pt"))
            for pt_name in pt_files:
                #done = all(os.path.exists(os.path.join(traj_dst, f"grouprank_{r}", pt_name)) for r in range(num_parts))
                done = all(MeshGraphNetsAirfoilDataset.checkifexist(os.path.join(traj_dst, f"grouprank_{r}", pt_name), load=True) for r in range(num_parts))
                if not overwrite and done:
                    continue
                try:
                    full_data = torch.load(os.path.join(traj_src, pt_name), weights_only=False, map_location="cpu")
                except Exception as e:
                    print(f"Failed to load file: {os.path.join(traj_src, pt_name)}", flush=True)
                    print(f"Error: {type(e).__name__}: {e}", flush=True)
                    raise

                for r in range(num_parts):
                    local_data, ghost_info = build_local_subgraph(full_data, node_assignment, r, num_parts)
                    out_path = os.path.join(traj_dst, f"grouprank_{r}", pt_name)
                    torch.save({"data": local_data, "ghost_info": ghost_info}, out_path)

            print(f"[partition_dataset] done: {traj_id}")
 
        print(f"[partition_dataset] finished at rank {rank}.")
    def _minmax_features(self):
        """
        #x: pos + node_type + velocity_t + pressure_t
        x = torch.cat([pos_t, node_type_oh, vel_t, pres_t], dim=-1)
        minimum and maximum values of node features from training set
        """
        self.min_nodefeat=Tensor([-20.0, -19.960529327392578, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        -254.21923828125, -329.4056091308594, 5337.0029296875]).view(1, -1)
        self.max_nodefeat=Tensor([20.0, 19.960529327392578, 1.0, 0.0, 1.0, 0.0, 1.0,
                                        437.01123046875, 331.3974304199219, 185220.609375]).view(1, -1)
        self.norm_mask = (self.max_nodefeat > self.min_nodefeat)

    def _find_tfrecord_files(self):
        files = sorted(glob.glob(os.path.join(self.path, "*.tfrecord")))
        assert files, f"No TFRecord files matching `*.tfrecord` under {self.path}"
        files = [filename for filename in files if self.include_string in os.path.basename(filename)]
        return files

    def decode_mgn_cfd(self, serialized_ex):
        """
        Decode one MeshGraphNets CFD into numpy arrays
        """
        if not tf_exist:
            raise RuntimeError("TensorFlow is required for loading *tfrecord data in MeshGraphNetsAirfoilDataset.")

        feat_desc={"cells": tf.io.FixedLenFeature([], tf.string),
                         "mesh_pos": tf.io.FixedLenFeature([], tf.string),
                         "node_type": tf.io.FixedLenFeature([], tf.string),
                         "velocity": tf.io.FixedLenFeature([], tf.string),
                         "pressure": tf.io.FixedLenFeature([], tf.string)}
        ex = tf.io.parse_single_example(serialized_ex, feat_desc)
        cells_raw = tf.io.decode_raw(ex["cells"], tf.int32)
        mesh_pos_raw = tf.io.decode_raw(ex["mesh_pos"], tf.float32)
        node_type_raw = tf.io.decode_raw(ex["node_type"], tf.int32)
        velocity_raw = tf.io.decode_raw(ex["velocity"], tf.float32)
        pressure_raw = tf.io.decode_raw(ex["pressure"], tf.float32)

        # shapes from MGN meta
        cells = tf.reshape(cells_raw, [-1, 3])       #[C, 3]
        mesh_pos = tf.reshape(mesh_pos_raw, [-1, 2]) #[N, 2]
        node_type = tf.reshape(node_type_raw, [-1])  #[N]
        N = node_type.shape[0]
        velocity = tf.reshape(velocity_raw, [-1, N, 2]) #[T, N, 2]
        pressure = tf.reshape(pressure_raw, [-1, N, 1]) #[T, N, 1]

        return {
            "cells": cells.numpy().astype(np.int32),
            "mesh_pos": mesh_pos.numpy().astype(np.float32),
            "node_type": node_type.numpy().astype(np.int32),
            "velocity": velocity.numpy().astype(np.float32),
            "pressure": pressure.numpy().astype(np.float32)
            }
    @staticmethod
    def checkifexist(filename, load=False):
        if not os.path.exists(filename):
            return False
        else:
            if not load:
                return True
            else:
                try:
                    torch.load(filename, weights_only=False, map_location="cpu")
                    return True
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
                    return False

    def get_stat(self):
        pass

    def process(self):
        """
        Stage 1
        Run once: TFRecord -> many Data graphs, saved as per-timestep .pt files,
        plus index.json with sample metadata and splits.
        Stage 2
        Partition every full graph into per-rank shards.
        """
        tf_files_all = self._find_tfrecord_files()
        if self.use_dist:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            tf_files = [f for i, f in enumerate(tf_files_all) if i % world_size == rank]
        else:
            tf_files = tf_files_all

        samples_local = []

        for tf_path in tf_files:
            ds = tf.data.TFRecordDataset(str(tf_path))
            for ex_idx, serialized in enumerate(ds):
                arrays = self.decode_mgn_cfd(serialized)
                cells = arrays["cells"]         #[CELL, 3]
                mesh_pos = arrays["mesh_pos"]   #[N, 2]
                node_type = arrays["node_type"] #[N]
                vel = arrays["velocity"]        #[T, N, 2]
                pres = arrays["pressure"]       #[T, N, 1]

                assert self.num_node_types > np.amax(node_type), f"{self.num_node_types, np.unique(node_type)}"

                T, N = vel.shape[0], vel.shape[1]
                if self.time_steps is None:
                    self.time_steps = T
                else:
                    assert self.time_steps==T, f"{self.time_steps, T}"

                assert pres.shape[0] == T and pres.shape[1] == N, f"{pres.shape, vel.shape}"

                stem = os.path.splitext(os.path.basename(tf_path) )[0]
                traj_id = f"{stem}_ex{ex_idx}"
                pos = self._to_tensor(mesh_pos, torch.float32)
                edge_index = self.cells_to_edge_index(cells, num_nodes=N, undirected=True)
                edge_attr = BaseCFDGraphDataset.mesh_edge_attr(pos, edge_index)

                node_type_t = torch.as_tensor(node_type, dtype=torch.long)
                node_type_oh = self.one_hot_node_type(node_type_t, self.num_node_types)

                for t in range(0, self.time_steps):
                    pt_name = f"graphdata_{t:05d}.pt"
                    filename = f"{self.processed_dir}/{traj_id}/{pt_name}"
                    samples_local.append(SampleId(group=traj_id, item=pt_name, path=filename, t=t))
                    if MeshGraphNetsAirfoilDataset.checkifexist(filename): 
                        continue
                    pos_t = pos
                    vel_t = self._to_tensor(vel[t], torch.float32) #[N,2]
                    pres_t = self._to_tensor(pres[t].reshape(N, -1), torch.float32)

                    #x: pos + node_type + velocity_t + pressure_t
                    x = torch.cat([pos_t, node_type_oh, vel_t, pres_t], dim=-1)
                    data = Data(x=x, pos=pos_t, edge_index=edge_index, edge_attr=edge_attr)
                    data.group = traj_id
                    data.t = int(t)
                    data.dt = int(self.dt)

                    os.makedirs(f"{self.processed_dir}/{traj_id}", exist_ok=True)
                    torch.save(data, filename)

                    #print("Pei debugging", filename, x.shape, edge_attr.shape, edge_attr.shape, flush=True)

        if self.use_dist:
            local_dicts = [s.__dict__ for s in samples_local]
            if rank == 0:
                all_dicts = [None for _ in range(world_size)]
            else:
                all_dicts = None

            dist.gather_object(local_dicts, object_gather_list=all_dicts, dst=0)
            if rank == 0:
                flat_dicts = [d for chunk in all_dicts for d in chunk]
                index_obj = {
                    "version": 1,
                    "num_samples": len(flat_dicts),
                    "samples": flat_dicts,
                }
                with open(self.processed_index, "w") as f:
                    json.dump(index_obj, f, indent=2)
            dist.barrier()
        else:
            index_obj = {
                "version": 1,
                "num_samples": len(samples_local),
                "samples": [s.__dict__ for s in samples_local],
            }
            with open(self.processed_index, "w") as f:
                json.dump(index_obj, f, indent=2)
        if self.group_size>1:
            # ── Stage 2: partition full graphs ─────────────────────────────────────
            if self.use_dist:
                self._run_partitioning(rank=rank, world_size=world_size)
                dist.barrier()
            else:
                self._run_partitioning()

    def _run_partitioning(self, rank=0, world_size=1):
        self.partition_dataset(
            src_processed_dir = self.processed_dir, dst_partition_root = self.partition_root,
            num_parts = self.group_size, method = self.partition_method, overwrite = False,
            rank=rank, world_size=world_size
            )

    def _load_times(self, case_dir):
        times=[]
        for pt_path in sorted(os.listdir(case_dir)):
            base = pt_path  #'graphdata_00012.pt'
            times.append(int(base[len("graphdata_"):-len(".pt")]))  #'00012'
        return times

    def discover_samples(self):
        """
        Discover samples from the *partitioned* shards for this rank.
        Falls back to full processed dir if partitioning hasn't run yet.
        """
        split_dir_exists = os.path.isdir(self.partition_root)
        samples = []
        src_root = self.partition_root if split_dir_exists else self.processed_dir
 
        for cdir in sorted(os.listdir(src_root)):
            full_path = os.path.join(src_root, cdir)
            if not os.path.isdir(full_path):
                continue
            # For partitioned layout the shard dir is grouprank_<r>/ inside traj dir
            rank_shard = os.path.join(full_path, f"grouprank_{self.group_rank}")
            if os.path.isdir(rank_shard):
                shard_path = rank_shard
            else:
                if self.group_size==1:
                    #full graph (single-rank mode)
                    shard_path = full_path
                else:
                    raise RuntimeError(f"Error: shard graph dir {rank_shard} is not found on  {self.group_rank} of {self.group_id}")
 
            times = self._load_times(shard_path)
            T     = len(times)
            for t in range(0, T - self.nsteps_input - self.leadtime_max + 1):
                pt_name = f"graphdata_{t:05d}.pt"
                samples.append(SampleId(group=cdir, item=pt_name, path=os.path.join(shard_path, pt_name), t=t))
        return samples

    def len(self):
        return len(self.active_indices)

    def norm_data(self, data):
        #data: [N,C]
        data_norm = (data - self.min_nodefeat)/torch.clamp_min(self.max_nodefeat-self.min_nodefeat, 1e-8)
        return torch.where(self.norm_mask, data_norm, data)
    
    def _load_shard(self, shard_path):
        """
        Load a shard .pt file.  Returns (local_data, ghost_info).
        Handles both partitioned format (dict with 'data'/'ghost_info')
        and legacy full-graph format (plain Data).
        out_path = os.path.join(traj_dst, f"grouprank_{r}", pt_name)
        torch.save({"data": local_data, "ghost_info": ghost_info}, out_path)
        """
        obj = torch.load(shard_path, weights_only=False, map_location="cpu")
        if isinstance(obj, dict) and "data" in obj:
            return obj["data"], obj["ghost_info"]
        #single-rank fallback: no ghost info
        N = obj.num_nodes
        owned_mask = torch.ones(N, dtype=torch.bool)
        ghost_info = GhostInfo(
            #owned_mask       = owned_mask,
            ghost_rank       = torch.empty(0, dtype=torch.long),
            ghost_remote_idx = torch.empty(0, dtype=torch.long),
            local_ghost_idx  = torch.empty(0, dtype=torch.long),
            send_rank        = [],
            send_local_idx   = [],
            recv_counts      = {},
        )
        return obj, ghost_info

    def __getitem__(self, index):
        if hasattr(index, '__len__') and len(index)==2:
            leadtime=index[1]
            index = index[0]
        else:
            leadtime=None

        base_idx = self.active_indices[index]
        meta = self.samples[base_idx]
        inp_startt = int(meta.t) #inclusive
        n_in = int(self.nsteps_input)
        inp_endt=inp_startt+n_in #exclusive

        if leadtime is None:
            leadtime = 1
            if self.leadtime_fixed:
                leadtime = self.leadtime_max
        else:
            leadtime = min(leadtime, self.time_steps-inp_endt+1)

        # input times: [t, t+1, ..., t + nsteps_input - 1]
        input_times = [inp_startt + k for k in range(n_in)]
        # target time: t + nsteps_input + leadtime
        target_t = inp_endt+ leadtime-1

        group = meta.group
        case_dir = os.path.dirname(meta.path)
        case_dir = re.sub(r"grouprank_\d+", f"grouprank_{self.group_rank}", case_dir)

        x_list = []
        pos = edge_index = edge_attr = None
        for t in input_times:
            pt_path = os.path.join(case_dir, f"graphdata_{t:05d}.pt")
            d_t, g_info = self._load_shard(pt_path)

            # collect features; topology & pos are static, so take from any step
            x_list.append(self.norm_data(d_t.x))
            if pos is None:
                pos = d_t.pos
                edge_index = d_t.edge_index
                edge_attr = d_t.edge_attr
            else:
                assert (pos == d_t.pos).all(), f"{pos, d_t.pos}"
                assert (edge_index==d_t.edge_index).all()
                assert (edge_attr==d_t.edge_attr).all()
        #shape: [nsteps_input, N, C] --> [N, nsteps_input, C]
        x_seq = torch.stack(x_list, dim=0).permute(1, 0, 2)

        target_path = os.path.join(case_dir, f"graphdata_{target_t:05d}.pt")
        d_y, _ = self._load_shard(target_path)
        #d_y.x layout: [pos(2), node_type_oh(K), vel(2), pres(1)]
        #velocity and pressure as target
        # [N, 3]
        y_xnorm = self.norm_data(d_y.x)
        y_state = y_xnorm[:, -3:]

        data = Data(x=x_seq, y=y_state, pos=pos, edge_index=edge_index, edge_attr=edge_attr, case=group)
        #data.x_seq = x_seq #[N, nsteps_input, F]
        #data.y = y_state #[N, 3] -> (vx, vy, p)
        data.t0 = inp_startt
        data.target_t = target_t
        data.group = group
        data.dt = int(self.dt)
        data.leadtime = torch.tensor([leadtime]).reshape(-1,1).to(torch.float32)
        bcs = self._get_specific_bcs()
        #print("Pei debugging", g_info, flush=True)
        return {"graph": data, "bcs": bcs, "field_labels_out": [self.field_names.index(x) for x in self.field_names_out], 
                "ghost_info" : g_info,             # for HaloExchange
                }

class GraphXGCDataset(BaseCFDGraphDataset):
    @staticmethod
    def _specifics():
        field_names_out = ['e_den', 'e_T_perp', 'e_T_para', 'e_u_para', 'i_T_perp','i_T_para', 'i_u_para', 'dpot']
        #Electron density, Electron perpendicular temperature, Electron parallel temperature, Electron parallel flow velocity
        #Ion perpendicular temperature, Ion parallel temperature, Ion parallel flow velocity
        #Electrostatic potential
        type = 'graphxgc'
        time_steps = None
        num_node_types = -1 # not used for now in this dataset
        return field_names_out, type, time_steps, num_node_types
    field_names = ["pos_r", "pos_z", "pos_phi"] + _specifics()[0]
    cases ={
            "n585pe_NT_XGC1_d3d_flow5_ti_d05_tanh": None,
            #"n560fr_ITER_PFPO_W_Ne": None, #HIP Memory, DataBatch(x=[40889504, 3, 11], edge_index=[2, 163463776],..
            "n613fr_KSTART_30306_q4_rmp_turbulence": None,
            "n565pe_PT_xgc1_d3d_adjust_flow2_for_C": None,
        }
    def _minmax_features(self):
        """
        #x: [x.mesh.r, x.mesh.z, 'e_den', 'e_T_perp', 'e_T_para', 'e_u_para', 'i_T_perp','i_T_para', 'i_u_para', 'dpot']
        minimum and maximum values of node features from training set
        """
        ## ITER pos min, max: [ 4.1053, -4.5743,  0.0000], [8.3938, 4.7196, 3.1416]
        ## KSTAR pos min, max: [ 1.2650, -1.4290,  0.0000], [2.3160, 1.0900, 6.2832]
        self.max_nodefeat={
            'n585pe_NT_XGC1_d3d_flow5_ti_d05_tanh':  Tensor([2.351099967956543, 0.8992699980735779, 2.9452431201934814, 1.1001165878696954e+21, 2133.937744140625, 2140.439697265625, 2357743.0, 1706.126708984375, 1713.288818359375, 96034.390625, 173.15145874023438]).view(1,-1),
            'n560fr_ITER_PFPO_W_Ne':  Tensor([8.393799781799316, 4.719600200653076, 3.0434179306030273, 3.7512120606633224e+22, 14578.7744140625, 14706.5888671875, 11718462.0, 8026.208984375, 8097.83984375, 200799.109375, 6986.49462890625]).view(1,-1),
            'n613fr_KSTART_30306_q4_rmp_turbulence':  Tensor([2.315999984741211, 1.090000033378601, 6.086835861206055, 8.685486550500246e+19, 2792.277587890625, 2803.875244140625, 714060.125, 2821.074951171875, 2827.564208984375, 169244.140625, 250.508056640625]).view(1,-1),
            'n565pe_PT_xgc1_d3d_adjust_flow2_for_C': Tensor([2.382699966430664, 0.8992699980735779, 2.9452431201934814, 2.6752822938461505e+21, 2134.515625, 2140.7275390625, 776236.0625, 1689.1220703125, 1697.4913330078125, 90799.984375, 54.65144729614258]).view(1,-1),
            }
        self.min_nodefeat={
            'n585pe_NT_XGC1_d3d_flow5_ti_d05_tanh':  Tensor([1.0672999620437622, -0.972819983959198, 0.0, 2.929097687905075e+16, 0.15490242838859558, 0.1553480476140976, -1976734.0, 0.4610286355018616, 0.4615546464920044, -117564.6640625, -219.35455322265625]).view(1,-1),
            'n560fr_ITER_PFPO_W_Ne':  Tensor([4.105299949645996, -4.5742998123168945, 0.0, 5.520008791929651e+16, 0.0400489866733551, 0.05387083441019058, -11256524.0, 1.982227087020874, 0.5815111994743347, -291430.21875, -11756.615234375]).view(1,-1), 
            'n613fr_KSTART_30306_q4_rmp_turbulence':  Tensor([1.2649999856948853, -1.4290000200271606, 0.0, 1.443928276467712e+16, 1.3826431035995483, 1.2111923694610596, -758377.0, 4.98678731918335, 1.9161503314971924, -212789.28125, -203.14584350585938]).view(1,-1),
            'n565pe_PT_xgc1_d3d_adjust_flow2_for_C':  Tensor([1.148900032043457, -0.972819983959198, 0.0, 2.947293316854579e+16, 0.1550767719745636, 0.15548869967460632, -537856.875, 0.44695305824279785, 0.44690054655075073, -57457.4765625, -75.2417221069336]).view(1,-1),
        }

    def norm_data(self, data, case):
        #data: [N,C]
        data_norm = (data - self.min_nodefeat[case])/torch.clamp_min(self.max_nodefeat[case]-self.min_nodefeat[case], 1e-8)
        norm_mask = (self.max_nodefeat[case]> self.min_nodefeat[case])

        return torch.where(norm_mask, data_norm, data)

    def get_stat(self):
        #FIXME: hardcoded for now
        """
        time intervals are different for  each experiment:
        n560fr_ITER_PFPO_W_Ne: 2 steps
        n613fr_KSTART_30306_q4_rmp_turbulence: 2 steps
        ## We no longer use the following cases. Thus, f3d_flag should be always True
        n579fr_ASDEX_U_XGCa_neutral: 10 steps
        n582fr_ASDEX_U_fav_gradb_XGCa_neutral: 10 steps
        """        
        for case in self.cases:
            folder_path = os.path.join(self.path, case)
            if not os.path.isdir(folder_path):
                print(f"Checking path {folder_path}!")
                continue
            xgc_files = []
            indices = []
            f3d=True

            ## Not all XGC cases have delta_phi and wedge_angle info in xgc.mesh.bp or xgc.units.bp
            ## Instead, we use "fort.input.used" text file
            # with ad2.FileReader(os.path.join(self.path, case, "xgc.mesh.bp")) as f:
            #     delta_phi = f.read("delta_phi")
            #     wedge_angle = f.read("wedge_angle")
            # num_wedges = int(2.0 * np.pi / wedge_angle)
            # num_planes = int(wedge_angle/delta_phi)
            # assert num_wedges == round(2.0 * np.pi / wedge_angle)
            # assert num_planes == round(wedge_angle/delta_phi)
            with open(os.path.join(folder_path, "fort.input.used"), encoding="utf-8", errors="ignore") as f:
                text = f.read()

            nphi = re.search(r"SML_NPHI_TOTAL\s*=\s*(\d+)", text)
            wedge = re.search(r"SML_WEDGE_N\s*=\s*(\d+)", text)

            num_planes = int(nphi.group(1)) if nphi else None
            num_wedges = int(wedge.group(1)) if wedge else None
            delta_phi = 2.0 * np.pi / num_wedges / num_planes if num_planes else None
            assert num_planes is not None
            assert num_wedges is not None
            assert delta_phi is not None

            for fname in os.listdir(folder_path):
                if fname.startswith("xgc.f3d.") and fname.endswith(".bp"):
                    dpotfile=fname.replace('.f3d.','.3d.')
                    if not os.path.exists(os.path.join(folder_path, dpotfile)):
                        print(f"WARNING SKIP {os.path.join(folder_path, fname)} because the matching {dpotfile} not found", flush=True)
                        continue
                    core = fname[len("xgc.f3d."): -len(".bp")]  # e.g. "00056"
                    idx = int(core)  # convert "00056" -> 56
                    indices.append(idx)
                    full_path = os.path.join(folder_path, fname)
                    xgc_files.append(full_path)
            if len(xgc_files)==0:
                for fname in os.listdir(folder_path):
                    if fname.startswith("xgc.f2d.") and fname.endswith(".bp"):
                        dpotfile=fname.replace('.f2d.','.2d.')          
                        if not os.path.exists(os.path.join(folder_path, dpotfile)):
                            print(f"WARNING SKIP {os.path.join(folder_path, fname)} because the matching {dpotfile} not found", flush=True)
                            continue
                        core = fname[len("xgc.f2d."): -len(".bp")]  # e.g. "00056"
                        idx = int(core)  # convert "00056" -> 56
                        indices.append(idx)
                        full_path = os.path.join(folder_path, fname)
                        xgc_files.append(full_path)
                        f3d = False
            indices, xgc_files = zip(*sorted(zip(indices, xgc_files)))
            count = len(indices)
            assert int(indices[1]-indices[0]) in [2, 10], f"checking files, {indices, xgc_files}"
            self.cases[case] = {
                "count": count,
                "time_indices": indices,
                "xgc_files": xgc_files,
                "f3d_flag": f3d, #if xgc case is 3d or not
                "dt": int(indices[1]-indices[0]), #case dependent
                "num_wedges": num_wedges,
                "num_planes": num_planes,
                "delta_phi": delta_phi,
            }
    
    @staticmethod
    def checkifexist(filename, load=False):
        if not os.path.exists(filename):
            return False
        
        if not load:
            return True
        
        try:
            torch.load(filename, weights_only=False, map_location="cpu")
            return True
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return False

    @staticmethod
    def partition_dataset(src_processed_dir, dst_partition_root, num_parts, method="metis", overwrite=False, rank=0, world_size=1, comm=None):
        """
        Partition processed XGC full graphs into per-rank graph shards.

        Saves:
            dst_partition_root/<case>/node_assignment.pt
            dst_partition_root/<case>/grouprank_<r>/graphdata_<t:05d>.pt

        Phase 1 distributes cases across MPI ranks to generate one
        node_assignment.pt file per case.

        Phase 2 distributes individual (case, pt_file, partition_rank) tasks
        across MPI ranks. Therefore, different MPI ranks may process different
        graph partitions of the same full graph file.
        """
        os.makedirs(dst_partition_root, exist_ok=True)
        case_dirs = [d for d in sorted(os.listdir(src_processed_dir)) if os.path.isdir(os.path.join(src_processed_dir, d)) and (d in GraphXGCDataset.cases)]

        # Collect the .pt files for every case. All ranks construct the same ordered list.
        case_pt_files = {}
        for case in case_dirs:
            case_src = os.path.join(src_processed_dir, case)
            pt_files = sorted(f for f in os.listdir(case_src) if f.startswith("graphdata_") and f.endswith(".pt"))
            if len(pt_files) == 0:
                if rank == 0:
                    print(f"[XGC partition_dataset] skip empty case {case}", flush=True)
                continue
            case_pt_files[case] = pt_files
        total_pt_files = sum(len(pt_files) for pt_files in case_pt_files.values())
        total_partition_tasks = total_pt_files * num_parts

        if rank == 0:
            print(f"[XGC partition_dataset] {len(case_dirs)} cases, {total_pt_files} pt files, {total_partition_tasks} partition tasks, {world_size} workers, num_parts={num_parts}, method={method}",flush=True)
            for case in case_dirs:
                print(f"[XGC partition_dataset] {case}, {len(case_pt_files[case])} pt files, {case_pt_files[case][0], case_pt_files[case][-1]}",flush=True)
        # ------------------------------------------------------------------
        # Phase 1: create node_assignment.pt once per case.
        # The cases can still be distributed across ranks here because this
        # phase only performs one partitioning operation per case.
        # ------------------------------------------------------------------
        for case_idx, (case, pt_files) in enumerate(case_pt_files.items()):
            if case_idx % world_size != rank:
                continue

            case_src = os.path.join(src_processed_dir, case)
            case_dst = os.path.join(dst_partition_root, case)

            os.makedirs(case_dst, exist_ok=True)

            for r in range(num_parts):
                os.makedirs(os.path.join(case_dst, f"grouprank_{r}"), exist_ok=True)

            assignment_path = os.path.join(case_dst, "node_assignment.pt")

            if overwrite or not os.path.exists(assignment_path):
                d0_path = os.path.join(case_src, pt_files[0])
                d0 = torch.load(d0_path, weights_only=False, map_location="cpu")

                #print("Pei debugging", rank, d0_path, d0.edge_index, d0.num_nodes, num_parts, flush=True)

                node_assignment = partition_graph(d0.edge_index, d0.num_nodes, num_parts, method=method)

                # Write atomically so other ranks never see a partially written file.
                temporary_assignment_path = (f"{assignment_path}.tmp.rank_{rank}")
                torch.save(node_assignment, temporary_assignment_path)
                os.replace(temporary_assignment_path, assignment_path)

                print(f"[XGC partition_dataset] rank {rank}: created assignment for {case}", flush=True)

        # All node-assignment files must exist before pt-file processing starts.
        if comm is not None:
            comm.Barrier()

        # When using torch.distributed instead of MPI, use:
        # dist.barrier()

        # ------------------------------------------------------------------
        # Phase 2: distribute case, pt-file, and graph-partition combinations.
        # ------------------------------------------------------------------
        def output_exists(path):
            try:
                return (os.path.isfile(path) and os.path.getsize(path) > 0)
            except OSError:
                return False
        def split_contiguous(tasks, num_workers):
            quotient, remainder = divmod(len(tasks), num_workers)

            chunks = []
            start = 0

            for worker_rank in range(num_workers):
                chunk_size = quotient

                if worker_rank < remainder:
                    chunk_size += 1

                end = start + chunk_size
                chunks.append(tasks[start:end])
                start = end
            return chunks
        if comm is not None:
            if rank == 0:
                pending_tasks = []
                existing_count = 0
                for case, pt_files in case_pt_files.items():
                    case_dst = os.path.join(dst_partition_root, case)

                    for pt_name in pt_files:
                        for r in range(num_parts):
                            out_path = os.path.join(case_dst, f"grouprank_{r}", pt_name)
                            if overwrite or not output_exists(out_path):
                            #if overwrite or not GraphXGCDataset.checkifexist(out_path, load=True): 
                                pending_tasks.append((case, pt_name, r))
                            else:
                                existing_count += 1
                # Split only the missing tasks.
                task_chunks = split_contiguous(pending_tasks, world_size)

                print(f"[XGC partition_dataset] {existing_count}/{total_partition_tasks} outputs already exist; {len(pending_tasks)} outputs need processing", flush=True)

                for worker_rank, worker_tasks in enumerate(task_chunks):
                    print(f"[XGC partition_dataset] rank {worker_rank}: assigned {len(worker_tasks)} pending tasks", flush=True)

            else:
                task_chunks = None
            my_pt_tasks = comm.scatter(task_chunks, root=0)
        else:
            pending_tasks = []

            for case, pt_files in case_pt_files.items():
                case_dst = os.path.join(dst_partition_root, case)

                for pt_name in pt_files:
                    for r in range(num_parts):
                        out_path = os.path.join(case_dst, f"grouprank_{r}", pt_name)
                        if overwrite or not output_exists(out_path):
                        #if overwrite or not GraphXGCDataset.checkifexist(out_path, load=True): 
                            pending_tasks.append((case, pt_name, r))

            # Split the filtered task list, rather than all possible tasks.
            task_chunks = split_contiguous(pending_tasks, world_size)
            my_pt_tasks = task_chunks[rank]
        
        ###pt_tasks = [(case, pt_name, r) for case, pt_files in case_pt_files.items() for pt_name in pt_files for r in range(num_parts)]
        ###my_pt_tasks = pt_tasks[rank::world_size]

        print(f"[XGC partition_dataset] rank {rank}: processing {len(my_pt_tasks)} pending pt files", flush=True)


        #Keep only the currently used assignment and full graph in memory.
        #Because partition_tasks is ordered by case and then pt_name, tasks
        #belonging to the same graph generally remain adjacent on each rank.
        cached_case = None
        cached_node_assignment = None

        cached_graph_key = None
        cached_full_data = None
        for task_idx, (case, pt_name, r) in enumerate(my_pt_tasks):
            case_src = os.path.join(src_processed_dir, case)
            case_dst = os.path.join(dst_partition_root, case)
            out_path = os.path.join(case_dst, f"grouprank_{r}", pt_name)

            #done = GraphXGCDataset.checkifexist(out_path, load=True) 
            #if not overwrite and done:
            #    continue
            if not overwrite and output_exists(out_path):
                print(f"[XGC partition_dataset] rank {rank} skip newly existing output {case}/{pt_name}, grouprank_{r}", flush=True)
                continue

            if cached_case != case:
                assignment_path = os.path.join(case_dst, "node_assignment.pt")
                cached_node_assignment = torch.load(assignment_path, weights_only=False, map_location="cpu")
                cached_case = case
                cached_graph_key = None
                cached_full_data = None
            

            graph_key = (case, pt_name)
            if cached_graph_key != graph_key:
                pt_path = os.path.join(case_src, pt_name)
                try:
                    #saved fulll graph in ./processed is directed by default
                    full_data = torch.load(pt_path, weights_only=False, map_location="cpu")
                    full_data.edge_index = to_undirected(full_data.edge_index, num_nodes=full_data.pos.size(0))
                    full_data.edge_attr = BaseCFDGraphDataset.mesh_edge_attr(full_data.pos, full_data.edge_index)
                except Exception as e:
                    print(f"Failed to load file: {pt_path}", flush=True)
                    print(f"Error: {type(e).__name__}: {e}", flush=True)
                    raise
                cached_full_data = full_data
                cached_graph_key = graph_key
                
            local_data, ghost_info = build_local_subgraph(cached_full_data, cached_node_assignment, r, num_parts)

            out_path = os.path.join(case_dst, f"grouprank_{r}", pt_name)
            #torch.save({"data": local_data, "ghost_info": ghost_info}, out_path)
            output = {"data": local_data, "ghost_info": ghost_info}
            # Save each output atomically. A failed torch.save will leave only
            # a temporary file, not an apparently completed out_path.
            temporary_out_path = f"{out_path}.tmp.rank_{rank}. pid_{os.getpid()}"

            torch.save(output, temporary_out_path)
            os.replace(temporary_out_path, out_path)

            print(f"[XGC partition_dataset] rank {rank}/{world_size}, task {task_idx}/{len(my_pt_tasks)} done: {case}/{pt_name}, grouprank_{r}", flush=True)

        print(f"[XGC partition_dataset] finished rank {rank}/{world_size}.", flush=True)

    def get_nodepos_edgeindexattr(self, x, case_state, case_path):

        num_planes = case_state["num_planes"] #1 for 2d
        num_wedges = case_state["num_wedges"] #1 for 2d
        delta_phi = case_state["delta_phi"] #2pi for 2d

        num_nodes = len(x.mesh.r)
        if not case_state["f3d_flag"]:
            pos_ = np.array([x.mesh.r, x.mesh.z, np.zeros_like(x.mesh.r)])
            pos = torch.as_tensor(pos_.T, dtype=torch.float32)
            edge_index = torch.as_tensor(x.mesh.triobj.edges.T)
            edge_attr = BaseCFDGraphDataset.mesh_edge_attr(pos, edge_index)
        else:
            ## need to know nextnode
            with ad2.FileReader(os.path.join(case_path, "xgc.mesh.bp")) as f:
                nextnode = f.read("nextnode")
            mk = (nextnode != np.arange(num_nodes))
            #print("num_wedges:", num_wedges, "num_planes:", num_planes, "delta_phi", delta_phi, flush=True)
            #print("num_nodes:", num_nodes, "num edges within plane:", len(x.mesh.triobj.edges), "num edges between planes:", len(mk), flush=True)

            ## Build 3D graph by connecting nodes in adjacent planes
            pos_list = list()
            edge_list = list()
            #build graph for one wedge only
            for iphi in range(num_planes):
                pos_ = np.array([x.mesh.r, x.mesh.z, np.ones_like(x.mesh.r) * delta_phi * iphi])
                pos_list.append(pos_)
                ## within-plane edges
                edges1 = x.mesh.triobj.edges + iphi * num_nodes
                edge_list.append(edges1)
                assert np.min(edges1) >=  iphi * num_nodes
                assert np.max(edges1) < (iphi + 1) * num_nodes

                ## between-plane edges
                iphi_next = (iphi + 1) % num_planes #connecting the last plane to the first
                src = np.arange(num_nodes) +  iphi * num_nodes
                dst = nextnode + iphi_next * num_nodes
                edges2 = np.stack([src[mk], dst[mk]], axis=1)
                edge_list.append(edges2)
                #print("plane id:", iphi, "next plan", iphi_next ,"edge1:", len(edges1), "edge2:", len(edges2), flush=True)
            
            pos_ = np.concatenate(pos_list, axis=1)  #[3, N * num_planes]
            pos = torch.as_tensor(pos_.T, dtype=torch.float32)
            edges3d = np.concatenate(edge_list, axis=0)
            edge_index = torch.as_tensor(edges3d.T)
            edge_attr = BaseCFDGraphDataset.mesh_edge_attr(pos, edge_index)
            # print("num edges:", edge_index.shape[1])
            assert edge_index.shape[1] ==  len(x.mesh.triobj.edges)* num_planes + num_planes * np.sum(mk), f"{edge_index.shape[1],  len(x.mesh.triobj.edges)* num_planes + (num_planes+1) * np.sum(mk)}"
            #assert edge_index.shape[1] == (num_wedges * num_planes + 1) * len(x.mesh.triobj.edges) + (num_wedges * num_planes) * np.sum(mk)
            ## pos: [N, 3], edge_index: [2, E], edge_attr: [E, 4]
        return pos, edge_index, edge_attr, num_nodes

    def process(self):
        """
        Stage 1:
            Convert raw XGC files into full PyG graph snapshots:processed/<case>/graphdata_{timestep:05d}.pt
            ##preprocess XGC dataset into .pt files
            ##Per timestep, build a graph and save as "graphdata_{timestep:05d}.pt"
            ##build index and save to self.processed_index
        Stage 2:
            If group_size > 1, partition each full graph into:
                partitioned_<group_size>/<case>/grouprank_<r>/graphdata_{timestep:05d}.pt
        """
        if self.use_dist:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        samples_local_cases = []
        for case in self.cases:
            case_state = self.cases[case]
            traj_id = case
            case_path = os.path.join(self.path, traj_id)

            x=xgc_base.xgc1(path=case_path)
            x.setup_mesh()
            num_planes = case_state["num_planes"]
            num_wedges = case_state["num_wedges"]

            pos, edge_index, edge_attr, num_nodes = self.get_nodepos_edgeindexattr(x, case_state, case_path)
            
            time_file_pairs_all = list(zip(case_state["time_indices"], case_state["xgc_files"]))

            
            if self.use_dist:
                tf_files = [(it, f) for i, (it, f) in enumerate(time_file_pairs_all) if i % world_size == rank]
            else:
                tf_files = time_file_pairs_all

            samples_local = []
            for t, filename in tqdm(tf_files):
                pt_name = f"graphdata_{t:05d}.pt"
                filename_graph = f"{self.processed_dir}/{traj_id}/{pt_name}"
                samples_local.append(SampleId(group=traj_id, item=pt_name, path=filename_graph, t=t))
                if self.checkifexist(filename_graph, load=True):
                    print(f"{self.group_id, filename_graph} found!", flush=True)
                    continue

                node_features = [pos[:, 0], pos[:, 1], pos[:, 2]]  # r, z, phi(phi=0 for 2d)
                if not case_state["f3d_flag"]:
                    for varname in self.field_names_out[:-1]:
                        # ['e_den', 'e_T_perp', 'e_T_para', 'e_u_para', 'i_T_perp','i_T_para', 'i_u_para', 'dpot']
                        variable = x.read_one_ad2_var(filename, varname)
                        assert variable.ndim==1, f"{variable.shape}"
                        print("variable.shape", variable.shape, flush=True)
                        node_features.append(torch.as_tensor(variable, dtype=torch.float32)) #use all 32 and f3d

                    #f"xgc.2d.{t:05d}.bp" for dpot
                    dpot = x.read_one_ad2_var(filename.replace('.f2d.','.2d.'), "dpot")
                    print("dpot.shape", dpot.shape, flush=True)
                    node_features.append(torch.as_tensor(dpot.reshape(-1), dtype=torch.float32))
                else:
                    for varname in self.field_names_out[:-1]:
                        # ['e_den', 'e_T_perp', 'e_T_para', 'e_u_para', 'i_T_perp','i_T_para', 'i_u_para', 'dpot']
                        variable = x.read_one_ad2_var(filename, varname)
                        assert variable.ndim==2 and variable.shape==(num_planes, num_nodes), f"{variable.shape}"
                        print(f"{varname}, {variable.shape}", flush=True)
                        variable = variable.reshape(-1)
                        node_features.append(torch.as_tensor(variable, dtype=torch.float32))
                    #f"xgc.3d.{t:05d}.bp" for dpot
                    print(filename, flush=True)
                    dpot = x.read_one_ad2_var(filename.replace('.f3d.','.3d.'), "dpot")
                    assert dpot.ndim==2 and dpot.shape==(num_planes, num_nodes), f"{dpot.shape}"
                    print("dpot.shape", dpot.shape, flush=True)
                    dpot = dpot.reshape(-1) #order "C" - last axis index changing fastest
                    node_features.append(torch.as_tensor(dpot, dtype=torch.float32))
                
                feature = torch.stack(node_features, dim=1)
                assert feature.shape==(num_nodes*num_planes, len(self.field_names)), f"{feature.shape,(num_nodes*num_planes, len(self.field_names))}"
                #assert feature.shape==(num_nodes*(num_wedges * num_planes + 1), len(self.field_names)), f"{feature.shape,(num_nodes*(num_wedges * num_planes + 1), len(self.field_names))}"
                data = Data(x=feature, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
                data.group = traj_id
                data.t = int(t)
                data.dt = case_state["dt"] #snapshots are saved every 2 steps in xgc?
                os.makedirs(f"{self.processed_dir}/{traj_id}", exist_ok=True)
                torch.save(data, filename_graph)

            samples_local_cases.extend(samples_local)
            
        if self.use_dist:
            local_dicts = [asdict(s) for s in samples_local_cases]
            if rank == 0:
                all_dicts = [None for _ in range(world_size)]
            else:
                all_dicts = None
            dist.gather_object(local_dicts, object_gather_list=all_dicts, dst=0)

            if rank == 0:
                flat_dicts = [d for chunk in all_dicts for d in chunk]
                index_obj = {
                    "version": 1,
                    "num_samples": len(flat_dicts),
                    "samples": flat_dicts,
                }
                print("PEIPEIDIST", index_obj, flush=True)
                with open(self.processed_index, "w") as f:
                    json.dump(index_obj, f, indent=2)
            dist.barrier()        
        else:
            index_obj = {
                "version": 1,
                "num_samples": len(samples_local_cases),
                "samples": [asdict(s) for s in samples_local_cases],
            }
            print("PEIPEI", index_obj, flush=True)
            with open(self.processed_index, "w") as f:
                json.dump(index_obj, f, indent=2)

        if self.group_size > 1:
            if self.use_dist:
                self._run_partitioning(rank=rank, world_size=world_size)
                #Make sure all partition shards are written before discover_samples().
                dist.barrier()
            else:
                self._run_partitioning(rank=0, world_size=1)
    
    def _run_partitioning(self, rank=0, world_size=1):
        self.partition_dataset(
            src_processed_dir = self.processed_dir, dst_partition_root = self.partition_root,
            num_parts = self.group_size, method = self.partition_method, overwrite = False,
            rank=rank, world_size=world_size
            )           
    def create_splits(self, samples):
        #split at individual sample level
        samplepaths = [s.path for s in samples]
        assert self.train_val_test is not None, f"train_val_test is {self.train_val_test}"
        assert abs(sum(self.train_val_test) - 1.0) < 1e-6
        rng = np.random.default_rng(2024)
        unique = np.array(sorted(set(samplepaths)))
        rng.shuffle(unique)

        n = len(unique)
        n_train = int(round(self.train_val_test[0] * n))
        n_val = int(round(self.train_val_test[1] * n))
        train_g = set(unique[:n_train])
        val_g = set(unique[n_train:n_train + n_val])

        splits = {"train": [], "val": [], "test": []}
        for i, g in enumerate(samplepaths):
            if g in train_g:
                splits["train"].append(i)
            elif g in val_g:
                splits["val"].append(i)
            else:
                splits["test"].append(i)

        return splits

    def _load_times(self, case_dir):
        times=[]
        for pt_path in sorted(os.listdir(case_dir)): #
            if not (pt_path.startswith("graphdata_") and pt_path.endswith(".pt")):
                continue
            base = pt_path  #'graphdata_00012.pt'
            times.append(int(base[len("graphdata_"):-len(".pt")]))  #'00012'
        return times

    def discover_samples(self):
        """
        Discover samples from either:
            group_size == 1:
                processed/<case>/graphdata_<t:05d>.pt

            group_size > 1:
                partitioned_<group_size>/<case>/grouprank_<group_rank>/graphdata_<t:05d>.pt
        """
        
        samples = []
        if self.group_size > 1:
            src_root = self.partition_root
        else:
            src_root = self.processed_dir

        if not os.path.isdir(src_root):
            raise RuntimeError(f"Sample root does not exist: {src_root}")
        
        for cdir in sorted(os.listdir(src_root)):
            if cdir not in self.cases:
                continue
            full_path = os.path.join(src_root, cdir)
            if not os.path.isdir(full_path):
                continue
            
            if self.group_size > 1:
                shard_path = os.path.join(full_path, f"grouprank_{self.group_rank}")
                if not os.path.isdir(shard_path):
                    raise RuntimeError(f"Shard graph dir {shard_path} is not found on group_rank={self.group_rank}, group_id={self.group_id}")
                case_dir = shard_path
            else:
                case_dir = full_path

            case_state = self.cases[cdir]
            times = self._load_times(case_dir)
            #print("PEIPEI", case_dir, times, case_state, flush=True)
            T = len(times)

            #if case_state["f3d_flag"]:
            assert T==case_state["count"], f"{case_dir}, T, case_state['count'], {T, case_state['count']}"
            for it in range(0, case_state["count"] - self.nsteps_input - self.leadtime_max + 1):
                t = case_state["time_indices"][it]
                pt_name = f"graphdata_{t:05d}.pt"
                samples.append(SampleId(group=cdir, item=pt_name, path=f"{case_dir}/{pt_name}", t=t))
        return samples

    def len(self):
        return len(self.active_indices)
    
    def _load_shard(self, shard_path):
        """
        Load either a partitioned shard or a full graph.
        Returns:
            local_data, ghost_info
        """
        obj = torch.load(shard_path, weights_only=False, map_location="cpu")

        if isinstance(obj, dict) and "data" in obj:
            return obj["data"], obj["ghost_info"]

        ghost_info = GhostInfo(
            ghost_rank=torch.empty(0, dtype=torch.long),
            ghost_remote_idx=torch.empty(0, dtype=torch.long),
            local_ghost_idx=torch.empty(0, dtype=torch.long),
            send_rank=[],
            send_local_idx=[],
            recv_counts={},
        )

        return obj, ghost_info

    def __getitem__(self, index):
        if hasattr(index, '__len__') and len(index)==2:
            leadtime= index[1]
            index = index[0]

            if torch.is_tensor(leadtime):
                leadtime = int(leadtime.item())
            else:
                leadtime = int(leadtime)
        else:
            leadtime=None

        base_idx = self.active_indices[index]
        meta = self.samples[base_idx]
        group = meta.group
        dt = self.cases[group]["dt"]
        assert dt in [2,10], f"check dt in {group, self.cases[group]}"
        ## Extract plane id if graph is 2d. Otherwise, use 0 for 3d graph.
        #plane_id = base_idx % self.cases[group]["num_planes"] if not self.cases[group]["f3d_flag"] else 0

        inp_start_it = meta.t//dt -1 #inclusive; indexing and -1 since not start from 0
        n_in = int(self.nsteps_input) #actually, number of consecutive snaphots
        inp_endt_it=inp_start_it+n_in #exclusive, ending indexing
        num_snapshots = self.cases[group]["count"]

        if leadtime is None:
            #leadtime = torch.randint(1, min(self.leadtime_max+1, num_snapshots-inp_endt_it), (1,))
            #FIXME: fix leadtime_max for now
            #leadtime = max(int(self.leadtime_max)//2, 1)
            leadtime = max(int(self.leadtime_max), 1)
            if inp_endt_it + leadtime > num_snapshots:
                inp_endt_it = num_snapshots - leadtime
                inp_start_it = inp_endt_it - n_in
        else:
            leadtime = min(leadtime, num_snapshots-inp_endt_it+1)

        # input time indexing: [it, it+1, ..., it + nsteps_input - 1]
        input_times_ind = [inp_start_it + k for k in range(n_in)]
        # target time indexing: it + nsteps_input + leadtime
        target_it = inp_endt_it + leadtime - 1

        case_dir = os.path.dirname(meta.path)
        case_dir = re.sub(r"grouprank_\d+", f"grouprank_{self.group_rank}", case_dir)

        x_list = []
        pos = edge_index = edge_attr = None
        g_info = None
        self._minmax_features()
        for it in input_times_ind:
            t= (it+1)*dt
            pt_path = os.path.join(case_dir, f"graphdata_{t:05d}.pt")
            #if self.group_id==0:
            #    print(f"Pei debugging input {t}, {pt_path}", flush=True)
            d_t, g_info = self._load_shard(pt_path)

            # collect features; topology & pos are static, so take from any step
            x_list.append(self.norm_data(d_t.x, group) if self.normalize else d_t.x)
            if pos is None:
                pos = d_t.pos
                edge_index = d_t.edge_index
                edge_attr = d_t.edge_attr
            else:
                assert (pos == d_t.pos).all(), f"{pos, d_t.pos}"
                assert (edge_index==d_t.edge_index).all()
                assert (edge_attr==d_t.edge_attr).all()
        #shape: [nsteps_input, N, C] --> [N, nsteps_input, C]
        x_seq = torch.stack(x_list, dim=0).permute(1, 0, 2)

        target_path = os.path.join(case_dir, f"graphdata_{(target_it+1)*dt:05d}.pt")
        d_y, _ = self._load_shard(target_path)
        #if self.group_id==0:
        #    print(f"Pei debugging out {target_t}, {target_path}", flush=True)

        #d_y.x 
        y_xnorm = self.norm_data(d_y.x, group) if self.normalize else d_y.x 
        indices_y=[self.field_names.index(x) for x in self.field_names_out]
        y_state = y_xnorm[:, indices_y]

        #the saved ones are directed graphs in ./processed/
        edge_index = to_undirected(edge_index, num_nodes=pos.size(0))
        edge_attr = BaseCFDGraphDataset.mesh_edge_attr(pos, edge_index)

        data = Data(x=x_seq, y=y_state, pos=pos, edge_index=edge_index, edge_attr=edge_attr, case=group)
        #data.x_seq = x_seq #[N, nsteps_input, F]
        #data.y = y_state #[N, Fout]
        data.t0 = (inp_start_it+1)*dt
        data.target_t = (target_it+1)*dt
        data.group = group
        data.dt = int(dt)
        data.leadtime = torch.tensor([leadtime]).reshape(-1, 1).to(torch.float32) #number of snapshots ahead
        bcs = self._get_specific_bcs()
        return {"graph": data, "bcs": bcs, "field_labels_out": [self.field_names.index(x) for x in self.field_names_out],
                "ghost_info": g_info, # for HaloExchange
                }

   
if __name__ == "__main__":
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='airfoil', type=str)
    parser.add_argument('--nooverwrite', action='store_true', help='default: overwrite')
    parser.add_argument("--numsplits", default=8, type=int)


    args = parser.parse_args()

    def setup_mpi_rank_only():
        from mpi4py import MPI
        if not MPI.Is_initialized():
            MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        return comm, rank, world_size
    
    def mpi_allreduce_minmax(local_min, local_max, comm):
        from mpi4py import MPI
        import numpy as np
        local_min_np = np.ascontiguousarray(local_min.detach().cpu().numpy())
        local_max_np = np.ascontiguousarray(local_max.detach().cpu().numpy())

        global_min_np = np.empty_like(local_min_np)
        global_max_np = np.empty_like(local_max_np)
        comm.Allreduce(local_min_np, global_min_np, op=MPI.MIN)
        comm.Allreduce(local_max_np, global_max_np, op=MPI.MAX)
        return global_min_np, global_max_np
        
    mode = "split"   # "split", "stats"
    #for split and stats we do not need ddp and we use mpi to accelerate the processing

    #dataset = "airfoil" #
    #dataset = "xgc" #
    dataset = args.dataset
    splitgraph=True #False #
    group_size = args.numsplits #8 #4
    partition_method="metis" #"random" #
    #overwrite=True #False #
    overwrite=not args.nooverwrite
    compute_stats=True

    if dataset == "airfoil":
        DatasetClass = MeshGraphNetsAirfoilDataset
        root_path = "/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil/"
        train_val_test = None
        dt = 1
        norm = False

        split_to_include = {"train": "train","val": "val", "test": "test"}

        def processed_dir_for_split(split):
            return os.path.join(root_path, split, "processed")

        def partition_root_for_split(split):
            return os.path.join(root_path, split, f"partitioned_{group_size}")

        stats_output_dir = os.path.join(root_path, "train", "stats")

    elif dataset == "xgc":
        DatasetClass = GraphXGCDataset
        root_path = "/lustre/orion/proj-shared/fus183/fusiond-seed-xgc1-data/"
        train_val_test = [0.8, 0.1, 0.1]
        dt = 1
        norm = False

        split_to_include = {"train": "","val": "", "test": ""}
        def processed_dir_for_split():
            # XGC uses one shared processed directory, then train/val/test split
            # is applied at the sample level.
            return os.path.join(root_path, "processed")

        def partition_root_for_split():
            # Recommended for XGC: one shared partition root, not per split.
            return os.path.join(root_path, f"partitioned_{group_size}")

        stats_output_dir = os.path.join(root_path, "processed", "train_stats")
    
    comm, rank, world_size = setup_mpi_rank_only()

    if mode == "split":
        # Pick which split's processed files to partition.
        # For Airfoil, this is usually train/val/test specific.
        # For XGC, all splits share the same processed directory.
        if dataset =="airfoil":
            split_for_partitions = ["train", "val"]
            src_processed_dirs = [processed_dir_for_split(split_for_partition) for split_for_partition in split_for_partitions]
            dst_partition_roots = [partition_root_for_split(split_for_partition) for split_for_partition in split_for_partitions]
        else:
            src_processed_dirs = [processed_dir_for_split()]
            dst_partition_roots = [partition_root_for_split()]
        
        for src_processed_dir, dst_partition_root in zip(src_processed_dirs, dst_partition_roots):
            if rank == 0:
                print(f"[partition] dataset={dataset}, src_processed_dir={src_processed_dir}, dst_partition_root={dst_partition_root}", flush=True)

            DatasetClass.partition_dataset(src_processed_dir=src_processed_dir, dst_partition_root=dst_partition_root,
                num_parts=group_size, method=partition_method, overwrite=overwrite,
                rank=rank, world_size=world_size, comm=comm)

            comm.Barrier()

            if rank == 0:
                print("[partition] finished.", flush=True)

        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()
    elif mode == "stats":
        ds_train = DatasetClass(path=root_path, include_string=split_to_include["train"],
            split="train", dt=dt, train_val_test=train_val_test,
            group_id=rank, group_rank=0, group_size=1,
            use_dist=False, Norm=norm, partition_method=partition_method)

        print(f"[rank {rank}] {dataset} train samples: {len(ds_train)}; example keys: {ds_train[0].keys()}", flush=True)

        ds_valid = DatasetClass(path=root_path, include_string=split_to_include["val"],
            split="val", dt=dt, train_val_test=train_val_test,
            group_id=rank, group_rank=0, group_size=1,
            use_dist=False, Norm=norm, partition_method=partition_method)

        print(f"[rank {rank}] {dataset} train samples: {len(ds_valid)}; example keys: {ds_valid[0].keys()}", flush=True)


        ds_test = DatasetClass(path=root_path, include_string=split_to_include["test"],
            split="test", dt=dt, train_val_test=train_val_test,
            group_id=rank, group_rank=0, group_size=1,
            use_dist=False, Norm=norm, partition_method=partition_method)

        print(f"[rank {rank}] {dataset} test samples: {len(ds_test)}; example keys: {ds_test[0].keys()}", flush=True)


        if compute_stats:
            num_graphs = len(ds_train)
            assert num_graphs > 0, "Empty train dataset."

            # Use a probe sample to determine feature dimension.
            probe = ds_train[0]["graph"].x
            probe = probe.squeeze(1)
            assert probe.ndim == 2, f"{probe.shape}"

            nfeat = probe.shape[1]
            local_max = torch.full((nfeat,), -math.inf, dtype=probe.dtype, device=probe.device)
            local_min = torch.full((nfeat,), math.inf, dtype=probe.dtype, device=probe.device)

            # Parallel over graph samples by global rank.
            for igraph in range(rank, num_graphs, world_size):
                sample = ds_train[igraph]
                data = sample["graph"]

                x = data.x
                x = x.squeeze(1)
                assert x.ndim == 2, f"{x.shape}"

                batch_max, _ = x.max(dim=0)
                batch_min, _ = x.min(dim=0)

                local_max = torch.maximum(local_max, batch_max)
                local_min = torch.minimum(local_min, batch_min)

                print(f"[rank {rank}] {dataset} train sample {igraph}/{num_graphs}, x={tuple(x.shape)}", flush=True)

            global_min_np, global_max_np = mpi_allreduce_minmax(
                local_min=local_min,
                local_max=local_max,
                comm=comm,
            )

            if rank == 0:
                min_values_list = global_min_np.tolist()
                max_values_list = global_max_np.tolist()

                print("Global feature_min:", min_values_list, flush=True)
                print("Global feature_max:", max_values_list, flush=True)

                os.makedirs(stats_output_dir, exist_ok=True)
                output_path = os.path.join(stats_output_dir, "minmax_node_features.json")

                with open(output_path, "w") as f:
                    json.dump({"feature_max": max_values_list, "feature_min": min_values_list}, f, indent=2)

                print(f"Saved stats to {output_path}", flush=True)
            
            comm.Barrier()
            if MPI.Is_initialized() and not MPI.Is_finalized():
                MPI.Finalize()
