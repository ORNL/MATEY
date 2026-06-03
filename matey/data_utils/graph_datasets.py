# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import json
import os
from dataclasses import dataclass
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
import re
import torch.distributed as dist
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
        group_id=0, group_rank=0, group_size=1, use_dist=False, partition_method  = "metis"):

        super().__init__()
        np.random.seed(2024)

        self.path = path
        self.processed_dir = self.path+f"/{split}/processed"
        self.processed_index = self.processed_dir + "/index.json"
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
        if self.use_dist:
            assert dist.is_available() and dist.is_initialized(), (
                "torch.distributed must be initialized when use_dist=True "
                "or group_size > 1"
            )

        self.partition_method = partition_method
        self.partition_root = self.path+f"/{split}/partitioned_{self.group_size}"
        os.makedirs(self.processed_dir, exist_ok=True)

        self.field_names_out, self.type, self.time_steps, self.num_node_types = self._specifics()
        self.title = self.type

        if not os.path.exists(self.processed_index) or (self.group_size>1 and not os.path.exists(self.partition_root)):
            self.process()
        
        with open(self.processed_index, "r") as f:
            self._processed_index = json.load(f)

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

    def mesh_edge_attr(self, pos, edge_index):
        #edge features: [dx, dy ,dz, |d|]
        src, dst = edge_index
        disp = pos[dst] - pos[src] #[E, dim]
        dist = torch.linalg.norm(disp, dim=-1, keepdim=True) #[E,1]
        return torch.cat([disp, dist], dim=-1)

    def one_hot_node_type(self, node_type, num_types):
        node_type = node_type.view(-1).long()
        return F.one_hot(node_type, num_classes=num_types).to(torch.float32)

    def create_splits(self, samples):
        
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
    def partition_dataset(src_processed_dir, dst_partition_root, num_parts, method = "metis", overwrite = False, rank=0, world_size=1):
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
        #print(f"[partition_dataset] {len(traj_dirs)} trajectories, {num_parts} parts, method={method}")

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
                edge_attr = self.mesh_edge_attr(pos, edge_index)

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
                    "splits": "ALL",
                }
                with open(self.processed_index, "w") as f:
                    json.dump(index_obj, f, indent=2)
            dist.barrier()
        else:
            index_obj = {
                "version": 1,
                "num_samples": len(samples_local),
                "samples": [s.__dict__ for s in samples_local],
                "splits": "ALL", #since it was pre-split already
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

        data = Data(x=x_seq, y=y_state, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
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

if __name__ == "__main__":
    from matey.utils import setup_dist
    device, world_size, local_rank, global_rank = setup_dist()
    rank = global_rank

    splitgraph=False #True #
    if splitgraph:
        #ds_train = MeshGraphNetsAirfoilDataset(path="/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil",include_string='train',split="train",dt=1, 
        #                                    group_id=0, group_rank=rank, group_size=world_size)
        #MeshGraphNetsAirfoilDataset.partition_dataset(
        #    src_processed_dir  = "/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil/train/processed/",
        #    dst_partition_root = "/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil/train/partitioned_4/",
        #    num_parts          = 4,
        #    method             = "random",
        #    overwrite          = False, #True, #
        #    rank=rank, 
        #    world_size=world_size
        #)
        #ds_val = MeshGraphNetsAirfoilDataset(path="/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil",include_string='val',split="val",dt=1, 
        #                                    group_id=0, group_rank=rank, group_size=world_size)
        MeshGraphNetsAirfoilDataset.partition_dataset(
            src_processed_dir  = "/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil/val/processed/",
            dst_partition_root = "/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil/val/partitioned_4/",
            num_parts          = 4,
            method             = "random",
            overwrite          = True, #
            rank=rank, 
            world_size=world_size
        )
    else:
        ds_train = MeshGraphNetsAirfoilDataset(path="/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil",include_string='train',split="train",dt=1, 
                                            group_id=0, group_rank=0, group_size=1, use_dist=True)
        print("Airfoil train samples:", len(ds_train), "example:", ds_train[0])
        #"""
        ds_valid = MeshGraphNetsAirfoilDataset(path="/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil",include_string='val',split="val",dt=1, 
                                            group_id=0, group_rank=0, group_size=1, use_dist=True)
        print("Airfoil valid samples:", len(ds_valid), "example:", ds_valid[0])
        """
        ds_test = MeshGraphNetsAirfoilDataset(path="/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil",include_string='test',split="test",dt=1, 
                                            group_id=0, group_rank=rank, group_size=world_size, use_dist=True)
        print("Airfoil valid samples:", len(ds_test), "example:", ds_test[0])
        """

        num_graphs = len(ds_train)
        local_max = None
        local_min = None

        # Parallel over graphs: each rank gets a strided subset
        for igraph in range(rank, num_graphs, world_size):
            print(f"Airfoil train sample {ds_train[igraph]}, {igraph}/{len(ds_train)}/{world_size}")
            data = ds_train[igraph]["graph"]
            x = data.x  
            x = x.squeeze(1)
            assert x.ndim==2, f"{x.shape}"
            batch_max, _ = x.max(dim=0)  #[num_features]
            batch_min, _ = x.min(dim=0)
            if local_max is None:
                local_max = batch_max
                local_min = batch_min
            else:
                local_max = torch.maximum(local_max, batch_max)
                local_min = torch.minimum(local_min, batch_min)


        global_max = local_max.clone()
        global_min = local_min.clone()

        # parallel reduction: take max/min across all ranks
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(global_min, op=dist.ReduceOp.MIN)

        global_max_np = global_max.cpu().numpy()
        global_min_np = global_min.cpu().numpy()

        if rank ==0:
            max_values_list = global_max_np.tolist()
            min_values_list = global_min_np.tolist()

            print("Global feature_min:", min_values_list, flush=True)
            print("Global feature_max:", max_values_list, flush=True)

            output_dir = "/lustre/orion/lrn037/proj-shared/deepmindmeshgraph/airfoil/train/stats"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"minmax_node_features.json")
            with open(output_path, "w") as f:
                json.dump({"feature_max": max_values_list, "feature_min": min_values_list}, f, indent=2)
