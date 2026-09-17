# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from operator import mul
from functools import reduce
from einops import rearrange, repeat
from ..utils.distributed_utils import closest_factors
from torch_geometric.nn import GCNConv, GraphNorm, avg_pool
from torch_geometric.data import Data
from collections import deque
from ..data_utils import HaloExchange_sync
import torch.distributed as dist

### Space utils
#FIXME: this function causes training instability. Keeping it now for reproducibility; We'll remove it
class RMSInstanceNormSpace(nn.Module):
    def __init__(self, dim, affine=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            #self.bias = nn.Parameter(torch.zeros(dim)) # Forgot to remove this so its in the pretrained weights

    def forward(self, x):
        #x: [TB, C, D, H, W]
        spatial_dims = tuple(range(x.ndim))[2:]
        std, mean = torch.std_mean(x, dim=spatial_dims, keepdim=True)
        x = (x) / (std + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None, None]
        return x

    
class PatchExpandinSpace(nn.Module):
    def __init__(self, dim, expand_ratio=2):
        super().__init__()
        self.proj2D = nn.Linear(dim, dim * (expand_ratio**2))
        self.proj3D = nn.Linear(dim, dim * (expand_ratio**3))
        self.expand_ratio = expand_ratio
    
    def forward(self, x, token_dim=[1, 1, 1],space_dim=3):
        #t b c seq_len
        #token_dim: target token_dim
        token_dim_inp=[token_dim[i]//self.expand_ratio for i in range(3)]

        T, B, C, seq_len = x.shape
        assert reduce(mul, token_dim_inp) == seq_len, f"checking dimensions, {token_dim_inp}, {seq_len}"

        x = rearrange(x, 't b c seqlen -> t b seqlen c')
        if space_dim==3:
            x = self.proj3D(x)   
            x = rearrange(x, 't b (d h w) cexp -> t b d h w cexp', d=token_dim_inp[0], h=token_dim_inp[1], w=token_dim_inp[2])
            x = rearrange(x, 't b d h w (c exprtd exprth exprtw) -> t b c (d exprtd) (h exprth) (w exprtw)', exprtd=self.expand_ratio, exprth=self.expand_ratio, exprtw=self.expand_ratio)
            x = rearrange(x, 't b c dexp hexp wexp -> t b c (dexp hexp wexp)')
        else:
            x = self.proj2D(x)
            x = rearrange(x, 't b (d h w) cexp -> t b d h w cexp', d=token_dim_inp[0], h=token_dim_inp[1], w=token_dim_inp[2])
            x = rearrange(x, 't b d h w (c exprth exprtw) -> t b c d (h exprth) (w exprtw)', exprth=self.expand_ratio, exprtw=self.expand_ratio)
            x = rearrange(x, 't b c d hexp wexp -> t b c (d hexp wexp)')
        
        """
        x = self.proj3D(x)  if space_dim==3 else self.proj2D(x)
        x = rearrange(x, 't b seqlen (c lenexp_ratio) -> t b c (lenexp_ratio seqlen)', lenexp_ratio=self.expand_ratio**space_dim)
        """
        #t,b,c,seq_len*self.expand_ratio**space_dim
        assert (T, B, C, seq_len*self.expand_ratio**space_dim) == x.shape
        return x
    
class PatchUpsampleinSpace(nn.Module):
    def __init__(self, in_channels, expand_ratio=2):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.nlevel=expand_ratio.bit_length() - 1
        modulelist2D = []
        modulelist3D = []
        for _ in range(self.nlevel):

            modulelist2D.append(nn.Upsample(scale_factor=2, mode='nearest'))
            modulelist2D.append(nn.Conv3d(in_channels, in_channels, kernel_size=(1, 2, 2), stride=1, 
                                    padding="same", padding_mode="reflect"))
            modulelist2D.append(nn.GELU())
            modulelist2D.append(nn.Conv3d(in_channels, in_channels, kernel_size=(1, 2, 2), stride=1, 
                                    padding="same", padding_mode="reflect"))

            modulelist3D.append(nn.Upsample(scale_factor=2, mode='nearest'))
            modulelist3D.append(nn.Conv3d(in_channels, in_channels, kernel_size=(2, 2, 2), stride=1, 
                                padding="same", padding_mode="reflect"))
            modulelist3D.append(nn.GELU())
            modulelist3D.append(nn.Conv3d(in_channels, in_channels, kernel_size=(2, 2, 2), stride=1, 
                                padding="same", padding_mode="reflect"))
        self.upsample2d = torch.nn.Sequential(*modulelist2D)
        self.upsample3d = torch.nn.Sequential(*modulelist3D)

    def forward(self, x, token_dim=[1, 1, 1],space_dim=3):
        #t b c seq_len
        #token_dim: target token_dim
        token_dim_inp=[token_dim[i]//self.expand_ratio for i in range(3)]

        T, B, C, seq_len = x.shape
        assert reduce(mul, token_dim_inp) == seq_len, f"checking dimensions, {token_dim_inp}, {seq_len}"

        x = rearrange(x, 't b c seqlen -> (t b) c seqlen')
        x = rearrange(x, 'tb c (d h w) -> tb c d h w', d=token_dim_inp[0], h=token_dim_inp[1], w=token_dim_inp[2])
        #x = self.upsample(x) #tb,c,dexp,hexp,wexp
        if space_dim==3:
            x = self.upsample3d(x)
        else:
            x = self.upsample2d(x)
            
        x = rearrange(x, '(t b) c dexp hexp wexp -> t b c (dexp hexp wexp)', t=T)
        
        #t,b,c,seq_len*self.expand_ratio**space_dim
        assert (T, B, C, seq_len*self.expand_ratio**space_dim) == x.shape
        return x
    
class UpsampleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1,1,1], bias=True):
        super().__init__()
        
        self.upsample = torch.nn.Sequential(
            nn.Upsample(scale_factor=kernel_size, mode='trilinear'),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias, padding_mode="reflect")
            )
        
    def forward(self, x):
        #B,C,D,H,W
        x = self.upsample(x)
        return x


class UpsampleinSpace(nn.Module):
    """ upsample solution fields
    """
    def __init__(self, patch_size=(1,16,16), channels=3, nconv=3, notransposed=False):
        #patch_size: (ps_z, ps_x, ps_y)
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.nconv = nconv
        self.ks = calc_ks4conv(patch_size=self.patch_size, nconv=self.nconv)
        self.notransposed = notransposed
    
        modulelist = []
        for ilayer in range(self.nconv-1):
            ks_ilayer = self.ks[-(ilayer+1)]
            modulelist.append(UpsampleConv3d(channels, channels, kernel_size=ks_ilayer, bias=False))
            modulelist.append(nn.InstanceNorm3d(channels, affine=True))
            modulelist.append(nn.GELU())
        modulelist.append(UpsampleConv3d(channels, channels, kernel_size=self.ks[0]))
        self.out_proj = torch.nn.Sequential(*modulelist)
        
    def forward(self, x):
        #B,C,D,H,W
        x = self.out_proj(x)
           
        return x
    
class SubsampledLinear(nn.Module):
    """
    Cross between a linear layer and EmbeddingBag - takes in input
    and list of indices denoting which state variables from the state
    vocab are present and only performs the linear layer on rows/cols relevant
    to those state variables

    Assumes (... C) input
    """
    def __init__(self, dim_in, dim_out, subsample_in=True):
        super().__init__()
        self.subsample_in = subsample_in
        self.dim_in = dim_in
        self.dim_out = dim_out
        temp_linear = nn.Linear(dim_in, dim_out)
        self.weight = nn.Parameter(temp_linear.weight)
        self.bias = nn.Parameter(temp_linear.bias)

    def forward(self, x, labels):
        # Note - really only works if all batches are the same input type
        labels = labels[0] # Figure out how to handle this for normal batches later
        label_size = len(labels)
        if self.subsample_in:
            assert max(labels)<self.dim_in, f"SubsampledLinear dim_in {self.dim_in} too small for max(labels):{max(labels)}, check n_states in config"
            scale = (self.dim_in / label_size)**.5 # Equivalent to swapping init to correct for given subsample of input
            x = scale * F.linear(x, self.weight[:, labels], self.bias)
            
        else:
            x = F.linear(x, self.weight[labels], self.bias[labels])
        return x

def calc_ks4conv(patch_size=(1,16,16), nconv=3):

    pz = closest_factors(patch_size[0], nconv)
    px = closest_factors(patch_size[1], nconv)
    py = closest_factors(patch_size[2], nconv) 
    #increasing

    ks = []
    for i in range(nconv):
        ks.append((pz[i], px[i], py[i]))

    assert reduce(mul, [ks[i][0] for i in range(len(ks))]) == patch_size[0]
    assert reduce(mul, [ks[i][1] for i in range(len(ks))]) == patch_size[1]
    assert reduce(mul, [ks[i][2] for i in range(len(ks))]) == patch_size[2]

    return ks

class hMLP_stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=(1,16,16), in_chans=3, embed_dim=768, nconv=3, use_linear=False):
        #patch_size: (ps_z, ps_x, ps_y)
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.nconv = nconv
        self.use_linear = use_linear
        self.ks = calc_ks4conv(patch_size=self.patch_size, nconv=self.nconv)

        if self.use_linear:
            self.linears = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.acts = nn.ModuleList()
            for ilayer in range(self.nconv):
                in_chans_ilayer = in_chans if ilayer==0 else embed_dim//4
                embed_ilayer = embed_dim if ilayer==self.nconv-1 else embed_dim//4
                kD, kH, kW = self.ks[ilayer]
                self.linears.append(nn.Linear(in_chans_ilayer * kD * kH * kW, embed_ilayer, bias=False))
                self.norms.append(nn.InstanceNorm3d(embed_ilayer, affine=True))
                self.acts.append(nn.GELU())
        else:
            modulelist = []
            for ilayer in range(self.nconv):
                in_chans_ilayer = in_chans if ilayer==0 else embed_dim//4
                embed_ilayer = embed_dim if ilayer==self.nconv-1 else embed_dim//4
                ks_ilayer = self.ks[ilayer]
                modulelist.append(nn.Conv3d(in_chans_ilayer, embed_ilayer, kernel_size=ks_ilayer, stride=ks_ilayer, bias=False))
                modulelist.append(nn.InstanceNorm3d(embed_ilayer, affine=True))
                modulelist.append(nn.GELU())
            self.in_proj = torch.nn.Sequential(*modulelist)

    def forward(self, x):
        if self.use_linear:
            for ilayer in range(self.nconv):
                TB = x.shape[0]
                kD, kH, kW = self.ks[ilayer]
                D, H, W = x.shape[2], x.shape[3], x.shape[4]
                x = rearrange(x, 'tb cin (nd kd) (nh kh) (nw kw) -> (tb nd nh nw) (cin kd kh kw)',
                              kd=kD, kh=kH, kw=kW)
                x = self.linears[ilayer](x)
                x = rearrange(x, '(tb nd nh nw) cout -> tb cout nd nh nw',
                              tb=TB, nd=D//kD, nh=H//kH, nw=W//kW)
                x = self.norms[ilayer](x)
                x = self.acts[ilayer](x)
            return x
        else:
            x = self.in_proj(x)
            return x

class hMLP_output(nn.Module):
    """ Patch to Image De-bedding
    """
    def __init__(self, patch_size=(1,16,16), out_chans=3, embed_dim=768, nconv=3, notransposed=False, smooth=False, use_linear=False):
        #patch_size: (ps_z, ps_x, ps_y)
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.nconv = nconv
        self.ks = calc_ks4conv(patch_size=self.patch_size, nconv=self.nconv)
        self.notransposed = notransposed
        self.smooth = smooth
        self.use_linear = use_linear and not notransposed  # linear only applies to ConvTranspose3d path

        if self.use_linear:
            self.linears = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.acts = nn.ModuleList()
            for ilayer in range(self.nconv-1):
                in_chans_ilayer = embed_dim if ilayer==0 else embed_dim//4
                embed_ilayer = embed_dim//4
                kD, kH, kW = self.ks[-(ilayer+1)]
                self.linears.append(nn.Linear(in_chans_ilayer, embed_ilayer * kD * kH * kW, bias=False))
                self.norms.append(nn.InstanceNorm3d(embed_ilayer, affine=True))
                self.acts.append(nn.GELU())
            # Final head
            kD, kH, kW = self.ks[0]
            self.out_head = nn.Linear(embed_dim//4, out_chans * kD * kH * kW)
            self.out_head_ks = self.ks[0]
            if self.smooth:
                self.smooth = nn.Conv3d(out_chans, out_chans, kernel_size=self.ks[0], stride=1, groups=out_chans, padding="same", padding_mode="reflect")
        else:
            modulelist = []
            for ilayer in range(self.nconv-1):
                in_chans_ilayer = embed_dim if ilayer==0 else embed_dim//4
                embed_ilayer = embed_dim//4
                ks_ilayer = self.ks[-(ilayer+1)]
                if self.notransposed:
                    modulelist.append(UpsampleConv3d(in_chans_ilayer, embed_ilayer, kernel_size=ks_ilayer, bias=False))
                else:
                    modulelist.append(nn.ConvTranspose3d(in_chans_ilayer, embed_ilayer, kernel_size=ks_ilayer, stride=ks_ilayer, bias=False))
                modulelist.append(nn.InstanceNorm3d(embed_ilayer, affine=True))
                modulelist.append(nn.GELU())
            self.out_proj = torch.nn.Sequential(*modulelist)
            if self.notransposed:
                out_head = UpsampleConv3d(embed_dim//4, out_chans, kernel_size=self.ks[0])
                self.out_head = out_head
            else:
                self.out_head = nn.ConvTranspose3d(embed_dim//4, out_chans, kernel_size=self.ks[0], stride=self.ks[0])
            if self.smooth:
                self.smooth = nn.Conv3d(out_chans, out_chans, kernel_size=self.ks[0], stride=1, groups=out_chans, padding="same", padding_mode="reflect")

    def forward(self, x):
        #B,C,D,H,W
        if self.use_linear:
            for ilayer in range(self.nconv-1):
                TB, _, D, H, W = x.shape
                kD, kH, kW = self.ks[-(ilayer+1)]
                cout = self.linears[ilayer].out_features // (kD * kH * kW)
                x = rearrange(x, 'tb cin d h w -> (tb d h w) cin')
                x = self.linears[ilayer](x)
                x = rearrange(x, '(tb d h w) (cout kd kh kw) -> tb cout (d kd) (h kh) (w kw)',
                              tb=TB, d=D, h=H, w=W, cout=cout, kd=kD, kh=kH, kw=kW)
                x = self.norms[ilayer](x)
                x = self.acts[ilayer](x)
            # Final head
            TB, _, D, H, W = x.shape
            kD, kH, kW = self.out_head_ks
            x = rearrange(x, 'tb cin d h w -> (tb d h w) cin')
            x = self.out_head(x)
            x = rearrange(x, '(tb d h w) (cout kd kh kw) -> tb cout (d kd) (h kh) (w kw)',
                          tb=TB, d=D, h=H, w=W, cout=self.out_chans, kd=kD, kh=kH, kw=kW)
            if self.smooth:
                x = self.smooth(x)
            return x
        else:
            x = self.out_proj(x)
            x = self.out_head(x)
            if self.smooth:
                x = self.smooth(x)
            return x

class GraphhMLP_stem(nn.Module):
    """graph to patch embedding"""
    def __init__(self, patch_size=[1,1,1], in_chans=3, embed_dim=768, nconv=3, ghost_sync=False):
        super().__init__()
        #assert patch_size==[1, 1 ,1], f"graph input heads only support patch size of 1 for now, but get {patch_size}"
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.nconv = nconv
        self.do_pool = self.patch_size != [1, 1, 1]
        self.ghost_sync = ghost_sync

        #assert not (self.do_pool and self.ghost_sync), f"currently not supporting both self.do_pool and self.ghost_sync {self.do_pool, self.ghost_sync}"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.GELU()

        for ilayer in range(nconv):
            in_chans_ilayer = in_chans if ilayer==0 else embed_dim//4
            embed_ilayer = embed_dim if ilayer==self.nconv-1 else embed_dim//4
            self.convs.append(GCNConv(in_chans_ilayer, embed_ilayer))
            self.norms.append(GraphNorm(embed_ilayer))

    def make_hop_cluster(self, edge_index, num_nodes, seeds):
        """
        Assign each node to the nearest seed in hop distance.
            edge_index: [2, E]
            num_nodes: int
            seeds: 1D LongTensor of seed node indices
        returns:
            seedid: [num_nodes] with values in [0, num_seeds-1]
        """
        adj = [[] for _ in range(num_nodes)]
        row, col = edge_index
        for u, v in zip(row.tolist(), col.tolist()):
            adj[u].append(v)
            adj[v].append(u)

        dist = [-1] * num_nodes #distance from the nearest seed
        seedid = [-1] * num_nodes #seed ID that claims the node
        q = deque()
        #starting with a list of nodes in seeds
        for cid, s in enumerate(seeds): 
            dist[s] = 0
            seedid[s] = cid
            q.append(s)

        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    seedid[v] = seedid[u]
                    q.append(v)
        
        #leftover nodes to seed ID 0 
        for i in range(num_nodes):
            if seedid[i] == -1:
                raise ValueError(f"Unexpected isolated nodes {i} {seedid}")
                #seedid[i] = 0

        return torch.tensor(seedid, dtype=torch.long, device=edge_index.device)
     
    def _make_hop_cluster(self, edge_index, num_nodes):
        patch_vol = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        target_num_clusters = max(1, num_nodes // patch_vol)

        if target_num_clusters >= num_nodes:
            seeds = torch.arange(num_nodes, device=edge_index.device)
        else:
            idx = torch.linspace(0, num_nodes - 1, steps=target_num_clusters, device=edge_index.device)
            seeds = idx.round().long().unique()

        cluster = self.make_hop_cluster(edge_index, num_nodes, seeds)

        return cluster

    def forward(self, data):
        """
        data:  (node_features, batch, edge_index, ghost_info)
            x          : [N_local, T, C]   float32   owned + ghost nodes
            batch      : [N_local]         long      PyG batch vector
            edge_index : [2, E_local]      long      local remapped edges
            ghost_info : GhostInfo | None
        Returns
        -------
        Same tuple with x replaced by [N_local, T, embed_dim].
        ghost rows in x are updated after the last layer so downstream
        modules can call halo.sync again if needed.
        """
        x, batch0, edge_index0, ghost_info, comm = data
        N, T, C= x.shape
        x_list=[]
        if self.do_pool:
            #Note: currently only support one graph
            if batch0.numel() > 0 and batch0.max().item() != 0:
                raise NotImplementedError("Hop clustering currently supports one graph per input only")
            cluster = self._make_hop_cluster(edge_index0, N)
        else:
            cluster = torch.arange(N, device=x.device)
        for it in range(T):
            h = x[:,it,:]
            batch = batch0
            edge_index = edge_index0
            for ilayer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                #sync ghost embeddings before aggregation
                if ghost_info is not None and self.ghost_sync:
                    if ilayer<2: #after ilayer==1, nodes have been aggregatered
                        h = HaloExchange_sync(h, ghost_info, comm)
                    elif not self.do_pool:
                        h = HaloExchange_sync(h, ghost_info, comm)
                if ilayer==1 and self.do_pool:
                    pyg_data = Data(x=h, edge_index=edge_index, batch=batch)
                    pooled = avg_pool(cluster, pyg_data)
                    h = pooled.x
                    batch = pooled.batch
                    edge_index = pooled.edge_index

                h_in = h
                h = conv(h, edge_index)
                h = norm(h, batch)
                h = self.act(h)
                if h.shape == h_in.shape:
                    h = h + h_in

                assert edge_index.min().item() >= 0
                assert edge_index.max().item() < h.size(0), (
                    f"Bad edge_index at layer {ilayer}: "
                    f"max={edge_index.max().item()}, num_nodes={h.size(0)}"
                )
            if ghost_info is not None and self.ghost_sync and not self.do_pool:
                h = HaloExchange_sync(h, ghost_info, comm)
            x_list.append(h)
        x_out = torch.stack(x_list, dim=1)
        unpool_info = {
            "cluster": cluster,
            "orig_batch": batch0,
            "orig_edge_index": edge_index0,
        }
        assert edge_index.min().item() >= 0
        assert edge_index.max().item() < x_out.size(0), (
            f"Bad edge_index at layer {ilayer}: "
            f"max={edge_index.max().item()}, num_nodes={h.size(0)}"
        )
        return (x_out, batch, edge_index, unpool_info, ghost_info, comm)
    
class GraphhMLP_output(nn.Module):
    def __init__(self, patch_size=[1,1,1], out_chans=3, embed_dim=768, nconv=3, smooth=False, ghost_sync=False):
        super().__init__()
        #assert patch_size==[1, 1 ,1], f"graph output heads only support patch size of 1 for now, but get {patch_size}"
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.nconv = nconv
        self.smooth_flag = smooth
        self.ghost_sync = ghost_sync
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.GELU()
        for ilayer in range(nconv - 1):
            in_chans_ilayer = embed_dim if ilayer==0 else embed_dim//4
            embed_ilayer = embed_dim//4
            self.convs.append(GCNConv(in_chans_ilayer, embed_ilayer))
            self.norms.append(GraphNorm(embed_ilayer))

        in_head = embed_dim if nconv == 1 else embed_dim//4
        self.out_head = nn.Sequential(
                        nn.Linear(in_head, in_head),
                        nn.GELU(),
                        nn.Linear(in_head, out_chans)
                    )
        if self.smooth_flag:
            self.smooth = GCNConv(out_chans, out_chans)
        else:
            self.smooth = None

    def forward(self, data, return_allT=False):
        """
        data:  (node_features, batch, edge_index, unpool_info, ghost_info)
            x          : [N_local, T, embed_dim]
            batch      : [N_local] long
            edge_index : [2, E_local] long
            ghost_info : GhostInfo | None

        Returns
        -------
        Same tuple with x replaced by [N_local, T, out_chans].
        Only owned rows carry meaningful predictions; ghost rows are
        byproducts of the final conv and should be masked away.
        """
        x, batch, edge_index, unpool_info, ghost_info, comm = data
        cluster = unpool_info["cluster"] #[N_orig]
        batch = unpool_info["orig_batch"] #[N_orig]
        edge_index = unpool_info["orig_edge_index"] #[2, E_orig]

        N_patch, T, C = x.shape
        N_orig = cluster.numel()

        out_chans = self.out_head[-1].out_features
        if return_allT:
            x_out = x.new_empty((N_orig, T, out_chans))
            tsteps=range(T)
        else:
            x_out = x.new_empty((N_orig, 1, out_chans))
            tsteps=[-1]

        for it in tsteps:
            h = x[:,it,:]  #[N_patch, C]
            h = h[cluster] #[N_orig, C]
            for conv, norm in zip(self.convs, self.norms):
            #for conv in self.convs:
             #sync before aggregation
                if ghost_info is not None and self.ghost_sync:
                    h = HaloExchange_sync(h, ghost_info, comm)
                h_in = h
                h = conv(h, edge_index)
                h = norm(h, batch)
                h = self.act(h)
                if h.shape == h_in.shape:
                    h = h + h_in
            h = self.out_head(h)
            if self.smooth is not None:
                if ghost_info is not None and self.ghost_sync:
                    h = HaloExchange_sync(h, ghost_info, comm)
                h = self.smooth(h, edge_index)
            if ghost_info is not None and self.ghost_sync:
                h = HaloExchange_sync(h, ghost_info, comm)
            x_out[:, it, :] = h 
        #[N_local, T, out_chans]
        return (x_out, batch, edge_index, ghost_info, comm)
