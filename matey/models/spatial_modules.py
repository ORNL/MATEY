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
from torch_geometric.nn import GCNConv, GraphNorm
from typing import List, Literal, Optional, Callable
import time
try:
    from neuralop.layers.gno_block import GNOBlock
    neuralop_exist = True
except ImportError:
    neuralop_exist = False
try:
    import sklearn
    sklearn_exist = True
except ImportError:
    sklearn_exist = False

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
    def __init__(self, patch_size=(1,16,16), in_chans=3, embed_dim=768, nconv=3):
        #patch_size: (ps_z, ps_x, ps_y)
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.nconv = nconv
        self.ks = calc_ks4conv(patch_size=self.patch_size, nconv=self.nconv)

        modulelist = []
        for ilayer in range(self.nconv):
            in_chans_ilayer = in_chans if ilayer==0 else embed_dim//4
            embed_ilayer = embed_dim if ilayer==self.nconv-1 else embed_dim//4
            ks_ilayer = self.ks[ilayer]
            #modulelist.append(nn.Conv2d(in_chans_ilayer, embed_ilayer, kernel_size=ks_ilayer, stride=ks_ilayer, bias=False))
            #modulelist.append(RMSInstanceNorm2d(embed_ilayer, affine=True))    #changed to RMSInstanceNormSpace
            modulelist.append(nn.Conv3d(in_chans_ilayer, embed_ilayer, kernel_size=ks_ilayer, stride=ks_ilayer, bias=False))
            modulelist.append(nn.InstanceNorm3d(embed_ilayer, affine=True))
            modulelist.append(nn.GELU())
        self.in_proj = torch.nn.Sequential(*modulelist)

    def forward(self, x):
        x = self.in_proj(x)
        return x

class hMLP_output(nn.Module):
    """ Patch to Image De-bedding
    """
    def __init__(self, patch_size=(1,16,16), out_chans=3, embed_dim=768, nconv=3, notransposed=False, smooth=False):
        #patch_size: (ps_z, ps_x, ps_y)
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.nconv = nconv
        self.ks = calc_ks4conv(patch_size=self.patch_size, nconv=self.nconv)
        self.notransposed = notransposed
        self.smooth = smooth
    
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
            """
            #previous implementation
            out_head = nn.ConvTranspose3d(embed_dim//4, out_chans, kernel_size=self.ks[0], stride=self.ks[0])
            self.out_stride = self.ks[0]
            self.out_kernel = nn.Parameter(out_head.weight)
            self.out_bias = nn.Parameter(out_head.bias)
            """

    def forward(self, x):
        #B,C,D,H,W
        x = self.out_proj(x)#.flatten(2).transpose(1, 2)
        if self.notransposed:
            #x = self.out_upsample(x)
            x = self.out_head(x)
            #x = F.conv3d(x, self.out_kernel[state_labels, :], self.out_bias[state_labels], stride=self.out_stride)
            #x = x[:,state_labels,...]
        else:
            """
            x = F.conv_transpose3d(x, self.out_kernel[:, state_labels], self.out_bias[state_labels], stride=self.out_stride)
            """
            x = self.out_head(x)
            if self.smooth:
                x = self.smooth(x)
            #x = x[:,state_labels,...]
        return x

class GraphhMLP_stem(nn.Module):
    """graph to patch embedding"""
    def __init__(self, patch_size=(1,1,1), in_chans=3, embed_dim=768, nconv=3):
        super().__init__()
        assert patch_size==[1, 1 ,1], f"graph input heads only support patch size of 1 for now, but get {patch_size}"
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.nconv = nconv

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.GELU()

        for ilayer in range(nconv):
            in_chans_ilayer = in_chans if ilayer==0 else embed_dim//4
            embed_ilayer = embed_dim if ilayer==self.nconv-1 else embed_dim//4
            self.convs.append(GCNConv(in_chans_ilayer, embed_ilayer))
            self.norms.append(GraphNorm(embed_ilayer))

    def forward(self, data):
        """
        data:  (node_features, batch, edge_index)
        """
        x, batch, edge_index = data
        N, T, C= x.shape
        x_list=[]
        for it in range(T):
            h = x[:,it,:]
            for conv, norm in zip(self.convs, self.norms):
                h_in = h
                h = conv(h, edge_index)
                h = norm(h, batch)
                h = self.act(h)
                if h.shape == h_in.shape:
                    h = h + h_in
            x_list.append(h)
        x_out = torch.stack(x_list, dim=1)
        return (x_out, batch, edge_index)
    
class GraphhMLP_output(nn.Module):
    def __init__(self, patch_size=(1,1,1), out_chans=3, embed_dim=768, nconv=3, smooth=False):
        super().__init__()
        assert patch_size==[1, 1 ,1], f"graph output heads only support patch size of 1 for now, but get {patch_size}"
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.nconv = nconv
        self.smooth_flag = smooth
        
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

    def forward(self, data):
        """
        data:  (node_features, batch, edge_index)
        """
        x, batch, edge_index = data
        N, T, C= x.shape
        x_list=[]
        for it in range(T):
            h = x[:,it,:]
            for conv, norm in zip(self.convs, self.norms):
            #for conv in self.convs:
                h_in = h
                h = conv(h, edge_index)
                h = norm(h, batch)
                h = self.act(h)
                if h.shape == h_in.shape:
                    h = h + h_in
            h = self.out_head(h)
            if self.smooth is not None:
                h = self.smooth(h, edge_index)
            x_list.append(h)
        x_out = torch.stack(x_list, dim=1)
        return (x_out, batch, edge_index)


class CustomNeighborSearch(nn.Module):
    def __init__(self, return_norm=False):
        super().__init__()
        self.search_fn = custom_neighbor_search
        self.return_norm = return_norm

    def forward(self, data, queries, radius):
        return_dict = self.search_fn(data, queries, radius, self.return_norm)
        return return_dict

def custom_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float, return_norm: bool=False):
    if not sklearn_exist:
        raise RuntimeError("sklearn is required for constructing neighbors.")

    kdtree = sklearn.neighbors.KDTree(data.cpu(), leaf_size=2)

    if return_norm:
        indices, dists = kdtree.query_radius(queries.cpu(), r=radius, return_distance=True)
        weights = torch.from_numpy(np.concatenate(dists)).to(queries.device)
    else:
        indices = kdtree.query_radius(queries.cpu(), r=radius)

    sizes = np.array([arr.size for arr in indices])
    nbr_indices = torch.from_numpy(np.concatenate(indices)).to(queries.device)
    nbrhd_sizes = torch.cumsum(torch.from_numpy(sizes).to(queries.device), dim=0)
    splits = torch.cat((torch.tensor([0.]).to(queries.device), nbrhd_sizes))

    nbr_dict = {}
    nbr_dict['neighbors_index'] = nbr_indices.long().to(queries.device)
    nbr_dict['neighbors_row_splits'] = splits.long()
    if return_norm:
        nbr_dict['weights'] = weights**2

    return nbr_dict


class ModifiedGNOBlock(GNOBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 coord_dim: int,
                 radius: float,
                 transform_type="linear",
                 weighting_fn: Optional[Callable]=None,
                 reduction: Literal['sum', 'mean']='sum',
                 pos_embedding_type: str='transformer',
                 pos_embedding_channels: int=32,
                 pos_embedding_max_positions: int=10000,
                 channel_mlp_layers: List[int]=[128,256,128],
                 channel_mlp_non_linearity=F.gelu,
                 channel_mlp: nn.Module=None,
                 use_torch_scatter_reduce: bool=True):
        super().__init__(in_channels, out_channels, coord_dim, radius,
                         transform_type, weighting_fn, reduction,
                         pos_embedding_type, pos_embedding_channels,
                         pos_embedding_max_positions, channel_mlp_layers,
                         channel_mlp_non_linearity, channel_mlp,
                         use_torch_scatter_reduce)

        self.neighbor_search = CustomNeighborSearch(return_norm=weighting_fn is not None)

        self.neighbors_dict = {}

    def forward(self, y, x, f_y, key):
        if f_y is not None:
            if f_y.ndim == 3 and f_y.shape[0] == -1:
                f_y = f_y.squeeze(0)

        key = f'{key}:{self.radius}:{y.shape}:{x.shape}'
        if not key in self.neighbors_dict:
            #  print(f'{key}: building new neighbors')
            neigh = self.neighbor_search(data=y, queries=x, radius=self.radius)
            self.neighbors_dict[key] = neigh
        else:
            #  print(f'{key}: using cached neighbors')
            pass

        if self.pos_embedding is not None:
            y_embed = self.pos_embedding(y)
            x_embed = self.pos_embedding(x)
        else:
            y_embed = y
            x_embed = x

        out_features = self.integral_transform(y=y_embed,
                                               x=x_embed,
                                               neighbors=self.neighbors_dict[key],
                                               f_y=f_y)

        return out_features

class GNOhMLP_stem(nn.Module):
    """Geometry to patch embedding"""
    def __init__(self, params, in_chans, out_chans):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.radius = params["radius_in"]

        self.gno = ModifiedGNOBlock(
            in_channels=in_chans,
            out_channels=out_chans,
            coord_dim=3,
            radius=self.radius,
            transform_type='nonlinear_kernelonly'
        )

        # FIXME: should there be a normalization layer here

        self.res = params["resolution"] # z, x, y

        # Latent grid is [(HWD) x 3]
        tx = torch.linspace(0, 1, self.res[1], dtype=torch.float32)
        ty = torch.linspace(0, 1, self.res[2], dtype=torch.float32)
        tz = torch.linspace(0, 1, self.res[0], dtype=torch.float32)
        X, Y, Z = torch.meshgrid(tx, ty, tz, indexing="ij")
        grid = torch.stack((X, Y, Z), dim=-1)
        self.latent_grid = torch.flatten(grid, end_dim=-2)


    def forward(self, data):
        """
        data:  (x, geometry)
        """
        x, geometry = data

        T, B, C, D, H, W = x.shape
        Dlat, Hlat, Wlat = self.res[0], self.res[1], self.res[2]

        out = torch.zeros(T, B, self.out_chans, Dlat, Hlat, Wlat, device=x.device)

        x = rearrange(x, 't b c d h w -> b t (h w d) c')

        # The challenge is that different samples in the same batch may correspond to different geometries
        input_grid = [None] * B
        latent_grid = [None] * B
        out = rearrange(out, 't b c d h w -> b t (h w d) c')
        for b in range(B):
            geometry_id = geometry["geometry_id"][b]
            input_grid[b] = torch.flatten(geometry["grid_coords"][b], end_dim=-2)

            # Rescale auxiliary grid
            bmin = [None] * 3
            bmax = [None] * 3
            for d in range(3):
                bmin[d] = input_grid[b][:,d].min()
                bmax[d] = input_grid[b][:,d].max()
            latent_grid[b] = self.latent_grid.to(device=x.device)
            for d in range(3):
                latent_grid[b][:,d] = bmin[d] + (bmax[d] - bmin[d]) * latent_grid[b][:,d]

            # Use T as batch
            out[b] = self.gno(y=input_grid[b], x=latent_grid[b], f_y=x[b], key=str(geometry_id) + ":in")
        out = rearrange(out, 'b t (h w d) c -> t b c d h w', d=Dlat, h=Hlat, w=Wlat)

        return out

class GNOhMLP_output(nn.Module):
    """Patch to geometry de-bedding"""
    def __init__(self, params, in_chans, out_chans):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.radius = params["radius_out"]

        self.gno = ModifiedGNOBlock(
            in_channels=in_chans,
            out_channels=out_chans,
            coord_dim=3,
            radius=self.radius,
            transform_type='nonlinear_kernelonly'
        )

        # FIXME: should there be a normalization layer here

        self.res = params["resolution"] # z, x, y

        # Latent grid is [(HWD) x 3]
        tx = torch.linspace(0, 1, self.res[1], dtype=torch.float32)
        ty = torch.linspace(0, 1, self.res[2], dtype=torch.float32)
        tz = torch.linspace(0, 1, self.res[0], dtype=torch.float32)
        X, Y, Z = torch.meshgrid(tx, ty, tz, indexing="ij")
        grid = torch.stack((X, Y, Z), dim=-1)
        self.latent_grid = torch.flatten(grid, end_dim=-2)

    def forward(self, data, space_dims):
        """
        data:  (x, geometry)
        """
        x, geometry = data

        T, B, C, N = x.shape
        D, H, W = space_dims
        Dlat, Hlat, Wlat = self.res[0], self.res[1], self.res[2]

        input_grid = [None] * B
        latent_grid = [None] * B
        for b in range(B):
            geometry_id = geometry["geometry_id"][b]
            input_grid[b] = torch.flatten(geometry["grid_coords"][b], end_dim=-2)

            # Rescale auxiliary grid
            bmin = [None] * 3
            bmax = [None] * 3
            for d in range(3):
                bmin[d] = input_grid[b][:,d].min()
                bmax[d] = input_grid[b][:,d].max()
            latent_grid[b] = self.latent_grid.to(device=x.device)
            for d in range(3):
                latent_grid[b][:,d] = bmin[d] + (bmax[d] - bmin[d]) * latent_grid[b][:,d]

        x = rearrange(x, 't b c (d h w) -> b t (h w d) c', d=Dlat, h=Hlat, w=Wlat)

        out = torch.zeros(T, B, self.out_chans, D, H, W, device=x.device)
        out = rearrange(out, 't b c d h w -> b t (h w d) c')
        for b in range(B):
            # Use T as batch
            out[b] = self.gno(y=latent_grid[b], x=input_grid[b], f_y=x[b], key=str(geometry_id) + ":out")
        out = rearrange(out, 'b t (h w d) c -> t b c d h w', d=D, h=H, w=W)

        return out
