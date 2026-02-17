import torch
import torch.nn as nn
import numpy as np
try:
    from neuralop.layers.channel_mlp import LinearChannelMLP
    from neuralop.layers.integral_transform import IntegralTransform
    from neuralop.layers.embeddings import SinusoidalEmbedding
    from neuralop.layers.gno_block import GNOBlock
    neuralop_exist = True
except ImportError:
    neuralop_exist = False
try:
    import sklearn
    sklearn_exist = True
except ImportError:
    sklearn_exist = False
import torch.nn.functional as F
from ..utils.forward_options import ForwardOptionsBase, TrainOptionsBase
from typing import List, Literal, Optional, Callable
from einops import rearrange
import psutil

import time

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

    start = time.time()
    kdtree = sklearn.neighbors.KDTree(data.cpu(), leaf_size=2)
    construction_time = time.time() - start

    start = time.time()
    if return_norm:
        indices, dists = kdtree.query_radius(queries.cpu(), r=radius, return_distance=True)
        weights = torch.from_numpy(np.concatenate(dists)).to(queries.device)
    else:
        indices = kdtree.query_radius(queries.cpu(), r=radius)
    query_time = time.time() - start

    print(f'neighbors: construction = {construction_time}, query = {query_time}', flush=True)

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

class ModifiedGNOBlock(nn.Module):
    """
    The code is equivalent to the original GNOBlock in neuraloperator, except
    for the use of custom neighbor search
    """
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
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coord_dim = coord_dim

        self.radius = radius

        if not neuralop_exist:
            raise RuntimeError("NeuralOp is required for running GNO module.")

        # Apply sinusoidal positional embedding
        self.pos_embedding_type = pos_embedding_type
        if self.pos_embedding_type in ['nerf', 'transformer']:
            self.pos_embedding = SinusoidalEmbedding(
                in_channels=coord_dim,
                num_frequencies=pos_embedding_channels,
                embedding_type=pos_embedding_type,
                max_positions=pos_embedding_max_positions
            )
        else:
            self.pos_embedding = None

        # Create in-to-out nb search module
        self.neighbor_search = CustomNeighborSearch(return_norm=weighting_fn is not None)

        # create proper kernel input channel dim
        if self.pos_embedding is None:
            # x and y dim will be coordinate dim if no pos embedding is applied
            kernel_in_dim = self.coord_dim * 2
            kernel_in_dim_str = "dim(y) + dim(x)"
        else:
            # x and y dim will be embedding dim if pos embedding is applied
            kernel_in_dim = self.pos_embedding.out_channels * 2
            kernel_in_dim_str = "dim(y_embed) + dim(x_embed)"

        if transform_type == "nonlinear" or transform_type == "nonlinear_kernelonly":
            kernel_in_dim += self.in_channels
            kernel_in_dim_str += " + dim(f_y)"

        if channel_mlp is not None:
            assert channel_mlp.in_channels == kernel_in_dim, f"Error: expected ChannelMLP to take\
                  input with {kernel_in_dim} channels (feature channels={kernel_in_dim_str}),\
                      got {channel_mlp.in_channels}."
            assert channel_mlp.out_channels == out_channels, f"Error: expected ChannelMLP to have\
                 {out_channels=} but got {channel_mlp.in_channels=}."
            channel_mlp = channel_mlp

        elif channel_mlp_layers is not None:
            if channel_mlp_layers[0] != kernel_in_dim:
                channel_mlp_layers = [kernel_in_dim] + channel_mlp_layers
            if channel_mlp_layers[-1] != self.out_channels:
                channel_mlp_layers.append(self.out_channels)
            channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)

        # Create integral transform module
        self.integral_transform = IntegralTransform(
            channel_mlp=channel_mlp,
            transform_type=transform_type,
            use_torch_scatter=use_torch_scatter_reduce,
            weighting_fn=weighting_fn,
            reduction=reduction
        )

        self.neighbors_dict = {}

    def forward(self, y, x, f_y, key):
        if f_y is not None:
            if f_y.ndim == 3 and f_y.shape[0] == -1:
                f_y = f_y.squeeze(0)

        key = f'{key}:{self.radius}:{y.shape}:{x.shape}'
        if not key in self.neighbors_dict:
            print(f'{key}: building new neighbors')
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


def build_gno(inner_model, params):
    model = GNOModel(inner_model, params)

    return model


class GNOModel(nn.Module):
    def __init__(self, inner_model, params=None):
        super().__init__()

        num_channels = params.gno["n_channels"]
        self.radius_in = params.gno["radius_in"]
        self.radius_out = params.gno["radius_out"]

        print(params, flush=True)
        self.gno_in = ModifiedGNOBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            coord_dim=3,
            radius=self.radius_in
            #  weighting_fn=params.weighting_fn,
            #  reduction=params.reduction
        )
        self.model = inner_model
        self.gno_out = ModifiedGNOBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            coord_dim=3,
            radius=self.radius_out
            #  weighting_fn=params.gno.weighting_fn,
            #  reduction=params.gno.reduction
        )

        self.model = inner_model

        self.res = params.gno["resolution"]

        bmin = [0, 0, 0]
        bmax = [1, 1, 1]
        self.latent_geom = self.generate_geometry(bmin, bmax, self.res)

    def generate_geometry(self, bmin, bmax, res):
        tx = np.linspace(bmin[0], bmax[0], res[0], dtype=np.float32)
        ty = np.linspace(bmin[1], bmax[1], res[1], dtype=np.float32)
        tz = np.linspace(bmin[2], bmax[2], res[2], dtype=np.float32)

        geometry = torch.from_numpy(np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1))
        return torch.flatten(geometry, end_dim=-2)

    def forward(self, x, state_labels, bcs, opts: ForwardOptionsBase, train_opts: Optional[TrainOptionsBase]=None):
        if opts.geometry == None:
            # Pass-through option without using geometry
            return self.model(x, state_labels, bcs, opts, train_opts)

        T, B, C, D, H, W = x.shape
        Dlat, Hlat, Wlat = self.res[0], self.res[1], self.res[2]

        out = torch.zeros(T, B, C, Dlat, Hlat, Wlat, device=x.device)

        # Pre-process using GNO
        # The challenge is that different samples in the same batch may correspond to different geometries
        input_geom = [None] * B
        latent_geom = [None] * B
        for b in range(B):
            geometry_id = opts.geometry["geometry_id"][b]
            input_geom[b] = torch.flatten(opts.geometry["geometry"][b], end_dim=-2)

            # Rescale auxiliary grid
            bmin = [None] * 3
            bmax = [None] * 3
            for d in range(3):
                bmin[d] = input_geom[b][:,d].min()
                bmax[d] = input_geom[b][:,d].max()
            latent_geom[b] = self.latent_geom.to(device=x.device)
            for d in range(3):
                latent_geom[b][:,d] = bmin[d] + (bmax[d] - bmin[d]) * latent_geom[b][:,d]

            # Use T as batch
            y = rearrange(x[:,b,:,:,:,:], 't c d h w -> t (h w d) c')
            out_y = self.gno_in(y=input_geom[b], x=latent_geom[b], f_y=y, key=str(geometry_id) + ":in")
            out[:,b,:,:,:,:] = rearrange(out_y, 't (h w d) c -> t c d h w', d=Dlat, h=Hlat, w=Wlat)

        # Run regular model
        out_model = self.model(out, state_labels, bcs, opts, train_opts)

        # Post-process using GNO
        out = torch.zeros(B, C, D, H, W, device=x.device)
        out_model = rearrange(out, 'b c d h w -> b (h w d) c')
        out = rearrange(out, 'b c d h w -> b (h w d) c')
        for b in range(B):
            out[b] = self.gno_out(y=latent_geom[b], x=input_geom[b], f_y=out_model[b], key=str(geometry_id) + ":out")
        out = rearrange(out, 'b (h w d) c -> b c d h w', d=D, h=H, w=W)

        return out
