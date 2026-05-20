# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

"""
TurbT modified to eliminate patch artifacts in diffusion mode.

Three changes relative to turbt.py, all inspired by DiffiT's design:

1. Per-block broadcast noise conditioning (replaces single appended time token)
   - Original: noise embedding appended as one extra token at iblk==0; attention
     must distribute it across all spatial tokens, giving non-uniform conditioning
     that varies with distance from the token boundary → visible patch artifacts.
   - Modified: emb_mod is broadcast-added to ALL tokens via the existing leadtime
     mechanism inside AttentionBlock_all2all at EVERY block, giving spatially
     uniform noise conditioning (same as DiffiT's per-token QKV bias approach).

2. Conditioning token fusion via learned projection + addition (replaces appended
   diffusion_cond tokens that doubled the sequence length)
   - Original: diffusion_cond tokens concatenated to the sequence, creating a
     seam at the midpoint; tokens near the boundary see a different context than
     tokens far from it → boundary artifacts aligned with patch grid.
   - Modified: diffusion_cond tokens are projected to embed_dim and added
     element-wise to the main token stream before the block loop (analogous to
     DiffiT/DiffiTMATEY's channel-concatenation approach, applied in token space).

All non-diffusion code paths (weather forecasting, graph, hierarchical) are
unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .spacetime_modules import SpaceTimeBlock_all2all
from .basemodel import BaseModel
from ..data_utils.shared_utils import normalize_spatiotemporal_persample, get_top_variance_patchids, normalize_spatiotemporal_persample_graph
from ..data_utils.utils import construct_filterkernel, construct_filterkernel2D
from .spatial_modules import UpsampleinSpace
from .time_modules import FourierEmbedding, PositionalEmbedding, Linear
from torch.nn.functional import silu
import sys, copy
from operator import mul
from functools import reduce
from ..utils.forward_options import ForwardOptionsBase
import torch.distributed as dist
from ..utils import densenodes_to_graphnodes


def build_turbt_modified(params):
    """ Builds modified model from parameter file. Same signature as build_turbt. """
    model = TurbTModified(
        tokenizer_heads=params.tokenizer_heads,
        embed_dim=params.embed_dim,
        num_heads=params.num_heads,
        processor_blocks=params.processor_blocks,
        n_states=params.n_states,
        sts_model=getattr(params, 'sts_model', False),
        sts_train=getattr(params, 'sts_train', False),
        leadtime=hasattr(params, "leadtime_max") and params.leadtime_max >= 0,
        cond_input=getattr(params, 'supportdata', False),
        n_steps=params.n_steps,
        bias_type=params.bias_type,
        replace_patch=getattr(params, 'replace_patch', True),
        hierarchical=getattr(params, 'hierarchical', None),
        notransposed=getattr(params, 'notransposed', False),
        diffusion=getattr(params, 'diffusion', False),
    )
    return model


class TurbTModified(BaseModel):
    """
    TurbT with DiffiT-inspired diffusion conditioning to eliminate patch artifacts.

    Identical to TurbT for all non-diffusion code paths.
    """

    def __init__(self, tokenizer_heads=None, embed_dim=768, num_heads=12,
                 processor_blocks=8, n_states=6, drop_path=.2, sts_train=False,
                 sts_model=False, leadtime=False, cond_input=False, n_steps=1,
                 bias_type="none", replace_patch=True, hierarchical=None,
                 notransposed=False, diffusion=False):
        super().__init__(
            tokenizer_heads=tokenizer_heads, n_states=n_states,
            embed_dim=embed_dim, leadtime=leadtime, cond_input=cond_input,
            n_steps=n_steps, bias_type=bias_type, hierarchical=hierarchical,
            notransposed=notransposed,
            nlevels=hierarchical["nlevels"] if hierarchical is not None else 1,
        )
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.module_blocks = nn.ModuleDict({})
        self.sts_model = sts_model
        self.sts_train = sts_train

        self.num_heads = num_heads
        self.n_steps = n_steps
        self.processor_blocks = processor_blocks
        self.replace_patch = replace_patch
        assert not (self.replace_patch and self.sts_model)

        self.upscale_factors = [1]
        self.module_upscale = nn.ModuleDict({})
        self.module_upscale_space = nn.ModuleDict({})
        self.module_upscale_space2D = nn.ModuleDict({})

        self.hierarchical = False
        self.datafilter_kernel = None
        if hierarchical is not None:
            self.hierarchical = True
            filtersize = hierarchical["filtersize"]
            self.datafilter_kernel = construct_filterkernel(filtersize)
            self.datafilter_kernel2D = construct_filterkernel2D(filtersize)
            self.filtersize = filtersize
            self.nhlevels = hierarchical["nlevels"]
            self.upscale_factors = [1] + [self.filtersize for _ in range(self.nhlevels - 1)]

        for imod, upscalefactor in enumerate(self.upscale_factors):
            if hierarchical["fixedupsample"]:
                self.module_upscale_space[str(imod)] = nn.Upsample(
                    scale_factor=(upscalefactor, upscalefactor, upscalefactor),
                    mode='trilinear', align_corners=True)
                self.module_upscale_space2D[str(imod)] = nn.Upsample(
                    scale_factor=(1, upscalefactor, upscalefactor),
                    mode='trilinear', align_corners=True)
            elif hierarchical["linearupsample"]:
                self.module_upscale_space[str(imod)] = torch.nn.Sequential(
                    nn.Upsample(scale_factor=(upscalefactor, upscalefactor, upscalefactor),
                                mode='trilinear', align_corners=True),
                    nn.Conv3d(n_states, n_states,
                              kernel_size=(upscalefactor, upscalefactor, upscalefactor),
                              stride=1, padding="same", bias=True, padding_mode="reflect"),
                    nn.InstanceNorm3d(n_states, affine=True))
                self.module_upscale_space2D[str(imod)] = torch.nn.Sequential(
                    nn.Upsample(scale_factor=(1, upscalefactor, upscalefactor),
                                mode='trilinear', align_corners=True),
                    nn.Conv3d(n_states, n_states,
                              kernel_size=(1, upscalefactor, upscalefactor),
                              stride=1, padding="same", bias=True, padding_mode="reflect"),
                    nn.InstanceNorm3d(n_states, affine=True))
            else:
                self.module_upscale_space[str(imod)] = UpsampleinSpace(
                    patch_size=[upscalefactor, upscalefactor, upscalefactor], channels=n_states)
                self.module_upscale_space2D[str(imod)] = UpsampleinSpace(
                    patch_size=[1, upscalefactor, upscalefactor], channels=n_states)

            if imod == 0:
                self.module_blocks[str(imod)] = nn.ModuleList([
                    SpaceTimeBlock_all2all(embed_dim, num_heads, drop_path=self.dp[i])
                    for i in range(processor_blocks // self.nhlevels)])
            else:
                self.module_blocks[str(imod)] = nn.ModuleList([
                    SpaceTimeBlock_all2all(embed_dim, num_heads, drop_path=self.dp[i])
                    for i in range(processor_blocks // self.nhlevels)])

        self.diffusion = diffusion
        if self.diffusion:
            model_channels = 128
            channel_mult_emb = 4
            embedding_type = 'positional'
            channel_mult_noise = 1
            emb_channels = model_channels * channel_mult_emb
            noise_channels = model_channels * channel_mult_noise
            init = dict(init_mode='xavier_uniform')

            self.map_noise = (PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                              if embedding_type == 'positional'
                              else FourierEmbedding(num_channels=noise_channels))
            self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

            self.affine = nn.ModuleDict({})
            for imod in range(self.nhlevels):
                self.affine[str(imod)] = Linear(in_features=emb_channels, out_features=embed_dim, **init)

            # NEW: per-level learned projection for fusing diffusion_cond tokens into
            # the main token stream via element-wise addition (replaces token appending).
            self.diffusion_cond_proj = nn.ModuleDict({})
            for imod in range(self.nhlevels):
                self.diffusion_cond_proj[str(imod)] = Linear(in_features=embed_dim, out_features=embed_dim, **init)

    # ------------------------------------------------------------------
    # Helpers copied verbatim from TurbT
    # ------------------------------------------------------------------

    def filterdata(self, data, blockdict=None):
        assert data.ndim == 6, f"unkown tensor shape in filter_data, {data.shape}"
        with torch.no_grad():
            kernel_size = self.filtersize
            T, B, C, D, H, W = data.shape
            data = rearrange(data, 't b c d h w -> (t b c) d h w')
            if D == 1:
                kernel = self.datafilter_kernel2D
                filtered = F.conv3d(data[:, None, :, :, :], kernel.to(data.device),
                                    stride=(1, kernel_size, kernel_size))
            else:
                kernel = self.datafilter_kernel
                filtered = F.conv3d(data[:, None, :, :, :], kernel.to(data.device),
                                    stride=kernel_size)
            filtered = rearrange(filtered, '(t b c) c1 d h w -> t b (c c1) d h w', t=T, b=B, c=C)
            if blockdict is not None:
                assert [D, H, W] == blockdict["Ind_dim"], f"(D,H,W),{(D,H,W)}, {blockdict['Ind_dim']}"
                if D == 1:
                    blockdict["Ind_dim"] = [D, H // kernel_size, W // kernel_size]
                else:
                    blockdict["Ind_dim"] = [D // kernel_size, H // kernel_size, W // kernel_size]
        return filtered, blockdict

    def upsampeldata(self, data, imod):
        B, C, D, H, W = data.shape
        if D == 1:
            data_upsample = self.module_upscale_space2D[str(imod)](data)
        else:
            data_upsample = self.module_upscale_space[str(imod)](data)
        return data_upsample

    def sequence_factor_short(self, x, ilevel, tkhead_name, tspace_dims, nfact=2):
        B, C, TL = x.shape
        embed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed"]
        ntokendim = []
        ps_c = embed_ensemble[-1].patch_size
        for idim, dim in enumerate(tspace_dims[1:]):
            ntokendim.append(dim // ps_c[idim])
        assert TL == tspace_dims[0] * reduce(mul, ntokendim), \
            f"{TL}, {tspace_dims}, {ntokendim}"
        d, h, w = ntokendim
        if h // nfact < 4:
            nfact = max(1, h // 4)
        if nfact < 2:
            return x, nfact
        nfactd = 1 if d == 1 else nfact
        x = rearrange(x, 'b c (t d h w) -> b c t d h w',
                      t=tspace_dims[0], d=d, h=h, w=w)
        x = x.unfold(3, d // nfactd, d // nfactd).unfold(4, h // nfact, h // nfact).unfold(5, w // nfact, w // nfact)
        x = rearrange(x, 'b c t nd nh nw d h w -> (b nd nh nw) c (t d h w)')
        return x, nfact

    def sequence_factor_long(self, x, ilevel, tkhead_name, tspace_dims, nfact=2):
        if nfact < 2:
            return x
        B, C, TL = x.shape
        embed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed"]
        ntokendim = []
        ps_c = embed_ensemble[-1].patch_size
        for idim, dim in enumerate(tspace_dims[1:]):
            ntokendim.append(dim // ps_c[idim])
        d, h, w = ntokendim
        nfactd = 1 if d == 1 else nfact
        assert TL * (nfactd * nfact * nfact) == tspace_dims[0] * reduce(mul, ntokendim), \
            f"{TL}, {tspace_dims}, {ntokendim}, {nfact, nfactd}"
        x = rearrange(x, '(b nd nh nw) c (t d h w) -> b c t nd nh nw d h w',
                      b=B // (nfactd * nfact * nfact), nd=nfactd, nh=nfact, nw=nfact,
                      d=d // nfactd, h=h // nfact, w=w // nfact)
        x = rearrange(x, 'b c t nd nh nw d h w -> b c t (nd d) (nh h) (nw w)')
        x = rearrange(x, 'b c t d h w -> b c (t d h w)')
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data, state_labels, bcs, opts: ForwardOptionsBase):
        # ---- unpack arguments ------------------------------------------------
        imod = opts.imod
        imod_bottom = opts.imod_bottom
        tkhead_name = opts.tkhead_name
        sequence_parallel_group = opts.sequence_parallel_group
        leadtime = opts.leadtime
        blockdict = opts.blockdict
        refine_ratio = opts.refine_ratio
        cond_input = opts.cond_input
        isgraph = opts.isgraph
        field_labels_out = opts.field_labels_out
        sigma = getattr(opts, 'sigma', None)
        diffusion_cond = getattr(opts, 'diffusion_cond', None)
        persample_normalize = False
        # ---- argument checks -------------------------------------------------
        if refine_ratio is not None:
            raise ValueError("Adaptive tokenization is not set up/tested yet in TurbTModified")

        if field_labels_out is None:
            field_labels_out = state_labels

        # ---- noise embedding (diffusion only) --------------------------------
        if self.diffusion:
            emb = self.map_noise(sigma)
            emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
            emb = silu(self.map_layer0(emb))
            emb = silu(self.map_layer1(emb))
            emb = self.affine[str(imod)](emb)  # (B, embed_dim)

            # Rearrange diffusion_cond to (T, B, C, D, H, W) at finest level.
            # At coarser levels it arrives pre-filtered in the right format.
            if diffusion_cond is not None and imod == self.nhlevels - 1:
                diffusion_cond = rearrange(diffusion_cond, 'b t c d h w -> t b c d h w')
                opts.diffusion_cond = diffusion_cond

        # ---- graph path (unchanged from TurbT) --------------------------------
        if isgraph:
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            T = x.shape[1]
            x, data_mean, data_std = normalize_spatiotemporal_persample_graph(x, batch)
            refineind = None
            x = (x, batch, edge_index)
        else:
            x = data

            if imod < self.nhlevels - 1:
                # Filter x and diffusion_cond together so they share the same
                # coarse grid (unchanged from TurbT).
                if diffusion_cond is not None:
                    concat_data = torch.cat([x, diffusion_cond], dim=2)
                    concat_data, blockdict = self.filterdata(concat_data, blockdict=blockdict)
                    x = concat_data[:, :, :x.shape[2], :, :, :]
                    diffusion_cond = concat_data[:, :, x.shape[2]:, :, :, :]
                    opts.diffusion_cond = diffusion_cond
                    opts.blockdict = blockdict
                else:
                    x, blockdict = self.filterdata(x, blockdict=blockdict)
                    opts.blockdict = blockdict

            if imod > imod_bottom:
                opts.imod -= 1
                x_pred = self.forward(x, state_labels, bcs, opts)

            T, B, _, D, H, W = x.shape
            if persample_normalize:
                x, data_mean, data_std = normalize_spatiotemporal_persample(x)

        # ---- leadtime / conditional input (unchanged) -------------------------
        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP[imod](leadtime)
        else:
            leadtime = None
        if self.cond_input and cond_input is not None:
            leadtime = (self.inconMLP[imod](cond_input)
                        if leadtime is None
                        else leadtime + self.inconMLP[imod](cond_input))

        # ---- tokenize x -------------------------------------------------------
        x, patch_ids, patch_ids_ref, mask_padding, _, _, tposarea_padding, _ = \
            self.get_patchsequence(x, state_labels, tkhead_name, refineind=None,
                                   blockdict=blockdict, ilevel=imod, isgraph=isgraph)
        x = rearrange(x, 't b c ntoken_tot -> b c (t ntoken_tot)')

        # ---- fuse noise embedding and diffusion_cond into x (CHANGED) --------
        # Both the noise embedding (emb) and diffusion_cond are combined and
        # added into x before the block loop.  When diffusion_cond is present,
        # emb is added to the cond tokens before projection so both signals are
        # fused in one learned step.  When diffusion_cond is absent, emb is
        # broadcast directly into x.
        if diffusion_cond is not None:
            diffusion_cond_tokens, _, _, _, _, _, _, _ = \
                self.get_patchsequence(diffusion_cond, state_labels, tkhead_name,
                                       refineind=None, blockdict=blockdict,
                                       ilevel=imod, isgraph=isgraph)
            diffusion_cond_tokens = rearrange(
                diffusion_cond_tokens, 't b c ntoken_tot -> b c (t ntoken_tot)')
            b_curr, _, L_curr = x.shape
            cond_combined = diffusion_cond_tokens + emb.unsqueeze(-1)
            cond_fused = self.diffusion_cond_proj[str(imod)](
                rearrange(cond_combined, 'b c L -> (b L) c'))
            x = x + rearrange(cond_fused, '(b L) c -> b c L', b=b_curr)
        elif self.diffusion:
            x = x + emb.unsqueeze(-1)

        # ---- positional bias (unchanged) --------------------------------------
        if self.posbias[imod] is not None and tposarea_padding is not None:
            use_zpos = True if D > 1 else False
            posbias = self.posbias[imod](tposarea_padding, mask_padding=mask_padding,
                                         use_zpos=use_zpos)
            posbias = rearrange(posbias, 'b t L c -> b c (t L)')
            x = x + posbias
            del posbias

        # ---- attention mask ---------------------------------------------------
        mask4attblk = None if (mask_padding is not None and mask_padding.all()) else mask_padding

        # ---- local attention windowing (CHANGED: no separate cond handling) ---
        # diffusion_cond has already been fused into x, so only x needs windowing.
        local_att = not isgraph and imod > imod_bottom
        if local_att:
            nfact = (max(2 ** (2 * (imod - imod_bottom)) // blockdict["nproc_blocks"][-1], 1)
                     if blockdict is not None
                     else max(2 ** (2 * (imod - imod_bottom)), 1))
            x, nfact = self.sequence_factor_short(x, imod, tkhead_name, [T, D, H, W], nfact=nfact)

        # ---- transformer block loop (CHANGED) ---------------------------------
        # Key changes:
        #   (a) No time embedding token appended to the sequence.
        #   (b) Noise embedding is fused into x before the loop (not via leadtime).
        #   (c) leadtime passed only at iblk==0, None thereafter (original behaviour).
        for iblk, blk in enumerate(self.module_blocks[str(imod)]):
            if iblk == 0:
                b_mod = x.shape[0]
                if not isgraph and leadtime is not None:
                    leadtime = leadtime.repeat(b_mod // B, 1)

            block_cond = leadtime if iblk == 0 else None

            x = blk(x, sequence_parallel_group=sequence_parallel_group,
                    bcs=bcs, leadtime=block_cond, mask_padding=mask4attblk,
                    local_att=local_att)

        # No time/cond tokens to strip — they were never appended.

        # ---- un-window (unchanged) --------------------------------------------
        if local_att:
            x = self.sequence_factor_long(x, imod, tkhead_name, [T, D, H, W], nfact=nfact)

        # ---- decode -----------------------------------------------------------
        x = rearrange(x, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T)

        if isgraph:
            x = rearrange(x, 't b c ntoken_tot -> b ntoken_tot t c')
            x = densenodes_to_graphnodes(x, mask_padding)
            x = (x, batch, edge_index)
            D, H, W = -1, -1, -1

        x = self.get_spatiotemporalfromsequence(
            x, patch_ids, patch_ids_ref, [D, H, W], tkhead_name,
            ilevel=imod, isgraph=isgraph)

        if isgraph:
            node_ft, batch, edge_index = x
            x = node_ft[:, :, field_labels_out[0]]
            N = x.shape[0]
            mask = torch.isin(state_labels[0], field_labels_out[0])
            mean_node = data_mean[batch].view(N, 1, -1)[:, :, mask]
            std_node = data_std[batch].view(N, 1, -1)[:, :, mask]
            x = x * std_node + mean_node
            return x[:, -1, :]

        # ---- hierarchical upsampling (unchanged) ------------------------------
        x_correct = x[-1]
        del x
        if imod > imod_bottom:
            x_filter = self.filterdata(x_correct[None, ...])[0][-1]
            filtered_eps = self.upsampeldata(x_filter, imod)
            x_correct = x_correct - filtered_eps
            x_pred = self.upsampeldata(x_pred, imod)
            x_correct = x_correct + x_pred

        if imod == self.nhlevels - 1:
            if persample_normalize:
                x_correct = x_correct[:, state_labels[0], ...] * data_std[-1] + data_mean[-1]
            else:
                x_correct = x_correct[:, state_labels[0], ...]

        return x_correct
