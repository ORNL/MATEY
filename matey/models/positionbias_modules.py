# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat

def positionbias_mod(bias_type, embed_dim):
    """
    Builds a time block from the parameter file.
    """
    if bias_type=="none":
        return None
    elif bias_type == "PositionAreaBias":
        return PositionAreaBias(embed_dim)
    else:
        raise NotImplementedError

class PositionAreaBias(nn.Module):
    #adopted from: https://github.com/microsoft/aurora/blob/main/aurora/model/posencoding.py
    #def __init__(self, hidden_dim, time_lower=1, time_upper=100, xy_lower=1e-3, xy_upper=1.0, area_lower=1e-4, area_upper=1.0):
    #def __init__(self, hidden_dim, time_lower=1, time_upper=100, zxy_lower=1e-3, zxy_upper=1.0, area_lower=1e-8, area_upper=1.0):
    def __init__(self, hidden_dim, time_lower=1, time_upper=100, zxy_lower=1e-4, zxy_upper=1.0, area_lower=1e-10, area_upper=1.0):
        super(PositionAreaBias, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_embed  = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pos_embed   = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_embed = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.time_lower = time_lower
        self.time_upper = time_upper
        self.zxy_lower = zxy_lower
        self.zxy_upper = zxy_upper
        self.area_lower = area_lower
        self.area_upper = area_upper

    def fourier_expansion(self, x, d, lower, upper):
        #add aurora link
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(lower <= x.abs(), torch.all(x.abs() <= upper))
        in_range_or_zero = torch.all(torch.logical_or(in_range, x == 0))
        if not in_range_or_zero:
            raise AssertionError(f"The input tensor is not within the configured range `[{lower}, {upper}]` {x.min(), x.max(), x.unique()}.")
        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if not (d % 2 == 0):
            raise ValueError("The dimensionality must be a multiple of two.")
        x = x.double() #([B*L_seq])
        wavelengths = torch.logspace(math.log10(lower), math.log10(upper), d // 2, base=10, device=x.device, dtype=x.dtype) #([d/2])
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths) #([B*L_seq, d//2])
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return encoding.float() #([B*L_seq, d])

    def pos_scale_enc(self, pos_t, pos_z, pos_x, pos_y, patch_area, use_zpos=False):
        #pos_t, pos_z, pos_x, pos_y, patch_area in size ([B*L_seq])
        assert self.hidden_dim % 4 == 0
        t_encode = self.fourier_expansion(pos_t, self.hidden_dim, self.time_lower, self.time_upper)  # (B*L_seq, D)
        if use_zpos:
            # Use 1/3 of dimensions for the zs of the midpoints of the patches, 1/3 for xs and the other 1/3 for the ys
            encode_z = self.fourier_expansion(pos_z, self.hidden_dim // 3, self.zxy_lower, self.zxy_upper)  # (B*L_seq, D/3)
            encode_x = self.fourier_expansion(pos_x, self.hidden_dim // 3, self.zxy_lower, self.zxy_upper)  # (B*L_seq, D/3)
            encode_y = self.fourier_expansion(pos_y, self.hidden_dim-(self.hidden_dim//3)*2, self.zxy_lower, self.zxy_upper)  # (B*L_seq, D/3)
            pos_encode = torch.cat((encode_z, encode_x, encode_y), axis=-1)  # (B*L_seq, D)
        else:
            # Use half of dimensions for the xs of the midpoints of the patches and the other half for the ys
            encode_x = self.fourier_expansion(pos_x, self.hidden_dim // 2, self.zxy_lower, self.zxy_upper)  # (B*L_seq, D/2)
            encode_y = self.fourier_expansion(pos_y, self.hidden_dim // 2, self.zxy_lower, self.zxy_upper)  # (B*L_seq, D/2)
            pos_encode = torch.cat((encode_x, encode_y), axis=-1)  # (B*L_seq, D)
        # No need to split things up for the scale encoding.
        scale_encode = self.fourier_expansion(patch_area, self.hidden_dim, self.area_lower, self.area_upper)  # (B*L_seq, D)
        return t_encode, pos_encode, scale_encode

    def forward(self, t_pos_area, bc=None, mask_padding=None, use_zpos=False):
        """
        #FIXME: BC is not supported yet
        #mask_padding: [B, L_tot] or None
        """
        assert t_pos_area.shape[-1]==5
        if t_pos_area.dim()==6:
            B, T, D, H, W, C = t_pos_area.shape
            t_pos_area_flat=rearrange(t_pos_area,'b t d h w c->(b t d h w) c')
            pos_time, pos_z, pos_x, pos_y, patch_area = (t_pos_area_flat[:, i].squeeze() for i in range(5))
        elif t_pos_area.dim()==4:
            B, T, L_tot, C = t_pos_area.shape
            t_pos_area_flat=rearrange(t_pos_area,'b t L_tot c->(b t L_tot) c')
            if mask_padding is None:
                pos_time, pos_z, pos_x, pos_y, patch_area = (t_pos_area_flat[:, i].squeeze() for i in range(5))
            else:
                assert B==mask_padding.shape[0] and L_tot==mask_padding.shape[1]
                mask_padding=repeat(mask_padding,'b len -> b t len', t=T)
                mask_padding_BL = rearrange(mask_padding, 'b t len -> (b t len)')
                pos_time, pos_z, pos_x, pos_y, patch_area = (t_pos_area_flat[mask_padding_BL, i].squeeze() for i in range(5))
        else:
            raise NotImplementedError

        #pos_time:[B*L]; pos_x: [B*L]; pos_y: [B*L]; patch_area: [B*L]
        if bc is not None:
            raise NotImplementedError
        time_enc, pos_enc, scale_enc = self.pos_scale_enc(pos_time, pos_z, pos_x, pos_y, patch_area, use_zpos=use_zpos)
        time_emb  = self.time_embed(time_enc)
        pos_emb   = self.pos_embed(pos_enc)
        scale_emb = self.scale_embed(scale_enc)
        tot_emb = time_emb + pos_emb + scale_emb #[B*L, hidden_dim]
        if t_pos_area.dim()==6:
            return rearrange(tot_emb,'(b t d h w) c -> b t d h w c', b=B, t=T, d=D, h=H, w=W)
        elif t_pos_area.dim()==4:
            if mask_padding is None:
                return rearrange(tot_emb,'(b t L_tot) c -> b t L_tot c', b=B, t=T)
            else:
                tot_emb_struct = torch.zeros(B*T*L_tot, tot_emb.shape[-1], device=t_pos_area.device)
                tot_emb_struct[mask_padding_BL,:]=tot_emb
            return rearrange(tot_emb_struct,'(b t L_tot) c -> b t L_tot c', b=B, t=T)
        else:
            raise NotImplementedError

