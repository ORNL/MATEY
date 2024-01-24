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
    """    
    FIXME: other types not tested yet
    elif bias_type == 'continuous':
        assert params.time_type == 'attention' and params.space_type == 'axial_attention'
        return ContinuousPositionBias1D(n_heads=params.num_heads)
    elif bias_type == 'rel':
        #FIXME: need to check if still works
        assert params.space_type == '2D_attention'
        return Relative2DPositionBias(n_heads=num_heads, isMLP = biasMLP)
    else:
        if params.time_type == 'attention' and params.space_type == 'axial_attention'
            return RelativePositionBias(n_heads=num_heads)
        else:
            raise ValueError(f"invalid {bias_type} for {params.time_type} {params.space_type}")
    """

class ContinuousPositionBias1D(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.num_heads = n_heads
        self.cpb_mlp = nn.Sequential(nn.Linear(1, 512, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, n_heads, bias=False))
        
    def forward(self, h, h2, bc=0):
        dtype, device = self.cpb_mlp[0].weight.dtype, self.cpb_mlp[0].weight.device
        if bc == 0: # Edges are actual endpoints
            relative_coords = torch.arange(-(h-1), h, dtype=dtype, device=device) / (h-1)
        elif bc == 1: # Periodic boundary conditions - aka opposite edges touch
            relative_coords = torch.cat([torch.arange(1, h//2+1, dtype=dtype, device=device),
                    torch.arange(-(h//2-1), h//2+1, dtype=dtype, device=device),
                    torch.arange(-(h//2-1), 0, dtype=dtype, device=device)
            ])  / (h-1)

        coords = torch.arange(h, dtype=torch.float32, device=device)
        coords = coords[None, :] - coords[:, None]
        coords = coords + (h-1)

        rel_pos_model = 16 * torch.sigmoid(self.cpb_mlp(relative_coords[:, None]).squeeze())
        biases = rel_pos_model[coords.long()]
        return biases.permute(2, 0, 1).unsqueeze(0).contiguous()

class RelativePositionBias(nn.Module):

    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16 

    Implementation of T5 relative position bias - can probably do better, but starting with something known.
    """
    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=32):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, bc=0):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        if bc == 1:
            thresh = klen // 2
            relative_position[relative_position < -thresh] = relative_position[relative_position < -thresh] % thresh
            relative_position[relative_position > thresh] = relative_position[relative_position > thresh] % -thresh
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, qlen, klen, bc=0):
        return self.compute_bias(qlen, klen, bc)  # shape (1, num_heads, qlen, klen)   

class Relative2DPositionBias(nn.Module):

    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16 

    Implementation of T5 relative position bias - can probably do better, but starting with something known.
    """
    def __init__(self, bidirectional=False, num_buckets=32, max_distance=128, n_heads=2, isMLP = False):
        super(Relative2DPositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.isMLP = isMLP
        if self.isMLP:
            self.relative_attention_bias = nn.Sequential(nn.Linear(2, 512, bias=True),
                                                         nn.ReLU(inplace=True),
                                                         nn.Linear(512, self.n_heads, bias=False))
        else:
            self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position_x, relative_position_y, bidirectional=False, num_buckets=32, max_distance=32):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = torch.abs(relative_position_x) + torch.abs(relative_position_y)
        if bidirectional:
            # num_buckets //= 2
            # ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            # n = torch.abs(n)
            raise NotImplementedError
        else:
            n = torch.abs(relative_position_x) + torch.abs(relative_position_y)
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, W, bc=[0, 0]):
        """ Compute binned relative position bias """
        try:
            device = self.relative_attention_bias[0].weight.device
        except:
            device = self.relative_attention_bias.weight.device
        context_position_x = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        context_position_y = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_position_x = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        memory_position_y = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        context_position_x = context_position_x.div(W, rounding_mode="floor")
        memory_position_x  = memory_position_x.div(W, rounding_mode="floor")
        context_position_y = torch.remainder(context_position_y, W)
        memory_position_y  = torch.remainder(memory_position_y, W)

        relative_position_x = memory_position_x - context_position_x  # shape (qlen, klen)
        relative_position_y = memory_position_y - context_position_y  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        if bc[0] == 1:
            thresh = W // 2
            relative_position_x[relative_position_x < -thresh] = relative_position_x[relative_position_x < -thresh] % thresh
            relative_position_x[relative_position_x > thresh] = relative_position_x[relative_position_x > thresh] % -thresh

        if bc[1] ==1:
            thresh = qlen//W// 2
            relative_position_y[relative_position_y < -thresh] = relative_position_y[relative_position_y < -thresh] % thresh
            relative_position_y[relative_position_y > thresh] = relative_position_y[relative_position_y > thresh] % -thresh

        

        if self.isMLP:
            rel_pos_2D_input = torch.stack((relative_position_x, relative_position_y), -1).to(torch.float) # shape (qlen, klen, 2)
            values = self.relative_attention_bias(rel_pos_2D_input) # shape (qlen, klen, num_heads)
            values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)

        else:
            rp_bucket = self._relative_position_bucket(
                relative_position_x,  # shape (qlen, klen)
                relative_position_y,  # shape (qlen, klen)
                bidirectional=self.bidirectional,
                num_buckets=self.num_buckets,
            )
            
            rp_bucket = rp_bucket.to(device)
            values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
            values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)

        return values

    def forward(self, qlen, klen, W, bc=[0, 0]):
        return self.compute_bias(qlen, klen, W, bc)  # shape (1, num_heads, qlen, klen)   

class AbsolutePositionBias(nn.Module):

    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16 

    Implementation of T5 relative position bias - can probably do better, but starting with something known.
    """
    def __init__(self, hidden_dim, n_tokens):
        super(AbsolutePositionBias, self).__init__()
        self.bias = nn.Parameter(torch.randn(1, n_tokens, hidden_dim)*.02)

    def forward(self):
        return self.bias  # shape (1, num_heads, qlen, klen)
     
class PositionAreaBias(nn.Module):
    #adopted from: https://github.com/microsoft/aurora/blob/main/aurora/model/posencoding.py
    #def __init__(self, hidden_dim, time_lower=1, time_upper=100, xy_lower=1e-3, xy_upper=1.0, area_lower=1e-4, area_upper=1.0):
    def __init__(self, hidden_dim, time_lower=1, time_upper=100, xy_lower=1e-3, xy_upper=1.0, area_lower=1e-5, area_upper=1.0):
        super(PositionAreaBias, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_embed  = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pos_embed   = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_embed = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.time_lower = time_lower
        self.time_upper = time_upper
        self.xy_lower = xy_lower
        self.xy_upper = xy_upper
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
    
    def pos_scale_enc(self, pos_t, pos_x, pos_y, patch_area):
        #pos_t, pos_x, pos_y, patch_area in size ([B*L_seq])
        assert self.hidden_dim % 4 == 0
        t_encode = self.fourier_expansion(pos_t, self.hidden_dim, self.time_lower, self.time_upper)  # (B*L_seq, D)
        # Use half of dimensions for the xs of the midpoints of the patches and the other half for the ys
        encode_x = self.fourier_expansion(pos_x, self.hidden_dim // 2, self.xy_lower, self.xy_upper)  # (B*L_seq, D/2)
        encode_y = self.fourier_expansion(pos_y, self.hidden_dim // 2, self.xy_lower, self.xy_upper)  # (B*L_seq, D/2)
        pos_encode = torch.cat((encode_x, encode_y), axis=-1)  # (B*L_seq, D)
        # No need to split things up for the scale encoding.
        scale_encode = self.fourier_expansion(patch_area, self.hidden_dim, self.area_lower, self.area_upper)  # (B*L_seq, D)
        return t_encode, pos_encode, scale_encode

    def forward(self, t_pos_area, bc=None, mask_padding=None):
        """
        #FIXME: BC is not supported yet
        #mask_padding: [B, L_tot] or None
        """
        assert t_pos_area.shape[-1]==4
        if t_pos_area.dim()==5:
            B, T, H, W, C = t_pos_area.shape
            t_pos_area_flat=rearrange(t_pos_area,'b t h w c->(b t h w) c')
            pos_time, pos_x, pos_y, patch_area = (t_pos_area_flat[:, i].squeeze() for i in range(4))
        elif t_pos_area.dim()==4:
            B, T, L_tot, C = t_pos_area.shape
            t_pos_area_flat=rearrange(t_pos_area,'b t L_tot c->(b t L_tot) c')
            if mask_padding is None:
                pos_time, pos_x, pos_y, patch_area = (t_pos_area_flat[:, i].squeeze() for i in range(4))
            else:
                assert B==mask_padding.shape[0] and L_tot==mask_padding.shape[1]
                mask_padding=repeat(mask_padding,'b len -> b t len', t=T)
                mask_padding_BL = rearrange(mask_padding, 'b t len -> (b t len)')
                pos_time, pos_x, pos_y, patch_area = (t_pos_area_flat[mask_padding_BL, i].squeeze() for i in range(4))  
        else:
            raise NotImplementedError
            
        #pos_time:[B*L]; pos_x: [B*L]; pos_y: [B*L]; patch_area: [B*L]
        if bc is not None:
            raise NotImplementedError
        time_enc, pos_enc, scale_enc = self.pos_scale_enc(pos_time, pos_x, pos_y, patch_area)
        time_emb  = self.time_embed(time_enc)
        pos_emb   = self.pos_embed(pos_enc)
        scale_emb = self.scale_embed(scale_enc)
        tot_emb = time_emb + pos_emb + scale_emb #[B*L, hidden_dim]
        if t_pos_area.dim()==5:
            return rearrange(tot_emb,'(b t h w) c -> b t h w c', b=B, t=T, h=H, w=W)
        elif t_pos_area.dim()==4:
            if mask_padding is None:
                return rearrange(tot_emb,'(b t L_tot) c -> b t L_tot c', b=B, t=T)
            else:
                tot_emb_struct = torch.zeros(B*T*L_tot, tot_emb.shape[-1], device=t_pos_area.device)
                tot_emb_struct[mask_padding_BL,:]=tot_emb
            return rearrange(tot_emb_struct,'(b t L_tot) c -> b t L_tot c', b=B, t=T)
        else:
            raise NotImplementedError

