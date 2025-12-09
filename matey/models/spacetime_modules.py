import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from .attention_modules import build_space_block, build_time_block, AttentionBlock_all2all

class SpaceTimeBlock(nn.Module):
    """
    Alternates spatial and temporal processing. Current code base uses
    1D attention over each axis. Spatial axes share weights.
    """
    def __init__(self, space_type, time_type, embed_dim, num_heads, bias_type="none", drop_path=0.0):
        super().__init__()
        space_block = build_space_block(space_type, embed_dim, num_heads, bias_type=bias_type)
        time_block  = build_time_block(time_type, embed_dim, num_heads, bias_type=bias_type)


        self.spatial = space_block(drop_path=drop_path)
        self.temporal= time_block(drop_path=drop_path)

    def forward(self, x, bcs, sequence_parallel_group=None, leadtime=None, t_pos_area=None):
        # input is: [T, B, C, D, H, W ]
        # t_pos_area: [b, t, d, h, w, 5]
        T = x.shape[0]

        # Time attention
        x = self.temporal(x, leadtime=leadtime, t_pos_area=t_pos_area, sequence_parallel_group=sequence_parallel_group) # Residual in block
        # Temporal handles the rearrange so still is t x b x c x d x h x w

        # Now do spatial attention
        x = rearrange(x, 't b c d h w -> (t b) c d h w')

        if t_pos_area is not None:
            t_pos_area = rearrange(t_pos_area, 'b t d h w c -> (t b) d h w c')

        if hasattr(self, 'spatial'):
            x = self.spatial(x, bcs, t_pos_area=t_pos_area, sequence_parallel_group=sequence_parallel_group) # Convnext has the residual in the block
        else:
            print("Warning: model without spatial")

        x = rearrange(x, '(t b) c d h w -> t b c d h w', t=T)

        return x

class SpaceTimeBlock_svit(nn.Module):
    """
    Alternates spatial and temporal processing.
    """
    def __init__(self, space_type, time_type, embed_dim, num_heads, bias_type="none", drop_path=0.0):
        super().__init__()
        if not(time_type=="all2all_time" and space_type=="all2all"):
            raise NotImplementedError

        space_block = build_space_block(space_type, embed_dim, num_heads, bias_type=bias_type)
        time_block  = build_time_block(time_type, embed_dim, num_heads, bias_type=bias_type)


        self.temporal = time_block(drop_path=drop_path)
        self.spatial = space_block(drop_path=drop_path)

    def forward(self, x, bcs, sequence_parallel_group=None, leadtime=None, mask_padding=None, t_pos_area=None):
        # input is t x b x c x slen
        # t_pos_area:[b, t, slen, 5]
        T, B, C, slen = x.shape

        # Time attention
        x = rearrange(x, 't b c slen -> (b slen) c t')
        if leadtime is not None:
            leadtime = leadtime.repeat_interleave(slen, dim=0)
        if mask_padding is not None:
            mask_padding = rearrange(mask_padding, 'b slen -> (b slen)')
            """
            t_pos_area_time = rearrange(t_pos_area, 'b t slen c -> (b slen) t c')[mask_padding]
            t_pos_area_time = rearrange(t_pos_area_time, 'b t c -> (b t) c')
            """
            x[mask_padding] = self.temporal(x[mask_padding], leadtime=leadtime[mask_padding]
                                            if leadtime is not None else leadtime, sequence_parallel_group=sequence_parallel_group)#, t_pos_area=t_pos_area_time) # Residual in block
            # Now do spatial attention
            x = rearrange(x, '(b slen) c t -> (b t) c slen', slen=slen)
            mask_padding = rearrange(mask_padding, '(b slen) -> b slen', b=B)
            mask_padding = mask_padding.repeat_interleave(T, dim=0)
            #t_pos_area_xy = rearrange(t_pos_area, 'b t slen c -> (b t slen) c')
            x = self.spatial(x, bcs=None, mask_padding=mask_padding, sequence_parallel_group=sequence_parallel_group) #, t_pos_area=t_pos_area_xy) # Convnext has the residual in the block
        else:
            #t_pos_area_time = rearrange(t_pos_area, 'b t slen c -> (b slen t) c')
            x = self.temporal(x, leadtime=leadtime, sequence_parallel_group=sequence_parallel_group)#, t_pos_area=t_pos_area_time) # Residual in block
            # Now do spatial attention
            x = rearrange(x, '(b slen) c t -> (b t) c slen', slen=slen)
            #t_pos_area_xy = rearrange(t_pos_area, 'b t slen c -> (b t slen) c')
            x = self.spatial(x, bcs=None, sequence_parallel_group=sequence_parallel_group)#, t_pos_area=t_pos_area_xy) # Convnext has the residual in the block
        x = rearrange(x, '(b t) c slen -> t b c slen', t=T)

        return x

class SpaceTimeBlock_all2all(nn.Module):
    """
    Alternates spatial and temporal processing.
    """
    def __init__(self, embed_dim, num_heads, bias_type='none', drop_path=0.):
        super().__init__()
        self.spatialtemporal = AttentionBlock_all2all(hidden_dim=embed_dim, num_heads=num_heads, bias_type=bias_type, drop_path=drop_path)

    def forward(self, x, sequence_parallel_group=None, bcs=None, leadtime=None, input_control=None, mask_padding=None, t_pos_area=None, local_att=False):
        # x:  b x c x tslen
        # leadtime: [b, c]
        # mask_padding: [b, slen]
        # t_pos_area:[b*tslen, 5]
        B, C, tslen = x.shape
        if mask_padding is not None:
            _, slen = mask_padding.shape
            T = tslen//slen
            mask_padding = repeat(mask_padding, 'b slen-> b (t slen)', t=T)
        # attention
        x = self.spatialtemporal(x, sequence_parallel_group=sequence_parallel_group, leadtime=leadtime, input_control=input_control, mask_padding=mask_padding, t_pos_area=t_pos_area, local_att=local_att) # Residual in block
        return x

