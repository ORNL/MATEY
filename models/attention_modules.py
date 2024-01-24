import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math
from functools import partial
from timm.layers import DropPath
try:
    from .shared_modules import MLP,InstanceNorm1d_Masked
    from .spatial_modules import RMSInstanceNorm2d
except:
    from shared_modules import MLP,InstanceNorm1d_Masked
    from spatial_modules import RMSInstanceNorm2d
import sys

def build_time_block(attention_type, embed_dim, num_heads, bias_type="none"):
    """
    Builds a time block from the parameter file. 
    """
    if attention_type == 'attention':
        return partial(AttentionBlock, embed_dim, num_heads, bias_type=bias_type)
    elif attention_type  =="all2all_time":
        return partial(AttentionBlock_all2all_time, embed_dim, num_heads, bias_type=bias_type)
    else:
        raise NotImplementedError

def build_space_block(attention_type, embed_dim, num_heads, bias_type="none", bias_MLP42D=False):
    if attention_type == 'axial_attention':
        return partial(AxialAttentionBlock, embed_dim, num_heads, bias_type=bias_type)
    elif attention_type == '2D_attention':
        return partial(Attention2DBlock, embed_dim, num_heads, bias_type=bias_type, biasMLP=bias_MLP42D)
    elif attention_type  =="all2all":         
        return partial(AttentionBlock_all2all, embed_dim, num_heads, bias_type=bias_type)
    else:
        raise NotImplementedError

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, drop_path=0, layer_scale_init_value=1e-6, bias_type='rel'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        self.bias_type=bias_type
        """
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == 'continuous':
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        elif bias_type == "PositionAreaBias":
            self.rel_pos_bias = PositionAreaBias(hidden_dim)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        """
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, leadtime=None, t_pos_area=None):
        # input is t x b x c x h x w 
        # t_pos_area: [b, t, h, w, 4]
        T, B, C, H, W = x.shape
        input = x.clone()
        # Rearrange and prenorm
        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.norm1(x)
        x = self.input_head(x) # Q, K, V projections
        # Rearrange for attention
        x = rearrange(x, '(t b) (he c) h w ->  (b h w) he t c', t=T, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        """
        if self.bias_type == "PositionAreaBias":
            assert t_pos_area is not None
            t_pos_area_time = rearrange(t_pos_area, 'b t h w c -> (b h w t) c')
            pos_t, pos_x, pos_y, area = (t_pos_area_time[:, i].squeeze() for i in range(4))
            rel_pos_bias_time = self.rel_pos_bias(pos_t, pos_x, pos_y, area)
            rel_pos_bias_time = rearrange(rel_pos_bias_time, '(b h w t) c -> (t b) c h w', b=B, h=H, w=W, t=T)
            rel_pos_bias = None
        else:
            rel_pos_bias = self.rel_pos_bias(T, T)
        if rel_pos_bias is not None:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_pos_bias) 
        else:
        """
        x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        # Rearrange after attention
        x = rearrange(x, '(b h w) he t c -> (t b) (he c) h w', h=H, w=W)
        """
        if self.bias_type == "PositionAreaBias":
            x = x + rel_pos_bias_time
        """
        x = self.norm2(x) 
        x = self.output_head(x)
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)
        if leadtime is not None:
            x = x + leadtime[None,:,:,None,None]

        output = self.drop_path(x*self.gamma[None, None, :, None, None]) + input
        return output

class AxialAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12,  drop_path=0, layer_scale_init_value=1e-6, bias_type='rel'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        self.bias_type = bias_type
        """
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == 'continuous':
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        elif bias_type == "PositionAreaBias":
            self.rel_pos_bias = PositionAreaBias(hidden_dim)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        """
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = MLP(hidden_dim)
        self.mlp_norm = RMSInstanceNorm2d(hidden_dim, affine=True)

    def forward(self, x, bcs, t_pos_area=None):
        # input is  b x c x h x w 
        # t_pos_area: [b, h, w, 4]
        B, C, H, W = x.shape
        input = x.clone()
        x = self.norm1(x)
        x = self.input_head(x)

        x = rearrange(x, 'b (he c) h w ->  b he h w c', he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)

        # Do attention with current q, k, v matrices along each spatial axis then average results
        # X direction attention
        qx, kx, vx = map(lambda x: rearrange(x, 'b he h w c ->  (b h) he w c'), [q,k,v])
        """
        if self.bias_type == "PositionAreaBias":
            assert t_pos_area is not None
            t_pos_area_x = rearrange(t_pos_area, 'b h w c -> (b h w) c')
            pos_t, pos_x, pos_y, area = (t_pos_area_x[:, i].squeeze() for i in range(4))
            rel_pos_bias_x = self.rel_pos_bias(pos_t, pos_x, pos_y, area)
            rel_pos_bias_x = rearrange(rel_pos_bias_x, '(b h w) c -> b c h w', b=B, h=H, w=W)
        else:
            rel_pos_bias_x = self.rel_pos_bias(W, W, bcs[0, 0])
        
        # Functional doesn't return attention mask :(
        if rel_pos_bias_x is not None and self.bias_type!="PositionAreaBias":
            xx = F.scaled_dot_product_attention(qx, kx, vx, attn_mask=rel_pos_bias_x)
        else:
        """
        xx = F.scaled_dot_product_attention(qx.contiguous(), kx.contiguous(), vx.contiguous())
        xx = rearrange(xx, '(b h) he w c -> b (he c) h w', h=H)
        """
        if self.bias_type == "PositionAreaBias":
            xx = xx + rel_pos_bias_x
        """
        # Y direction attention 
        qy, ky, vy = map(lambda x: rearrange(x, 'b he h w c ->  (b w) he h c'), [q,k,v])
        """
        if self.bias_type == "PositionAreaBias":
            assert t_pos_area is not None
            t_pos_area_y = rearrange(t_pos_area, 'b h w c -> (b w h) c')
            pos_t, pos_x, pos_y, area = (t_pos_area_y[:, i].squeeze() for i in range(4))
            rel_pos_bias_y = self.rel_pos_bias(pos_t, pos_x, pos_y, area)
            rel_pos_bias_y = rearrange(rel_pos_bias_y, '(b w h) c -> b c h w', b=B, h=H, w=W)
        else:
            rel_pos_bias_y = self.rel_pos_bias(H, H, bcs[0, 1])

        if rel_pos_bias_y is not None and self.bias_type!="PositionAreaBias":
            xy = F.scaled_dot_product_attention(qy, ky, vy, attn_mask=rel_pos_bias_y)
        else: # I don't understand why this was necessary but it was
        """
        xy = F.scaled_dot_product_attention(qy.contiguous(), ky.contiguous(), vy.contiguous())
        xy = rearrange(xy, '(b w) he h c -> b (he c) h w', w=W)
        """
        if self.bias_type == "PositionAreaBias":
            xy = xy + rel_pos_bias_y
        """
        # Combine
        x = (xx + xy) / 2
        x = self.norm2(x)
        x = self.output_head(x)
        x = self.drop_path(x*self.gamma_att[None, :, None, None]) + input

        # MLP
        input = x.clone()
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.mlp(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None, :, None, None] * x)

        return output

class Attention2DBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12,  drop_path=0, layer_scale_init_value=1e-6, bias_type='rel', biasMLP=False):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        self.vnorm = nn.LayerNorm(hidden_dim//num_heads)
        """
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y, W, bcs: None
        elif bias_type == 'continuous':
            raise NotImplementedError
            #self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        elif bias_type == 'rel':
            # raise NotImplementedError
            self.rel_pos_bias = Relative2DPositionBias(n_heads=num_heads, isMLP = biasMLP)
        else:
            raise NotImplementedError
        """
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = MLP(hidden_dim)
        self.mlp_norm = RMSInstanceNorm2d(hidden_dim, affine=True)

    #from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    def posemb_sincos_2d(self, h, w, dim, temperature: int = 10000, dtype = torch.float32):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)
    
    def debug_nan(self,x, message=""):
        #print(message, "min, max", torch.min(x), torch.max(x))
        if torch.isinf(x).any():
            print(f"INF detected in prediction, after {message}: ",x)

        if torch.isnan(x).any():
            print(f"NAN detected in prediction, after {message}: ",x)
            for name, param in self.named_parameters():
                if param.requires_grad and torch.isnan(param.data).any():
                    print("NAN detected in model parameters: ", name, param.data.numel())
                else:
                    print("No NAN in model parameters: ", name, param.data.numel())
            sys.exit(-1)

    def forward(self, x, bcs):
        # input is t x b x c x h x w 
        B, C, H, W = x.shape
        input = x.clone()

        self.debug_nan(x, message="attension, begin, x")

        x = self.norm1(x)

        self.debug_nan(x, message="attension, after norm1, x")

        x = self.input_head(x)

        x = rearrange(x, 'b (he c) h w ->  b he h w c', he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)

        self.debug_nan(q, message="after qkv split, q")
        self.debug_nan(k, message="after qkv split, k")
        self.debug_nan(v, message="after qkv split, v")

        q, k, v = self.qnorm(q), self.knorm(k), self.vnorm(v)


        self.debug_nan(q, message="normalized, before attention part, q")
        self.debug_nan(k, message="normalized, before attention part, k")
        self.debug_nan(v, message="normalized, before attention part, v")


        # Do 2D attention 
        q, k, v = map(lambda x: rearrange(x, 'b he h w c ->  b he (h w) c'), [q,k,v])
        """
        rel_pos_bias = self.rel_pos_bias(H*W, H*W, W, bcs[0, :])
        # Functional doesn't return attention mask :(
        if rel_pos_bias is not None:
            # raise NotImplementedError
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_pos_bias)
        else:
        """
        x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        x = rearrange(x, 'b he (h w) c -> b he h w c', h=H)
        x = rearrange(x, 'b he h w c -> b (he c) h w')

        self.debug_nan(x, message="after scaled_dot_product_attention, x")

        x = self.norm2(x)
        x = self.output_head(x)
        x = self.drop_path(x*self.gamma_att[None, :, None, None]) + input

        # MLP
        input = x.clone()
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.mlp(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None, :, None, None] * x)

        self.debug_nan(x, message="after mlp_norm, x")

        return output
    
class AttentionBlock_all2all_time(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, drop_path=0, layer_scale_init_value=1e-6, bias_type='rel'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.input_head  = nn.Linear(hidden_dim, 3*hidden_dim)
        self.output_head = nn.Linear(hidden_dim, hidden_dim)

        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        """
        if bias_type == 'none':
            self.rel_pos_bias = None
        elif bias_type == "PositionAreaBias":
            self.rel_pos_bias = PositionAreaBias(hidden_dim)
        else:
            raise NotImplementedError
        """
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 

    def forward(self, x, leadtime=None):
        #x: b x c x token_len (len)  
        #leadtime: btoken_len x c
        #t_pos_area: token_len x 4
        B, C, len = x.shape
        input = x.clone()
        
        # Rearrange and prenorm
        x = self.norm1(x)
        x = rearrange(x, 'b c len -> (b len) c')
        x = self.input_head(x) # Q, K, V projections
        # Rearrange for attention        
        x = rearrange(x, '(b len) (he c) ->  b he len c', len=len, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        x = F.scaled_dot_product_attention(q, k, v) 
        
        # Rearrange after attention
        x = rearrange(x, 'b  he len c -> b (he c) len')
        """
        if self.rel_pos_bias is not None:
            if t_pos_area is not None:
                pos_t, pos_x, pos_y, area = (t_pos_area[:, i].squeeze() for i in range(4))
                rel_pos_bias = self.rel_pos_bias(pos_t, pos_x, pos_y, area)
                rel_pos_bias = rearrange(rel_pos_bias, '(b len) c -> b c len', b=B)
                x = x + rel_pos_bias
        """

        x = self.norm2(x) 
        x = rearrange(x, 'b c len -> (b len) c')
        x = self.output_head(x)
        x = rearrange(x, '(b len) c-> b c len', b=B)
        if leadtime is not None:
            x = x + leadtime[:,:, None]

        x = self.drop_path(x*self.gamma_att[None, :, None]) + input

        return x
        
class AttentionBlock_all2all(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, drop_path=0, layer_scale_init_value=1e-6, bias_type='rel'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.norm1_mask = InstanceNorm1d_Masked(hidden_dim, affine=True, gamma=self.norm1.weight, beta=self.norm1.bias)
        self.norm2_mask = InstanceNorm1d_Masked(hidden_dim, affine=True, gamma=self.norm2.weight, beta=self.norm2.bias)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.input_head  = nn.Linear(hidden_dim, 3*hidden_dim)
        self.output_head = nn.Linear(hidden_dim, hidden_dim)

        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        """
        if bias_type == 'none':
            self.rel_pos_bias = None
        elif bias_type == "PositionAreaBias":
            self.rel_pos_bias = PositionAreaBias(hidden_dim)
        else:
            raise NotImplementedError
        """
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 

        self.mlp = MLP(hidden_dim)
        self.mlp_norm = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.mlp_norm_mask = InstanceNorm1d_Masked(hidden_dim, affine=True, gamma=self.mlp_norm.weight, beta=self.mlp_norm.bias)

    def forward_padding(self, x, mask_padding, bcs=None, leadtime=None, t_pos_area_padding=None):
        # x: b x c x token_len (len)  
        # mask_padding: b x token_len
        # leadtime: b x c
        B, C, len = x.shape
        input = x.clone()
        x = self.norm1_mask(x, mask_padding)
        x = rearrange(x, 'b c len -> (b len) c')
        mask_padding_BL = rearrange(mask_padding, 'b len -> (b len)')
        # Rearrange and prenorm
        x_qkv = repeat(x, 'blen c -> blen (n3 c)', n3=3)
        x_qkv[mask_padding_BL, :] = self.input_head(x[mask_padding_BL]) # Q, K, V projections
        x = x_qkv
        # Rearrange for attention        
        x = rearrange(x, '(b len) (he c) ->  b he len c', len=len, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q = rearrange(q, 'b he len c -> (b len) he c')
        k = rearrange(k, 'b he len c -> (b len) he c')
        q[mask_padding_BL], k[mask_padding_BL] = self.qnorm(q[mask_padding_BL]), self.knorm(k[mask_padding_BL])
        q = rearrange(q, '(b len) he c -> b he len c', b=B)
        k = rearrange(k, '(b len) he c -> b he len c', b=B)

        mask2d_padding = torch.logical_and(mask_padding[:,:,None], mask_padding[:,None, :])
        mask2d_padding = repeat(mask2d_padding, 'b len lensrc-> b he len lensrc', he=self.num_heads)
        #Note: Attention boolean mask is not properly implemented in F.scaled_dot_product_attention (Pei, July 2024)
        ### we use float masking, instead
        mask2d_padding = mask2d_padding.float()*1e6-1e6
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask2d_padding) 
        if torch.isnan(x).any():
            print(x)
            sys.exit(-1)
        # Rearrange after attention
        x = rearrange(x, 'b  he len c -> b (he c) len')
        """
        or x = rearrange(x, 'b  he len c -> (b len) (he c)')
        if t_pos_area_padding is not None:
            pos_t, pos_x, pos_y, area = (t_pos_area_padding[mask_padding_BL, i].squeeze() for i in range(4))
            rel_pos_bias = self.rel_pos_bias(pos_t, pos_x, pos_y, area)
            x[mask_padding_BL, :] = x[mask_padding_BL, :] + rel_pos_bias        
            x = rearrange(x, '(b len) c -> b c len', b=B)
        """

        x = self.norm2_mask(x, mask_padding)
        x = rearrange(x, 'b  c  len -> (b len) c')
        input = rearrange(input, 'b  c  len -> (b len) c')
        x[mask_padding_BL] = self.output_head(x[mask_padding_BL])
        x[mask_padding_BL] = self.drop_path(x[mask_padding_BL]*self.gamma_att[None, :]) + input[mask_padding_BL]
        if leadtime is not None:
            leadtime = leadtime.repeat_interleave(len, dim=0)
            x[mask_padding_BL] = x[mask_padding_BL] + leadtime[mask_padding_BL,:]


        # MLP
        input = x.clone()
        x[mask_padding_BL] = self.mlp(x[mask_padding_BL])
        x = rearrange(x, '(b len) c -> b c len', b=B)
        x = self.mlp_norm_mask(x, mask_padding)
        x = rearrange(x, 'b  c  len -> (b len) c')
        x[mask_padding_BL] = self.drop_path(x[mask_padding_BL]*self.gamma_mlp[None, :]) + input[mask_padding_BL]
        x = rearrange(x, '(b len) c-> b c len', b=B)
    
        if torch.isnan(x).any():
            print(x)
            sys.exit(-1)

        return x

    def forward(self, x, leadtime=None, bcs=None, mask_padding=None, t_pos_area=None):
        #x: b x c x token_len (len)  
        #leadtime: btoken_len x c
        #t_pos_area: token_len x 4
        B, C, len = x.shape
        input = x.clone()
        if mask_padding is not None:
            return self.forward_padding(x, mask_padding, bcs=bcs, leadtime=leadtime, t_pos_area_padding=t_pos_area)
        
        # Rearrange and prenorm
        x = self.norm1(x)
        x = rearrange(x, 'b c len -> (b len) c')
        x = self.input_head(x) # Q, K, V projections
        # Rearrange for attention        
        x = rearrange(x, '(b len) (he c) ->  b he len c', len=len, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        x = F.scaled_dot_product_attention(q, k, v) 
        
        # Rearrange after attention
        x = rearrange(x, 'b  he len c -> b (he c) len')
        """
        if self.rel_pos_bias is not None:
            if t_pos_area is not None:
                pos_t, pos_x, pos_y, area = (t_pos_area[:, i].squeeze() for i in range(4))
                rel_pos_bias = self.rel_pos_bias(pos_t, pos_x, pos_y, area)
                rel_pos_bias = rearrange(rel_pos_bias, '(b len) c -> b c len', b=B)
                x = x + rel_pos_bias
        """

        x = self.norm2(x) 
        x = rearrange(x, 'b c len -> (b len) c')
        x = self.output_head(x)
        x = rearrange(x, '(b len) c-> b c len', b=B)
        if leadtime is not None:
            x = x + leadtime[:,:, None]

        x = self.drop_path(x*self.gamma_att[None, :, None]) + input

        # MLP
        input = x.clone()
        x = rearrange(x, 'b c len-> b len c')
        x = self.mlp(x)
        x = rearrange(x, 'b len c -> b c len')
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None, :, None] * x)

        return output
