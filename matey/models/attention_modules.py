import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math
from functools import partial
from timm.layers import DropPath
from .ringX_attn import ringX_attn_func
from .shared_modules import MLP, InstanceNorm1d_Masked
from .spatial_modules import RMSInstanceNormSpace

import sys
from flash_attn import flash_attn_func

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
        self.norm1 = nn.InstanceNorm3d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm3d(hidden_dim, affine=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)),
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.input_head = nn.Conv3d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        self.bias_type=bias_type
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, leadtime=None, sequence_parallel_group=None, t_pos_area=None, local_att=False):
        # input is t x b x c x d xh x w
        # t_pos_area: [b, t, d, h, w, 5]
        T, _, _, D, H, W = x.shape
        input = x.clone()
        # Rearrange and prenorm
        x = rearrange(x, 't b c d h w -> (t b) c d h w')
        x = self.norm1(x)
        x = self.input_head(x) # Q, K, V projections
        # Rearrange for attention
        x = rearrange(x, '(t b) (he c) d h w ->  (b d h w) he t c', t=T, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        if local_att or sequence_parallel_group is None:
            #x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
            dtype = x.dtype
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16) 
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            x = flash_attn_func(q, k, v) #(b, len, he, c)
            x = x.transpose(1,2).to(dtype)
        else:
            # use ringX 
            dtype = x.dtype
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)  
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            loc_x = ringX_attn_func(q, k, v, causal=False, group=sequence_parallel_group) #all ddp ranks      
            x = loc_x
            x = x.transpose(1,2).to(dtype)
        # Rearrange after attention
        x = rearrange(x, '(b d h w) he t c -> (t b) (he c) d h w', d=D, h=H, w=W)
        x = self.norm2(x)
        x = self.output_head(x)
        x = rearrange(x, '(t b) c d h w -> t b c d h w', t=T)
        if leadtime is not None:
            x = x + leadtime[None,:,:,None, None,None]

        output = self.drop_path(x*self.gamma[None, None, :, None, None, None]) + input
        return output

class AxialAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12,  drop_path=0, layer_scale_init_value=1e-6, bias_type='rel'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSInstanceNormSpace(hidden_dim, affine=True)
        self.norm2 = RMSInstanceNormSpace(hidden_dim, affine=True)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)),
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)),
                            requires_grad=True) if layer_scale_init_value > 0 else None

        self.input_head = nn.Conv3d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        self.bias_type = bias_type
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = MLP(hidden_dim)
        self.mlp_norm = RMSInstanceNormSpace(hidden_dim, affine=True)

    def forward(self, x, bcs, sequence_parallel_group=None, t_pos_area=None, local_att=False):
        # input is  b x c x d x h x w
        # t_pos_area: [b, d, h, w, 5]
        D, H, W = x.shape[2:]
        input = x.clone()
        x = self.norm1(x)
        x = self.input_head(x)

        x = rearrange(x, 'b (he c) d h w ->  b he d h w c', he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)

        # Do attention with current q, k, v matrices along each spatial axis then average results
       
        # X direction attention
        qx, kx, vx = map(lambda x: rearrange(x, 'b he d h w c ->  (b d h) he w c'), [q,k,v])
        if local_att or sequence_parallel_group is None:
            #xx = F.scaled_dot_product_attention(qx.contiguous(), kx.contiguous(), vx.contiguous())
            dtype = x.dtype
            qx, kx, vx = qx.to(torch.bfloat16), kx.to(torch.bfloat16), vx.to(torch.bfloat16) 
            qx, kx, vx = qx.transpose(1,2), kx.transpose(1,2), vx.transpose(1,2)
            xx = flash_attn_func(qx, kx, vx) #(b, len, he, c)
            xx = xx.transpose(1,2).to(dtype)
        else:
            # use ringX 
            dtype = x.dtype
            qx, kx, vx = qx.to(torch.bfloat16), kx.to(torch.bfloat16), vx.to(torch.bfloat16)  
            qx, kx, vx = qx.transpose(1,2), kx.transpose(1,2), vx.transpose(1,2)
            loc_xx = ringX_attn_func(qx, kx, vx, causal=False, group=sequence_parallel_group) #all ddp ranks      
            xx = loc_xx
            xx = xx.transpose(1,2).to(dtype)
        xx = rearrange(xx, '(b d h) he w c -> b (he c) d h w', d=D, h=H)
        # Y direction attention
        qy, ky, vy = map(lambda x: rearrange(x, 'b he d h w c ->  (b d w) he h c'), [q,k,v])
        if local_att or sequence_parallel_group is None:
            #xy = F.scaled_dot_product_attention(qy.contiguous(), ky.contiguous(), vy.contiguous())
            dtype = xx.dtype
            qy, ky, vy = qy.to(torch.bfloat16), ky.to(torch.bfloat16), vy.to(torch.bfloat16) 
            qy, ky, vy = qy.transpose(1,2), ky.transpose(1,2), vy.transpose(1,2)
            xy = flash_attn_func(qy, ky, vy) #(b, len, he, c)
            xy = xy.transpose(1,2).to(dtype)
        else:
            # use ringX 
            dtype = xx.dtype
            qy, ky, vy = qy.to(torch.bfloat16), ky.to(torch.bfloat16), vy.to(torch.bfloat16)  
            qy, ky, vy = qy.transpose(1,2), ky.transpose(1,2), vy.transpose(1,2)
            loc_xy = ringX_attn_func(qy, ky, vy, causal=False, group=sequence_parallel_group) #all ddp ranks      
            xy = loc_xy
            xy = xy.transpose(1,2).to(dtype)
        xy = rearrange(xy, '(b d w) he h c -> b (he c) d h w', d=D, w=W)
        # Combine
        if D>1:
             # Z direction attneion
            qz, kz, vz = map(lambda x: rearrange(x, 'b he d h w c ->  (b h w) he d c'), [q,k,v])
            if local_att or sequence_parallel_group is None:
                #xz = F.scaled_dot_product_attention(qz.contiguous(), kz.contiguous(), vz.contiguous())
                dtype = xy.dtype
                qz, kz, vz = qz.to(torch.bfloat16), kz.to(torch.bfloat16), vz.to(torch.bfloat16) 
                qz, kz, vz = qz.transpose(1,2), kz.transpose(1,2), vz.transpose(1,2)
                xz = flash_attn_func(qz, kz, vz) #(b, len, he, c)
                xz = xz.transpose(1,2).to(dtype)
            else:
                # use ringX 
                dtype = xy.dtype
                qz, kz, vz = qz.to(torch.bfloat16), kz.to(torch.bfloat16), vz.to(torch.bfloat16)  
                qz, kz, vz = qz.transpose(1,2), kz.transpose(1,2), vz.transpose(1,2)
                loc_xz = ringX_attn_func(qz, kz, vz, causal=False, group=sequence_parallel_group) #all ddp ranks      
                xz = loc_xz
                xz = xz.transpose(1,2).to(dtype)
            xz = rearrange(xz, '(b h w) he d c -> b (he c) d h w', h=H, w=W)

            x = (xz + xx + xy) / 3.0
        else:
            x = (xx + xy) / 2.0
        x = self.norm2(x)
        x = self.output_head(x)
        x = self.drop_path(x*self.gamma_att[None, :, None, None, None]) + input

        # MLP
        input = x.clone()
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.mlp(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None, :, None, None, None] * x)

        return output

class Attention2DBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12,  drop_path=0, layer_scale_init_value=1e-6, bias_type='rel', biasMLP=False):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSInstanceNormSpace(hidden_dim, affine=True)
        self.norm2 = RMSInstanceNormSpace(hidden_dim, affine=True)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)),
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)),
                            requires_grad=True) if layer_scale_init_value > 0 else None

        self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        self.vnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = MLP(hidden_dim)
        self.mlp_norm = RMSInstanceNormSpace(hidden_dim, affine=True)

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

    def forward(self, x, bcs, sequence_parallel_group=None):
        # input is t x b x c x h x w
        B, C, H, W = x.shape
        input = x.clone()

        x = self.norm1(x)

        x = self.input_head(x)

        x = rearrange(x, 'b (he c) h w ->  b he h w c', he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)

        q, k, v = self.qnorm(q), self.knorm(k), self.vnorm(v)


        # Do 2D attention
        q, k, v = map(lambda x: rearrange(x, 'b he h w c ->  b he (h w) c'), [q,k,v])
        x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        x = rearrange(x, 'b he (h w) c -> b he h w c', h=H)
        x = rearrange(x, 'b he h w c -> b (he c) h w')

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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, sequence_parallel_group=None, leadtime=None, local_att=False):
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
        if local_att or sequence_parallel_group is None:
            #x = F.scaled_dot_product_attention(q, k, v)
            dtype = x.dtype
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16) 
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            x = flash_attn_func(q, k, v) #(b, len, he, c)
            x = x.transpose(1,2).to(dtype)
        else:
            # use ringX 
            dtype = x.dtype
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)  
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            loc_x = ringX_attn_func(q, k, v, causal=False, group=sequence_parallel_group) #all ddp ranks      
            x = loc_x
            x = x.transpose(1,2).to(dtype)

        # Rearrange after attention
        x = rearrange(x, 'b  he len c -> b (he c) len')

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
        self.norm1_mask = InstanceNorm1d_Masked(hidden_dim, affine=True)
        self.norm2_mask = InstanceNorm1d_Masked(hidden_dim, affine=True)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)),
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.input_head  = nn.Linear(hidden_dim, 3*hidden_dim)
        self.output_head = nn.Linear(hidden_dim, hidden_dim)

        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = MLP(hidden_dim)
        self.mlp_norm = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.mlp_norm_mask = InstanceNorm1d_Masked(hidden_dim, affine=True)

    def debugonly_extractref(self, x_padding, mask_padding, T=5):
        mask_padding = rearrange(mask_padding, 'b c0 (t ntoken) -> b c0 t ntoken', t=T)
        mask_padding = mask_padding[...,256:].clone()
        mask_padding = rearrange(mask_padding, 'b c0 t ntoken -> b c0 (t ntoken)', t=T)
        assert torch.all(mask_padding), mask_padding

        x_padding = rearrange(x_padding, 'b c (t ntoken) -> b c t ntoken', t=T)
        x_padding_true = x_padding[:,:,:,256:].clone()
        x_padding_true = rearrange(x_padding_true, 'b c t ntoken -> b c (t ntoken)', t=T)
        return x_padding_true, mask_padding

    def debugonly_assembleback(x_inp, x, T=5):
        x =  rearrange(x, 'b c (t len) -> b c t len', t=T)
        x_inp[:,:,:,256:]=x.clone()
        x = rearrange(x_inp,'b c t ntoken -> b c (t ntoken)')
        return x

    def forward_padding(self, x, mask_padding, bcs=None, leadtime=None, t_pos_area_padding=None):
        # x: b x c x token_len (len)
        # mask_padding: b x token_len
        # leadtime: b x c
        mask_padding = mask_padding.unsqueeze(1) # b x 1 x token_len
        B, _, len = x.shape
        input = x.clone()
        x = self.norm1_mask(x, mask_padding)
        x = rearrange(x, 'b c len -> (b len) c')
        mask_padding_BL = rearrange(mask_padding, 'b c1 len -> (b c1 len)')
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

        mask2d_padding = repeat(mask_padding, 'b c1 len -> b (len1 c1) len', len1=len).unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask2d_padding) 
        # Rearrange after attention
        x = rearrange(x, 'b  he len c -> b (he c) len')
        if torch.isnan(x).any():
            print("Debugging for NAN, mask2dpadding:", mask2d_padding[0,0,:,:], q, k , v,flush=True)
            for i in range(64):
                print(f"after att {i}", x[0,:,i], flush=True)
            sys.exit(-1)

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

    def forward(self, x, sequence_parallel_group=None, leadtime=None, input_control=None, bcs=None, mask_padding=None, t_pos_area=None, local_att=False):
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

        if local_att or sequence_parallel_group is None:
            #x = F.scaled_dot_product_attention(q, k, v)
            dtype = x.dtype
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16) 
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            x = flash_attn_func(q, k, v) #(b, len, he, c)
            x = x.transpose(1,2).to(dtype)
        else:
            # use ringX 
            dtype = x.dtype
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)  
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            loc_x = ringX_attn_func(q, k, v, causal=False, group=sequence_parallel_group) #all ddp ranks      
            x = loc_x
            x = x.transpose(1,2).to(dtype)

        # Rearrange after attention
        x = rearrange(x, 'b  he len c -> b (he c) len')
        #print(x.shape)
        x = self.norm2(x)
        x = rearrange(x, 'b c len -> (b len) c')
        x = self.output_head(x)
        x = rearrange(x, '(b len) c-> b c len', b=B)
        if leadtime is not None:
            x = x + leadtime[:,:, None]
        
        if input_control is not None:
            x = x + input_control[:,:, None]

        x = self.drop_path(x*self.gamma_att[None, :, None]) + input

        # MLP
        input = x.clone()
        x = rearrange(x, 'b c len-> b len c')
        x = self.mlp(x)
        x = rearrange(x, 'b len c -> b c len')
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None, :, None] * x)

        return output
