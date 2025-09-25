import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from operator import mul
from functools import reduce
from einops import rearrange, repeat
from ..utils.distributed_utils import closest_factors

### Space utils
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
        std, mean = torch.std_mean(x, dim=spatial_dims, keepdims=True)
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
            modulelist.append(RMSInstanceNormSpace(channels, affine=True))
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
            modulelist.append(RMSInstanceNormSpace(embed_ilayer, affine=True))
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
            modulelist.append(RMSInstanceNormSpace(embed_ilayer, affine=True))
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
