import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

### Space utils
class RMSInstanceNorm2d(nn.Module):
    def __init__(self, dim, affine=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            #self.bias = nn.Parameter(torch.zeros(dim)) # Forgot to remove this so its in the pretrained weights
    
    def forward(self, x):
        std, mean = torch.std_mean(x, dim=(-2, -1), keepdims=True)
        x = (x) / (std + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None]  
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
            scale = (self.dim_in / label_size)**.5 # Equivalent to swapping init to correct for given subsample of input
            x = scale * F.linear(x, self.weight[:, labels], self.bias)
        else:
            x = F.linear(x, self.weight[labels], self.bias[labels])
        return x

def calc_ks4conv2d(patch_size=(16,16), nconv2d=3):
    #design three layers of Conv2d so that we get H/patch_size at output
    #8-->2^(1+1+1); 16-->2^(2+1+1); 32-->2^(3+1+1)
    #the original one is [4, 2, 2] for 16
    assert patch_size[0] == patch_size[1]
    nexp = int(math.log2(patch_size[0]))
    ks = [nexp//nconv2d]*nconv2d
    ks[0] = nexp - sum(ks[1:])
    return [2**item for item in ks]

class hMLP_stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=(16,16), in_chans=3, embed_dim=768, nconv2d=3):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.nconv2d = nconv2d
        self.ks = calc_ks4conv2d(patch_size=self.patch_size, nconv2d=self.nconv2d)
        modulelist = []
        for ilayer in range(self.nconv2d):
            in_chans_ilayer = in_chans if ilayer==0 else embed_dim//4
            embed_ilayer = embed_dim if ilayer==self.nconv2d-1 else embed_dim//4
            ks_ilayer = self.ks[ilayer]
            modulelist.append(nn.Conv2d(in_chans_ilayer, embed_ilayer, kernel_size=ks_ilayer, stride=ks_ilayer, bias=False))
            modulelist.append(RMSInstanceNorm2d(embed_ilayer, affine=True))
            modulelist.append(nn.GELU())
        self.in_proj = torch.nn.Sequential(*modulelist)
    
    def forward(self, x):
        x = self.in_proj(x)
        return x
       
class hMLP_output(nn.Module):
    """ Patch to Image De-bedding
    """
    def __init__(self, patch_size=(16,16), out_chans=3, embed_dim=768, nconv2d=3):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.nconv2d = nconv2d
        self.ks = calc_ks4conv2d(patch_size=self.patch_size, nconv2d=self.nconv2d)
        modulelist = []
        for ilayer in range(self.nconv2d-1):
            in_chans_ilayer = embed_dim if ilayer==0 else embed_dim//4
            embed_ilayer = embed_dim//4
            ks_ilayer = self.ks[-(ilayer+1)]
            modulelist.append(nn.ConvTranspose2d(in_chans_ilayer, embed_ilayer, kernel_size=ks_ilayer, stride=ks_ilayer, bias=False))
            modulelist.append(RMSInstanceNorm2d(embed_ilayer, affine=True))
            modulelist.append(nn.GELU())
        self.out_proj = torch.nn.Sequential(*modulelist)
        out_head = nn.ConvTranspose2d(embed_dim//4, out_chans, kernel_size=self.ks[0], stride=self.ks[0])
        self.out_stride = self.ks[0]
        self.out_kernel = nn.Parameter(out_head.weight)
        self.out_bias = nn.Parameter(out_head.bias)
    
    def forward(self, x, state_labels):
        x = self.out_proj(x)#.flatten(2).transpose(1, 2)
        x = F.conv_transpose2d(x, self.out_kernel[:, state_labels], self.out_bias[state_labels], stride=self.out_stride)
        return x
    