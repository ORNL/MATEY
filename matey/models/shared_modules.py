# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import torch.nn as nn
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class InstanceNorm1d_Masked(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, gamma=None, beta=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.gamma = None
        self.beta = None
        if self.affine:
            if gamma is None and beta is None:
                self.gamma = torch.nn.Parameter(torch.ones((self.num_features), requires_grad=True))
                self.beta = torch.nn.Parameter(torch.zeros((self.num_features), requires_grad=True))
            else:
                self.gamma = gamma
                self.beta = beta
    def forward(self, x, mask):
        B, C, L = x.shape
        assert C==self.num_features
        assert (B, 1, L) == mask.shape

        len_sample = torch.sum(mask, -1) # (B, 1)
        meanf = torch.sum(x * mask, -1)/ len_sample # (B,C)
        mean_reshape = meanf.unsqueeze(-1).expand_as(x)  # (B, C, L)
        
        var_term = ((x - mean_reshape) * mask)**2  # (B,C,L)
        varf = torch.sum(var_term, -1) / len_sample  #(B,C)
        var_reshape  = varf.unsqueeze(-1).expand_as(x)    # (B, C, L)

        # compute output
        x = rearrange(x, 'b c len -> (b len) c')
        mean_reshape = rearrange(mean_reshape, 'b c len -> (b len) c')
        var_reshape  = rearrange(var_reshape,  'b c len -> (b len) c')
        mask = rearrange(mask, 'b c1 len -> (b c1 len)') #c1=1

        x[mask] = (x[mask]  - mean_reshape[mask] ) / torch.sqrt(var_reshape[mask]  + self.eps)   # (BL, C)

        if self.affine:
            x[mask]  = x[mask]  * self.gamma[None, :] + self.beta[None, :]

        x = rearrange(x, '(b len) c -> b c len', b=B)

        return x
