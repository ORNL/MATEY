# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

#SSIM metric taken from BLASTNET2.0: https://github.com/blastnet/blastnet2_sr_benchmark
#SSIM functions modified from https://github.com/jinh0park/pytorch-ssim-3D
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import sys
from math import exp
from ..utils.distributed_utils import assemble_samples, broadcast_scalar

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2


    C1 = (0.1)**2
    C2 = (0.3)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 9, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        if img1.dim() == 6:
            B, T, C, D, H, W = img1.shape
            img1 = img1.reshape(B * T, C, D, H, W)
            img2 = img2.reshape(B * T, C, D, H, W)

        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim3D(img1, img2, window_size = 9, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def remove_edges(arr):
    if arr.dim() == 6:
        return arr[:,:,:,1:-1,1:-1,1:-1] # B, T, C, D, H, W
    elif arr.dim() == 5:    
        return arr[:,:,1:-1,1:-1,1:-1] # B, C, D, H, W
    else:
        raise ValueError("Input tensor must be 5D or 6D.")
    
def get_unnormalized(pred, tar, sub_dataset, device, dtype=torch.float64):
    if not hasattr(sub_dataset, "get_normalization"):
        print(f"{sub_dataset.__class__.__name__} has no `get_normalization()` method. Returning pred and tar unchanged.")
        return pred, tar

    mean, std = sub_dataset.get_normalization(device=device, dtype=dtype)

    mean = torch.as_tensor(mean, device=device, dtype=dtype)
    mean = mean.view(1, 1, -1, 1, 1, 1)

    std = torch.as_tensor(std, device=device, dtype=dtype)
    std = std.view(1, 1, -1, 1, 1, 1)

    return pred * std + mean, tar * std + mean

def calculate_ssim3D(pred, target):
    ssim_module = SSIM3D()
    pred = remove_edges(pred)
    target = remove_edges(target)
    ssim_value = ssim_module(pred, target)
    
    return ssim_value

def get_ssim(tar,output,blockdict,global_rank,current_group,group_rank,group_size,device,dataset,dset_index):
    pred_all, tar_all = assemble_samples(tar, output, blockdict, global_rank, current_group, group_rank, group_size, device)
    if global_rank == 0:
        pred_all, tar_all = get_unnormalized(pred_all, tar_all, dataset.sub_dsets[dset_index[0]], device) # unnormalize to physical units/scale
        avg_ssim = calculate_ssim3D(pred_all, tar_all)
    else:
        avg_ssim = None
    avg_ssim = broadcast_scalar(avg_ssim, src=0, device=output.device)
    
    return avg_ssim