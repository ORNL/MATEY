# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.distributed as dist
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import subprocess
import json
import os


import torch.distributed.nn.functional as dist_nn_f
from torch import Tensor
from torch_geometric.data import Data
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, NamedTuple
from torch_geometric.loader import ClusterData
from torch_geometric.utils import to_undirected

def unwrap_leadtime_config(leadtime_config):
    leadtime_max=leadtime_config.get("leadtime_max", 1)
    leadtime_fixed=leadtime_config.get("leadtime_fixed", False)
    leadtime_returnfull=leadtime_config.get("leadtime_returnfull", False)
    return leadtime_max, leadtime_fixed, leadtime_returnfull

def extract_batch(data_iter, device=None):
    """return minibatch of data_iter"""
    try:
        data = next(data_iter) 
    except:
        print("In the exception...")
        return None
    if device:
        try:
            inp, dset_index, field_labels, bcs, tar, leadtime = map(lambda x: x.to(device), data)
            refineind = None
        except:
            inp, dset_index, field_labels, bcs, tar, refineind, leadtime = map(lambda x: x.to(device), data)
    else:
        try:
            inp, dset_index, field_labels, bcs, tar, leadtime =  data
            refineind = None
        except:
            inp, dset_index, field_labels, bcs, tar, refineind, leadtime = data
    return inp, dset_index, field_labels, bcs, tar, refineind, leadtime

def process_batch_data(inp, tar, refineind, hierarchical, params, datafilter_kernels=None):
    """prepare data for turbulence transformer"""
    if hierarchical:
        D, H, W = inp.shape[3:]
        assert refineind is None, "need to implement hierarchical for adaptive"
        filedata_mods, blockdict_mods = multimods_turbulencetransformer(inp, tar, datafilter_kernels, params.hierarchical)
    else:
        filedata_mods = [(inp, tar)]
        blockdict_mods = [None]

    return filedata_mods, blockdict_mods

def extract_data_forsequenceparallel(data_iter, hierarchical, params, datafilter_kernels, group_rank, current_group, device):
    if group_rank == 0:
        batch = extract_batch(data_iter)  
        if batch is None:
            return None
        inp, dset_index, field_labels, bcs, tar, refineind, leadtime = batch
        inp = rearrange(inp, 'b t c d h w -> t b c d h w')
        if hierarchical:
            filedata_mods, blockdict_mods = multimods_turbulencetransformer(inp, tar, datafilter_kernels, params.hierarchical)
        else:
            filedata_mods = [(inp, tar)]
            blockdict_mods = [None]
        broadcast_list=[filedata_mods, blockdict_mods, dset_index.to(device), field_labels.to(device), bcs.to(device), refineind, leadtime.to(device)]
    else:
        broadcast_list=[None, None, None, None, None, None, None]
    global_src = dist.get_global_rank(current_group, 0)
    dist.broadcast_object_list(broadcast_list, src=global_src, group=current_group)
    filedata_mods, blockdict_mods, dset_index, field_labels, bcs, refineind, leadtime = broadcast_list
    if hierarchical:
        assert refineind is None, "need to implement hierarchical for adaptive"
    return filedata_mods, blockdict_mods, dset_index, field_labels, bcs, refineind, leadtime 

def construct_filterkernels(filtersize):
    datafilter_kernels=[]
    for kernel_size in filtersize:
        center = kernel_size // 2
        x, y, z = np.indices((kernel_size, kernel_size, kernel_size))
        dist2 = (x - center)**2 + (y - center)**2 + (z - center)**2
        kernel = np.exp(-dist2/2.0)
        kernel /= np.sum(kernel)
        gaussian_kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        datafilter_kernels.append(gaussian_kernel)
    return datafilter_kernels

def construct_filterkernel(kernel_size):
    with torch.no_grad():
        center = kernel_size // 2
        x, y, z = np.indices((kernel_size, kernel_size, kernel_size))
        dist2 = (x - center)**2 + (y - center)**2 + (z - center)**2
        kernel = np.exp(-dist2/2.0)
        kernel /= np.sum(kernel)
        gaussian_kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return gaussian_kernel

def construct_filterkernel2D(kernel_size):
    with torch.no_grad():
        center = kernel_size // 2
        x, y = np.indices((kernel_size, kernel_size))
        dist2 = (x - center)**2 + (y - center)**2
        kernel = np.exp(-dist2/2.0)
        kernel /= np.sum(kernel)
        gaussian_kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return gaussian_kernel

def construct_multimods_v2(datax, datay, datafilter_kernels, hierarchical_parameters):
    #T,B,C,D,H,W
    T,B,C,D,H,W = datax.shape
    assert (B,C,D,H,W) == datay.shape

    filtersize = hierarchical_parameters["filtersize"]
    if "cubsize" in hierarchical_parameters:
        cubsize = [[_cubsize, _cubsize, _cubsize] for _cubsize in hierarchical_parameters["cubsize"]]
    else:
        cubsize = [[D//sizeRT, H //sizeRT, W //sizeRT] for sizeRT in hierarchical_parameters["cubsizeRT"]]

    assert cubsize[0]==[D, H, W], f"largest cubsize should be domain size, {cubsize[0]}, {[D, H, W]}"

    datax = rearrange(datax, 't b c d h w -> (t b c) d h w')
    datay = rearrange(datay, 'b c d h w -> (b c) d h w')
    filedata_mods=[]
    blockdict_mods=[]
    # Apply the filter
    rank = dist.get_rank()
    for imod, (kernel, kernel_size, cropsize) in enumerate(zip(datafilter_kernels, filtersize, cubsize)): 
        """
        hierarchical:
        filtersize: [8, 4, 1]
        cubsizeRT: [1, 2, 8] #ratio to the loaded data size, e.g., 256, would be 256/[1,2,4] = [256, 128, 64]
        """
        ###filter data###
        filteredx = F.conv3d(datax[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
        filteredx = rearrange(filteredx, '(t b c) c1 d h w -> t b (c c1) d h w', t=T, b=B) #c1=1
        filteredy = F.conv3d(datay[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
        filteredy = rearrange(filteredy, '(b c) c1 d h w -> b (c c1) d h w', b=B) #c1=1
        #print(f"Pei check shape, imod {imod}, rank {rank}, {filteredx.shape}, {filteredy.shape}, {kernel_size}, {cropsize}, {datax.shape}, {datay.shape}", flush=True)
        ###crop data###
        id=torch.randint(D//kernel_size-cropsize[0]//kernel_size+1, (1,))
        ih=torch.randint(H//kernel_size-cropsize[1]//kernel_size+1, (1,))
        iw=torch.randint(W//kernel_size-cropsize[2]//kernel_size+1, (1,))
        id_end = id+cropsize[0]//kernel_size
        ih_end = ih+cropsize[1]//kernel_size
        iw_end = iw+cropsize[2]//kernel_size
        filteredx = filteredx[...,id:id_end,ih:ih_end,iw:iw_end]
        filteredy = filteredy[...,id:id_end,ih:ih_end,iw:iw_end]
        filedata_mods.append((filteredx, filteredy))
        print(f"Pei check shape, imod {imod}, rank {rank}, {[id, id_end,ih,ih_end,iw, iw_end]}", flush=True)
        ###
        blockdict={}
        blockdict["Lzxy"] = [float(cropsize[0])/D, float(cropsize[1])/H, float(cropsize[2])/W]
        blockdict["zxy_start"] = [1.0/D*(id*kernel_size), 1.0/H*(ih*kernel_size), 1.0/W*(iw*kernel_size)]
        blockdict["Ind_start"]=[id, ih, iw]
        blockdict["Ind_end"]  =[id_end, ih_end, iw_end]
        blockdict["Ind_dim"]  =[D//kernel_size, H//kernel_size, W//kernel_size]
        blockdict_mods.append(blockdict)
        print(f"Pei construct_multimods, imod {imod}, rank {rank}, {filteredx.shape}, {filteredy.shape}, {kernel_size}, {cropsize}, {blockdict}", flush=True)

    return filedata_mods, blockdict_mods


def construct_multimods_v3(datax, datay, datafilter_kernels, hierarchical_parameters, stride=[1, 1, 1], blockdict=None):
    #B,T,C,D,H,W
    B,T,C,D,H,W = datax.shape
    assert (B,C,D,H,W) == datay.shape, f"{datax.shape}, {datay.shape}"

    filtersize = hierarchical_parameters["filtersize"]
    if "cubsize" in hierarchical_parameters:
        cubsize = [[_cubsize, _cubsize, _cubsize] for _cubsize in hierarchical_parameters["cubsize"]]
    else:
        cubsize = [[D//sizeRT, H //sizeRT, W //sizeRT] for sizeRT in hierarchical_parameters["cubsizeRT"]]

    assert cubsize[0]==[D, H, W], f"largest cubsize should be domain size, {cubsize[0]}, {[D, H, W]}"

    datax = rearrange(datax, 'b t c d h w -> (b t c) d h w')
    datay = rearrange(datay, 'b c d h w -> (b c) d h w')
    filedata_mods=[]
    blockdict_mods=[]
    # Apply the filter
    for imod, (kernel, kernel_size, cropsize) in enumerate(zip(datafilter_kernels, filtersize, cubsize)): 
        """
        hierarchical:
        filtersize: [8, 4, 1]
        cubsizeRT: [1, 2, 8] #ratio to the loaded data size, e.g., 256, would be 256/[1,2,4] = [256, 128, 64]
        """
        ###filter data###
        filteredx = F.conv3d(datax[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
        filteredx = rearrange(filteredx, '(b t c) c1 d h w -> b t (c c1) d h w', t=T, b=B) #c1=1
        filteredy = F.conv3d(datay[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
        filteredy = rearrange(filteredy, '(b c) c1 d h w -> b (c c1) d h w', b=B) #c1=1
        ###crop data###
        ind_d=torch.arange(start=0, end=D//kernel_size-cropsize[0]//kernel_size+1, step=stride[0])
        ind_h=torch.arange(start=0, end=H//kernel_size-cropsize[1]//kernel_size+1, step=stride[1])
        ind_w=torch.arange(start=0, end=W//kernel_size-cropsize[2]//kernel_size+1, step=stride[2])
        id=ind_d[torch.randint(len(ind_d), (1,))]
        ih=ind_h[torch.randint(len(ind_h), (1,))]
        iw=ind_w[torch.randint(len(ind_w), (1,))]

        id_end = id + cropsize[0]//kernel_size
        ih_end = ih + cropsize[1]//kernel_size
        iw_end = iw + cropsize[2]//kernel_size
 
        filteredx = filteredx[...,id:id_end,ih:ih_end,iw:iw_end]
        filteredy = filteredy[...,id:id_end,ih:ih_end,iw:iw_end]
        filedata_mods.append((filteredx, filteredy))
        ###
        if blockdict is not None:
            blockdict_mod=copy.deepcopy(blockdict) 
            #e.g.,{'Lzxy': [0.25, 0.25, 0.5], 'nproc_blocks': [4, 4, 2], 
            #'Ind_dim': [256, 256, 512], 'Ind_start': [tensor(256), tensor(768), tensor(512)], 
            #'zxy_start': [tensor(0.2500), tensor(0.7500), tensor(0.5000)]}
            assert [D,H,W] == blockdict["Ind_dim"], f"(D,H,W),{(D,H,W)}, {blockdict['Ind_dim']}"
            Lz, Lx, Ly = blockdict["Lzxy"]
            Lz_start, Lx_start, Ly_start = blockdict["zxy_start"]
        else:
            #no split
            Lz, Lx, Ly = 1.0, 1.0, 1.0
            Lz_start, Lx_start, Ly_start = 0.0, 0.0, 0.0
            blockdict_mod = {}
        ########
        #Ind variables are for each local split
        blockdict_mod["Ind_start_loc"]=[id, ih, iw] #local mode start
        blockdict_mod["Ind_end_loc"]  =[id_end, ih_end, iw_end] #local mode end
        blockdict_mod["Ind_dim"] = [D//kernel_size, H//kernel_size, W//kernel_size] #total mode size 
        #Absolute location and lengths, assuming domain starts at (0,0,0) and ends at (1,1,1)
        blockdict_mod["zxy_start"] = [Lz_start+float(id*kernel_size)/D*Lz, Lx_start+float(ih*kernel_size)/H*Lx, Ly_start+float(iw*kernel_size)/W*Ly]
        blockdict_mod["Lzxy"] = [float(cropsize[0])/D*Lz, float(cropsize[1])/H*Lx, float(cropsize[2])/W*Ly]
        ##########    
        blockdict_mods.append(blockdict_mod)
        #print(f"Pei construct_multimods, imod {imod}, rank {dist.get_rank()}, {filteredx.shape}, {filteredy.shape}, {kernel_size}, {cropsize}, {blockdict_mod}", flush=True)
    return filedata_mods, blockdict_mods

def filter_data(data, kernel, kernel_size):
    if data.ndim==5:
        B,C,D,H,W = data.shape
        data = rearrange(data, 'b c d h w -> (b c) d h w')
        filtered = F.conv3d(data[:,None,:,:,:], kernel.to(data.device), stride=kernel_size)
        filtered = rearrange(filtered, '(b c) c1 d h w -> b (c c1) d h w', b=B) #c1=1   
    elif data.ndim==6: 
        B,T,C,D,H,W = data.shape
        data = rearrange(data, 'b t c d h w -> (b t c) d h w')
        # Apply the filter
        filtered = F.conv3d(data[:,None,:,:,:], kernel.to(data.device), stride=kernel_size)
        filtered = rearrange(filtered, '(b t c) c1 d h w -> b t (c c1) d h w', t=T, b=B) #c1=1
    else:
        raise ValueError(f"unkown tensor shape in filter_data, {data.shape}")    
    return filtered

def construct_multimods_MG(datax, datay, datafilter_kernels, hierarchical_parameters, stride=[1, 1, 1], blockdict=None):
    #B,T,C,D,H,W
    B,T,C,D,H,W = datax.shape
    assert (B,C,D,H,W) == datay.shape, f"{datax.shape}, {datay.shape}"

    filtersize = hierarchical_parameters["filtersize"]
    if "cubsize" in hierarchical_parameters:
        cubsize = [[_cubsize, _cubsize, _cubsize] for _cubsize in hierarchical_parameters["cubsize"]]
    else:
        cubsize = [[D//sizeRT, H //sizeRT, W //sizeRT] for sizeRT in hierarchical_parameters["cubsizeRT"]]

    for imod in range(len(filtersize)):
        assert cubsize[imod]==[D, H, W], f"In MG, mode cubsize should be domain size, {cubsize[imod]}, {[D, H, W]}"

    datax = rearrange(datax, 'b t c d h w -> (b t c) d h w')
    datay = rearrange(datay, 'b c d h w -> (b c) d h w')
    filedata_mods=[]
    blockdict_mods=[]
    # Apply the filter
    for imod, (kernel, kernel_size, cropsize) in enumerate(zip(datafilter_kernels, filtersize, cubsize)): 
        """
        hierarchical:
        filtersize: [8, 4, 1]
        cubsizeRT: [1, 2, 8] #ratio to the loaded data size, e.g., 256, would be 256/[1,2,4] = [256, 128, 64]
        """
        ###filter data###
        filteredx = F.conv3d(datax[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
        filteredx = rearrange(filteredx, '(b t c) c1 d h w -> b t (c c1) d h w', t=T, b=B) #c1=1
        filteredy = F.conv3d(datay[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
        filteredy = rearrange(filteredy, '(b c) c1 d h w -> b (c c1) d h w', b=B) #c1=1       
        filedata_mods.append((filteredx, filteredy))
        ###
        if blockdict is not None:
            blockdict_mod=copy.deepcopy(blockdict) 
            #e.g.,{'Lzxy': [0.25, 0.25, 0.5], 'nproc_blocks': [4, 4, 2], 
            #'Ind_dim': [256, 256, 512], 'Ind_start': [tensor(256), tensor(768), tensor(512)], 
            #'zxy_start': [tensor(0.2500), tensor(0.7500), tensor(0.5000)]}
            assert [D,H,W] == blockdict["Ind_dim"], f"(D,H,W),{(D,H,W)}, {blockdict['Ind_dim']}"
            Lz, Lx, Ly = blockdict["Lzxy"]
            Lz_start, Lx_start, Ly_start = blockdict["zxy_start"]
        else:
            #no split
            Lz, Lx, Ly = 1.0, 1.0, 1.0
            Lz_start, Lx_start, Ly_start = 0.0, 0.0, 0.0
            blockdict_mod = {}
        ########
        #Ind variables are for each local split
        blockdict_mod["Ind_start_loc"]=[0, 0, 0] #local mode start
        blockdict_mod["Ind_end_loc"]  =[D//kernel_size, H//kernel_size, W//kernel_size] #local mode end
        blockdict_mod["Ind_dim"] = [D//kernel_size, H//kernel_size, W//kernel_size] #total mode size 
        #Absolute location and lengths, assuming domain starts at (0,0,0) and ends at (1,1,1)
        blockdict_mod["zxy_start"] = [Lz_start, Lx_start, Ly_start]
        blockdict_mod["Lzxy"] = [float(cropsize[0])/D*Lz, float(cropsize[1])/H*Lx, float(cropsize[2])/W*Ly]
        ##########    
        blockdict_mods.append(blockdict_mod)
        #print(f"Pei construct_multimods, imod {imod}, rank {dist.get_rank()}, {filteredx.shape}, {filteredy.shape}, {kernel_size}, {cropsize}, {blockdict_mod}", flush=True)
    return filedata_mods, blockdict_mods

def construct_multimods(datax, datay, datafilter_kernels, hierarchical_parameters):
    #T,B,C,D,H,W
    T,B,C,D,H,W = datax.shape
    assert (B,C,D,H,W) == datay.shape

    filtersize = hierarchical_parameters["filtersize"]
    if "cubsize" in hierarchical_parameters:
        cubsize = [[_cubsize, _cubsize, _cubsize] for _cubsize in hierarchical_parameters["cubsize"]]
    else:
        cubsize = [[D//sizeRT, H //sizeRT, W //sizeRT] for sizeRT in hierarchical_parameters["cubsizeRT"]]
    datax = rearrange(datax, 't b c d h w -> (t b c) d h w')
    datay = rearrange(datay, 'b c d h w -> (b c) d h w')
    filedata_mods=[]
    blockdict_mods=[]
    # Apply the filter
    rank = dist.get_rank()
    for kernel, kernel_size, cropsize in zip(datafilter_kernels, filtersize, cubsize): 
        ###crop data###
        id=torch.randint(D-cropsize[0]+1, (1,))
        ih=torch.randint(H-cropsize[1]+1, (1,))
        iw=torch.randint(W-cropsize[2]+1, (1,))
        data_cropx = datax[:,id:id+cropsize[0],ih:ih+cropsize[1],iw:iw+cropsize[2]]
        data_cropy = datay[:,id:id+cropsize[0],ih:ih+cropsize[1],iw:iw+cropsize[2]]
        ###filter data###
        filteredx = F.conv3d(data_cropx[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
        filteredx = rearrange(filteredx, '(t b c) c1 d h w -> t b (c c1) d h w', t=T, b=B) #c1=1
        filteredy = F.conv3d(data_cropy[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
        filteredy = rearrange(filteredy, '(b c) c1 d h w -> b (c c1) d h w', b=B) #c1=1
        filedata_mods.append((filteredx, filteredy))
        ###
        blockdict={}
        blockdict["Lzxy"] = [float(cropsize[0])/D, float(cropsize[1])/H, float(cropsize[2])/W]
        blockdict["zxy_start"] = [1.0/D*id, 1.0/H*ih, 1.0/W*iw]
        blockdict_mods.append(blockdict)
        print(f"Pei construct_multimods rank {rank},{kernel_size}, {cropsize}, {blockdict} ", flush=True)

    return filedata_mods, blockdict_mods

def construct_finemods_decomp(datax, datay, datafilter_kernels, hierarchical_parameters, iblock=0):
    #return the decomposition of finest modes
    #T,B,C,D,H,W
    T,B,C,D,H,W = datax.shape
    assert (B,C,D,H,W) == datay.shape

    filtersize = hierarchical_parameters["filtersize"]
    if "cubsize" in hierarchical_parameters:
        cubsize = [[_cubsize, _cubsize, _cubsize] for _cubsize in hierarchical_parameters["cubsize"]]
    else:
        cubsize = [[D//sizeRT, H //sizeRT, W //sizeRT] for sizeRT in hierarchical_parameters["cubsizeRT"]]
    datax = rearrange(datax, 't b c d h w -> (t b c) d h w')
    datay = rearrange(datay, 'b c d h w -> (b c) d h w')
    filedata_mods=[]
    blockdict_mods=[]
    # Apply the filter
    rank = dist.get_rank()
    kernel = datafilter_kernels[-1]
    kernel_size = filtersize[-1]
    cropsize = cubsize[-1]
    icount=-1
    for id in range(0, D, cropsize[0]):
        for ih in range(0, H, cropsize[1]):
            for iw in range(0, W, cropsize[2]):
                icount += 1
                if icount != iblock:
                    continue
                ###crop data###
                data_cropx = datax[:,id:id+cropsize[0],ih:ih+cropsize[1],iw:iw+cropsize[2]]
                data_cropy = datay[:,id:id+cropsize[0],ih:ih+cropsize[1],iw:iw+cropsize[2]]
                ###filter data###
                filteredx = F.conv3d(data_cropx[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
                filteredx = rearrange(filteredx, '(t b c) c1 d h w -> t b (c c1) d h w', t=T, b=B) #c1=1
                filteredy = F.conv3d(data_cropy[:,None,:,:,:], kernel.to(datax.device), stride=kernel_size)
                filteredy = rearrange(filteredy, '(b c) c1 d h w -> b (c c1) d h w', b=B) #c1=1
                filedata_mods.append((filteredx, filteredy))
                ###
                blockdict={}
                blockdict["Lzxy"] = [float(cropsize[0])/D, float(cropsize[1])/H, float(cropsize[2])/W]
                blockdict["zxy_start"] = [1.0/D*id, 1.0/H*ih, 1.0/W*iw]
                blockdict_mods.append(blockdict)
                print(f"Pei construct_finemods_decomp rank {rank},{kernel_size}, {cropsize}, {blockdict} ", flush=True)
    return filedata_mods, blockdict_mods

def multimods_turbulencetransformer(x, y, datafilter_kernels, hierarchical_parameters, return_decomp_finest=None):
    """
    split a sample based on sequence split groups
    """
    if return_decomp_finest is not None:
        filedata_mods, blockdict_mods=construct_finemods_decomp(x, y, datafilter_kernels, hierarchical_parameters, iblock=return_decomp_finest)
    else:
        #filedata_mods, blockdict_mods=construct_multimods(x, y, datafilter_kernels, hierarchical_parameters)
        filedata_mods, blockdict_mods=construct_multimods_v2(x, y, datafilter_kernels, hierarchical_parameters)
    #figure_checking(x, y, filedata_mods); exit(0)
    del x,y
    ##############################################################
    return filedata_mods, blockdict_mods

def generate_grid(space_dims):
    [D,H,W]=space_dims
    z_min = 0.0; z_max = 1.0
    y_min = 0.0; y_max = 1.0
    x_min = 0.0; x_max = 1.0
    z_new = torch.linspace(z_min, z_max, D)
    x_new = torch.linspace(x_min, x_max, H)
    y_new = torch.linspace(y_min, y_max, W)
    zv, xv, yv = torch.meshgrid(z_new,  x_new, y_new, indexing='ij')
    grid3d = torch.stack((yv, xv, zv), dim=-1) #Note: the order is to match the convention in grid_sample
    return grid3d

def mods_assemble(data_mod, blockdict, grid3d):
    #input: data_mod - tensors contain cut scales, with spatial info saved in blockdict
    #return: mapped data to mesh grid3d 
    B = data_mod.shape[0]
    Lz = blockdict["Lzxy"][0]
    Lx = blockdict["Lzxy"][1]
    Ly = blockdict["Lzxy"][2]
    Lz_start = blockdict["zxy_start"][0].item()
    Lx_start = blockdict["zxy_start"][1].item()
    Ly_start = blockdict["zxy_start"][2].item()
    grid3d_norm = grid3d.clone()
    #so that -1 and 1 correspond to the edge of data_mod
    #Note: y, x, z in grid3d: 0,1,2
    grid3d_norm[...,0] = 2.0 * (grid3d[...,0] - Ly_start) / Ly - 1.0
    grid3d_norm[...,1] = 2.0 * (grid3d[...,1] - Lx_start) / Lx - 1.0
    grid3d_norm[...,2] = 2.0 * (grid3d[...,2] - Lz_start) / Lz - 1.0 
    #D,H,W,3 --> 1,D,H,W,3 --> B,D,H,W,3
    grid3d_norm = grid3d_norm.unsqueeze(0).repeat(B, 1, 1, 1, 1) # B,D,H,W,3  
    #print("Pei debug z", grid3d_norm[...,0].min(), grid3d_norm[...,0].max())
    #print("Pei debug x", grid3d_norm[...,1].min(), grid3d_norm[...,1].max())
    #print("Pei debug y", grid3d_norm[...,2].min(), grid3d_norm[...,2].max())
    data = F.grid_sample(data_mod, grid3d_norm, mode='bilinear', padding_mode='zeros', align_corners=True)
    #figure_checking_assemble(data_mod, data, plotdir="./imgs/"+f"Lz_{Lz}_Lx_{Lx}_Ly_{Ly}_Lz0_{Lz_start}_Lx0_{Lx_start}_Ly0_{Ly_start}")
    return data
"""
def mods_assemble(data_mod, grid3d):
    #input: data_mod - tensors contain cut scales, with spatial info saved in blockdict
    #return: mapped data to mesh grid3d
    B = data_mod.shape[0]
    Lz=1.0; Lx=1.0; Ly=1.0
    Lz_start=0.0; Lx_start=0.0; Ly_start=0.0
    grid3d_norm = grid3d.clone()
    #so that -1 and 1 correspond to the edge of data_mod
    #Note: y, x, z in grid3d: 0,1,2
    grid3d_norm[...,0] = 2.0 * (grid3d[...,0] - Ly_start) / Ly - 1.0
    grid3d_norm[...,1] = 2.0 * (grid3d[...,1] - Lx_start) / Lx - 1.0
    grid3d_norm[...,2] = 2.0 * (grid3d[...,2] - Lz_start) / Lz - 1.0
    #D,H,W,3 --> 1,D,H,W,3 --> B,D,H,W,3
    grid3d_norm = grid3d_norm.unsqueeze(0).repeat(B, 1, 1, 1, 1) # B,D,H,W,3
    #print("Pei debug z", grid3d_norm[...,0].min(), grid3d_norm[...,0].max())
    #print("Pei debug x", grid3d_norm[...,1].min(), grid3d_norm[...,1].max())
    #print("Pei debug y", grid3d_norm[...,2].min(), grid3d_norm[...,2].max())
    data = F.grid_sample(data_mod, grid3d_norm, mode='bilinear', padding_mode='zeros', align_corners=True)
    #figure_checking_assemble(data_mod, data, plotdir="./imgs/"+f"Lz_{Lz}_Lx_{Lx}_Ly_{Ly}_Lz0_{Lz_start}_Lx0_{Lx_start}_Ly0_{Ly_start}")
    return data
"""
def figure_checking(datax, datay, filedata_mods):
    T,B,C,D,H,W = datax.shape
    varnames = ['Vx', 'Vy', 'Vw', 'pressure']
    casesets=["original","crop","filter1","filter2"]
    for ib in range(B):
        fig, axs = plt.subplots(4,4, figsize=(20, 20))
        for irow in range(4):
            if irow==0:
                data = datax[1, ib,:,:,:,:]
            else:
                data = filedata_mods[irow-1][0][1, ib,:,:,:,:]
            C,D,H,W = data.shape
            plot_contour(axs[irow,:], data[:,D//2,:,:], varnames, casesets[irow])
        fig.tight_layout()
        plt.savefig(f"check_croppingfiltering_sampleID{ib}_x1.png")
        plt.close()
    for ib in range(B):
        fig, axs = plt.subplots(4,4, figsize=(20, 20))
        for irow in range(4):
            if irow==0:
                data = datay[ib,:,4,:,:]
            else:
                data = filedata_mods[irow-1][1][ib,:,4,:,:]
            plot_contour(axs[irow,:], data, varnames, casesets[irow])
        fig.tight_layout()
        plt.savefig(f"check_croppingfiltering_sampleID{ib}_y.png")
        plt.close()

def plot_contour(axs, data, varnames, caseset, nvar=4):
    for ivar in range(nvar):
        icol = ivar
        ax = axs[icol]
        cs = ax.contourf(data[ivar,:,:].squeeze().cpu().detach().numpy().transpose(), cmap="jet", levels=50)
        ax.set_title(varnames[ivar]+"; "+caseset)         
        ax.set_aspect('equal')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cs, cax=cax, orientation='vertical')
    return cs

def decompress_zstd(file_path):
        """
        Use system 'zstd' to decompress the leaf chunk. No extra Python packages needed.
        """
        # Decompress to stdout and capture in Python
        out = subprocess.run(["zstd", "-d", "-c", file_path], check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        return out
def locate_leaf_chunk_file(chunks_dir, timestep):
    """
    For chunk index [timestep, 0, 0, 0, 0], descend under c/ until we reach the leaf file.
    Typical path: c/<t>/0/0/0/0  (last '0' is the leaf file with no extension).
    The current implementation assumes the following:
    - 4 levels of directories under c/ (for the 4 dimensions other than time: D, H, W, C).
    - All non-time dimensions are chunked with chunk size 1, leading to directories named '0' at each level.
    - The leaf chunk file is named '0' with no extension.
    """
    path = os.path.join(chunks_dir, str(timestep))
    for _ in range(4): 
        next_path = os.path.join(path, "0")
        if os.path.isdir(next_path):
            path = next_path
        else:
            if os.path.isfile(next_path):
                return next_path
            if os.path.isfile(path):
                return path
            raise FileNotFoundError(f"Leaf chunk file not found under: {os.path.join(chunks_dir, str(timestep))}")
    leaf = os.path.join(path, "0")
    if os.path.isfile(leaf):
        return leaf
    raise FileNotFoundError(f"Expected leaf file at: {leaf}")

def load_zarr_metadata(path):
    """
    Load Zarr v3 metadata from zarr.json.
    We assume the metadata is stored in a file named 'zarr.json' at the given path, which is typical for Zarr v3 datasets.
    """
    with open(os.path.join(path, "zarr.json"), "r") as f:
        return json.load(f)

def list_timestep_indices(chunks_dir):
    """List available timestep indices from chunk directories."""
    indices = []
    for item in os.listdir(chunks_dir):
        if os.path.isdir(os.path.join(chunks_dir, item)):
            try:
                idx = int(item)
                indices.append(idx)
            except ValueError:
                continue
    return sorted(indices)
    
# ──────────────────────────────────────────────────────────────────────────────
# Graph partitioning helpers
# ──────────────────────────────────────────────────────────────────────────────

class GhostInfo(NamedTuple):
    """
    Everything a rank needs to perform a halo exchange for one graph snapshot.

    owned_mask      : bool [N_local]  – True  → owned node
    ghost_rank      : int  [G]        – which rank owns each ghost node
    ghost_remote_idx: int  [G]        – index of the ghost inside that rank's
                                         local owned node list
    local_ghost_idx : int  [G]        – position of each ghost in this rank's
                                         local node tensor
    send_rank       : int  [S]        – ranks that need a slice of our embeddings
    send_local_idx  : list of int[]   – for each send_rank, the local indices to
                                         send (these are owned nodes that are
                                         ghosts on another rank)
    recv_counts     : dict rank→int   – how many ghost scalars to recv per rank
                                         (needed when F varies at runtime)
    """
    #owned_mask       : Tensor               # [N_local] bool
    ghost_rank       : Tensor               # [G] long
    ghost_remote_idx : Tensor               # [G] long
    local_ghost_idx  : Tensor               # [G] long
    send_rank        : List[int]
    send_local_idx   : List[Tensor]         # one tensor per send rank
    recv_counts      : Dict[int, int]       # rank → num ghost nodes from that rank

@torch.no_grad()
def check_metis_graph(data, case_name):
    edge_index = data.edge_index.detach().cpu().long().contiguous()
    src, dst = edge_index
    num_nodes = int(data.num_nodes)
    num_edges = int(edge_index.size(1))

    print(
        f"[METIS check] case={case_name}, "
        f"num_nodes={num_nodes}, num_edges={num_edges}",
        flush=True,
    )

    # -------------------------------------------------------------
    # 1. Index range
    # -------------------------------------------------------------
    min_index = int(edge_index.min())
    max_index = int(edge_index.max())

    if min_index < 0 or max_index >= num_nodes:
        raise RuntimeError(
            f"{case_name}: invalid node indices: "
            f"min={min_index}, max={max_index}, "
            f"num_nodes={num_nodes}"
        )

    # -------------------------------------------------------------
    # 2. Self-loops
    # -------------------------------------------------------------
    num_self_loops = int((src == dst).sum())

    # -------------------------------------------------------------
    # 3. Exact symmetry, including duplicate multiplicity
    #
    # Encode (src, dst) as one int64 value and compare it against
    # the encoded reversed edges (dst, src).
    # -------------------------------------------------------------
    edge_key = src * num_nodes + dst
    reverse_key = dst * num_nodes + src

    edge_key = torch.sort(edge_key).values
    reverse_key = torch.sort(reverse_key).values

    asymmetric_mask = edge_key != reverse_key
    num_asymmetric = int(asymmetric_mask.sum())

    # -------------------------------------------------------------
    # 4. Duplicate adjacency entries
    # -------------------------------------------------------------
    num_duplicates = int(
        (edge_key[1:] == edge_key[:-1]).sum()
    )

    # -------------------------------------------------------------
    # 5. Degree information
    # -------------------------------------------------------------
    out_degree = torch.bincount(src, minlength=num_nodes)
    in_degree = torch.bincount(dst, minlength=num_nodes)

    num_zero_out_degree = int((out_degree == 0).sum())
    num_zero_in_degree = int((in_degree == 0).sum())
    degree_mismatch_nodes = int((out_degree != in_degree).sum())

    print(
        f"[METIS check] case={case_name}: "
        f"index_range=[{min_index}, {max_index}], "
        f"self_loops={num_self_loops}, "
        f"duplicates={num_duplicates}, "
        f"asymmetric_entries={num_asymmetric}, "
        f"degree_mismatch_nodes={degree_mismatch_nodes}, "
        f"zero_out_degree={num_zero_out_degree}, "
        f"zero_in_degree={num_zero_in_degree}, "
        f"max_out_degree={int(out_degree.max())}",
        flush=True,
    )

    if num_asymmetric != 0:
        raise RuntimeError(
            f"{case_name}: graph is not exactly undirected; "
            f"found {num_asymmetric} asymmetric adjacency entries"
        )



def _partition_metis(edge_index, num_nodes, num_parts):
    """
    Partition a graph with PyG's ClusterData (METIS under the hood).
    Returns node-to-partition assignment [num_nodes] long tensor (0 … num_parts-1).
    """
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    dummy = Data(edge_index=edge_index.long(), num_nodes=num_nodes)
    check_metis_graph(dummy, "debugging")

    #print("Pei debugging [input]", dummy, num_parts, flush=True)
    cluster_data = ClusterData(dummy, num_parts=num_parts, log=True)

    # ClusterData stores node_perm (sorted by partition) and partptr (partition boundaries)
    # reconstruct assignment[original_node] = partition_id
    node_perm = cluster_data.partition.node_perm # [N] permuted node indices
    partptr   = cluster_data.partition.partptr   # [num_parts+1] partition boundaries

    assignment = torch.empty(num_nodes, dtype=torch.long)
    for part_id in range(num_parts):
        start = int(partptr[part_id])
        end   = int(partptr[part_id + 1])
        assignment[node_perm[start:end]] = part_id

    #print("Pei debugging [partition successful]", len(node_perm), len(partptr), num_parts, flush=True)

    return assignment
def _partition_random(num_nodes, num_parts, seed= 2024):
    rng = np.random.default_rng(seed)
    assignment = np.arange(num_nodes) % num_parts
    rng.shuffle(assignment)
    return torch.from_numpy(assignment).long()

def partition_graph(edge_index, num_nodes, num_parts, method= "metis"):
    """Return node-to-part assignment tensor [num_nodes]."""
    if method == "metis":
        return _partition_metis(edge_index, num_nodes, num_parts)
    elif method == "random":
        return _partition_random(num_nodes, num_parts)
    else:
        raise ValueError(f"Unknown partition method: {method!r}. Choose 'metis' or 'random'.")

def build_local_subgraph(data, node_assignment, group_rank, num_parts):
    """
    Extract the local subgraph for *rank* from a full graph, and build the
    GhostInfo descriptor needed for halo exchange.

    Parameters
    ----------
    data            : full PyG Data  (x, pos, edge_index, edge_attr)
    node_assignment : [N] long, partition assignment per node
    group_rank      : int, this data rank's id
    num_parts       : group_size

    Returns
    -------
    local_data  : PyG Data over owned + ghost nodes
                  node ordering: [owned_0 … owned_K | ghost_0 … ghost_G]
    ghost_info  : GhostInfo namedtuple
    """
    N           = data.num_nodes
    edge_index  = data.edge_index    #[2, E]
    src_nodes   = edge_index[0]
    dst_nodes   = edge_index[1]

    owned_global = torch.where(node_assignment == group_rank)[0]  #[K]
    owned_set    = set(owned_global.tolist())

    # ── find ghost nodes ──────────────────────────────────────────────────────
    # A ghost is a neighbour (via any edge endpoint) of an owned node that
    # belongs to a different rank.
    ghost_global_set = set()
    for e in range(edge_index.shape[1]):
        s, d = int(src_nodes[e]), int(dst_nodes[e])
        if s in owned_set and d not in owned_set:
            ghost_global_set.add(d)
        elif d in owned_set and s not in owned_set:
            ghost_global_set.add(s)

    ghost_global = torch.tensor(sorted(ghost_global_set), dtype=torch.long)  # [G]

    # ── build global to local index map ─────────────────────────────────────────
    K = owned_global.shape[0]
    G = ghost_global.shape[0]

    local_nodes     = torch.cat([owned_global, ghost_global], dim=0)   # [K+G]
    global_to_local = {int(g): l for l, g in enumerate(local_nodes.tolist())}

    # ── filter and remap edges ────────────────────────────────────────────────
    # Keep edges where at least one endpoint is an owned node.
    edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    for e in range(edge_index.shape[1]):
        s, d = int(src_nodes[e]), int(dst_nodes[e])
        if s in owned_set or d in owned_set:
            if s in global_to_local and d in global_to_local:
                edge_mask[e] = True

    kept_edges     = edge_index[:, edge_mask] #[2, E']
    local_edge_src = torch.tensor([global_to_local[int(n)] for n in kept_edges[0].tolist()], dtype=torch.long)
    local_edge_dst = torch.tensor([global_to_local[int(n)] for n in kept_edges[1].tolist()], dtype=torch.long)
    local_edge_index = torch.stack([local_edge_src, local_edge_dst], dim=0)

    # ── build local features ─────────────────────────────────────────────────
    local_x        = data.x[local_nodes]         if data.x is not None        else None
    local_pos      = data.pos[local_nodes]        if data.pos is not None      else None
    local_edge_attr= data.edge_attr[edge_mask]    if data.edge_attr is not None else None

    local_data = Data(x = local_x, pos = local_pos, edge_index = local_edge_index, edge_attr  = local_edge_attr, num_nodes = K+G)
    # carry through any extra scalar attributes
    for key in data.keys():
        if key not in ("x", "pos", "edge_index", "edge_attr"):
            setattr(local_data, key, getattr(data, key))

    # ── build GhostInfo ───────────────────────────────────────────────────────
    owned_mask = torch.zeros(K + G, dtype=torch.bool)
    owned_mask[:K] = True

    ghost_rank_list       = []
    ghost_remote_idx_list = []
    local_ghost_idx_list  = []

    # For each remote rank r, find the list of owned node indices on r that correspond to our ghost nodes.
    # "remote_idx" = position of the ghost in rank r's *owned* node list.
    # We need the same ordering that rank r will use, which is:
    #   torch.where(node_assignment == r)[0]  (sorted)
    owned_per_rank = {}
    for r in range(num_parts):
        if r != group_rank:
            owned_per_rank[r] = torch.where(node_assignment == r)[0].tolist()

    for local_g_idx in range(G):
        global_g = int(ghost_global[local_g_idx])
        owner_r  = int(node_assignment[global_g])
        remote_i = owned_per_rank[owner_r].index(global_g)   # position in owner's owned list
        ghost_rank_list.append(owner_r)
        ghost_remote_idx_list.append(remote_i)
        local_ghost_idx_list.append(K + local_g_idx)         # ghost comes after owned in local tensor

    ghost_rank_t        = torch.tensor(ghost_rank_list,       dtype=torch.long)
    ghost_remote_idx_t  = torch.tensor(ghost_remote_idx_list, dtype=torch.long)
    local_ghost_idx_t   = torch.tensor(local_ghost_idx_list,  dtype=torch.long)

    # ── build send lists (what *this* rank must send to others) ──────────────
    # rank r needs ghost data for nodes that we own; find which of our owned nodes appear as ghosts on each remote rank r.
    send_rank_list      = []
    send_local_idx_list = []
    recv_counts         = {}

    # Count ghosts grouped by owner rank
    ghosts_per_owner = defaultdict(int)
    for r in ghost_rank_list:
        ghosts_per_owner[r] += 1
    recv_counts = dict(ghosts_per_owner)

    # For sends: for each remote rank r, determine which of OUR owned nodes are ghosts *on rank r*.  
    # We detect this by looking at r's ghost list, but since we don't have that during dataset construction we reconstruct
    # it symmetrically: node g is a ghost on rank r if g \in owned_set(this rank)
    # and g is adjacent to at least one node owned by r.
    for r in range(num_parts):
        if r == group_rank:
            continue
        remote_owned_set = set(owned_per_rank[r])
        nodes_to_send = []
        for e in range(edge_index.shape[1]):
            s, d = int(src_nodes[e]), int(dst_nodes[e])
            # edge crosses boundary: one end is ours, the other belongs to r
            if s in owned_set and d in remote_owned_set:
                local_idx = global_to_local[s]
                if local_idx not in nodes_to_send:
                    nodes_to_send.append(local_idx)
            elif d in owned_set and s in remote_owned_set:
                local_idx = global_to_local[d]
                if local_idx not in nodes_to_send:
                    nodes_to_send.append(local_idx)
        if nodes_to_send:
            send_rank_list.append(r)
            send_local_idx_list.append(torch.tensor(sorted(nodes_to_send), dtype=torch.long))

    ghost_info = GhostInfo(
        #owned_mask       = owned_mask,
        ghost_rank       = ghost_rank_t,
        ghost_remote_idx = ghost_remote_idx_t,
        local_ghost_idx  = local_ghost_idx_t,
        send_rank        = send_rank_list,
        send_local_idx   = send_local_idx_list,
        recv_counts      = recv_counts,
    )

    return local_data, ghost_info

def HaloExchange_sync(node_feat, info, comm):
    device = node_feat.device
    dtype  = node_feat.dtype
    F_dim  = node_feat.shape[1]
    world  = dist.get_world_size(comm)
    G      = len(info.ghost_rank)
    K      = node_feat.shape[0] - G

    #print(f"Checking node size: total nodes {node_feat.shape[0]}; own nodes {K}; ghost node {G}", flush=True)

    # ── send/receive (unchanged) ───────────────────────────────────────────
    send_chunks = [
        torch.empty((0, F_dim), dtype=dtype, device=device)
        for _ in range(world)
    ]
    for dst_rank, idx in zip(info.send_rank, info.send_local_idx):
        send_chunks[int(dst_rank)] = (
            node_feat.index_select(0, idx.to(device)).contiguous()
        )

    input_split_sizes  = [int(c.shape[0]) for c in send_chunks]
    output_split_sizes = [int(info.recv_counts.get(src, 0)) for src in range(world)]

    send_buf = torch.cat(send_chunks, dim=0).contiguous()
    recv_buf = torch.empty(
        (sum(output_split_sizes), F_dim), dtype=dtype, device=device
    )

    recv_buf = dist_nn_f.all_to_all_single(
        output             = recv_buf,
        input              = send_buf,
        output_split_sizes = output_split_sizes,
        input_split_sizes  = input_split_sizes,
        group              = comm,
    )

    # ── build recv_positions: recv_positions[g] = row in recv_buf for ghost g ──
    offset_by_src = {}
    start = 0
    for src in range(world):
        offset_by_src[src] = start
        start += output_split_sizes[src]

    ghosts_by_src = defaultdict(list)
    for g in range(G):
        src        = int(info.ghost_rank[g])
        remote_idx = int(info.ghost_remote_idx[g])
        ghosts_by_src[src].append((remote_idx, g))

    recv_positions = torch.empty(G, dtype=torch.long, device=device)
    for src, pairs in ghosts_by_src.items():
        pairs.sort(key=lambda x: x[0])
        for k, (_, g) in enumerate(pairs):
            recv_positions[g] = offset_by_src[src] + k

    ghost_synced = recv_buf[recv_positions]               # [G, F_dim]

    # no clone, no in-place — clean autograd graph
    return torch.cat([node_feat[:K], ghost_synced], dim=0)

def check_same_sample_across_halo(data, ghost_info, comm):
    rank = dist.get_rank(comm)
    world = dist.get_world_size(comm)

    group = data.group
    if isinstance(group, (list, tuple)):
        group = group[0]

    sig = (str(group), int(data.t0), int(data.target_t))

    sigs = [None for _ in range(world)]
    dist.all_gather_object(sigs, sig, group=comm)

    #if rank == 0:
    #    print("Pei debugging sample sigs:", sigs, flush=True)

    if len(set(sigs)) != 1:
        raise RuntimeError(f"Halo sample mismatch: rank={rank}, sig={sig}, all={sigs}")
    
    x = data.x#[nnodes, T, C]
    h0 = x[:, 0, :].contiguous()
    h1 = HaloExchange_sync(h0, ghost_info, comm)

    maxdiff_all = (h1 - h0).abs().max().item()

    G      = len(ghost_info.ghost_rank)
    K      = x.shape[0] - G

    maxdiff_own = (h1[:K] - h0[:K]).abs().max().item()

    if ghost_info is not None and len(ghost_info.local_ghost_idx) > 0:
        ghost_idx = ghost_info.local_ghost_idx.to(h0.device)
        maxdiff_ghost = (h1[ghost_idx] - h0[ghost_idx]).abs().max().item()
    else:
        maxdiff_ghost = 0.0

    assert maxdiff_all<1e-6 and maxdiff_own<1e-6 and maxdiff_ghost<1e-6, (
        f"Ghost0 [rank {dist.get_rank(comm)}] first halo maxdiff_all={maxdiff_all:.6e}, "
        f"maxdiff_own={maxdiff_own:.6e}, "
        f"maxdiff_ghost={maxdiff_ghost:.6e}"
        )
