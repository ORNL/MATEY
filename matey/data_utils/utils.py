import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.distributed as dist
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from functools import reduce
from operator import mul
import random
import math

def closest_factors(n, dim):
    #temporary, from Andrey's
    assert n > 0 and dim > 0, f"{n} and {dim} must be greater than 0"

    if dim == 1:
        return [n]
    
    """
    factors = []
    i = 2
    nn = n
    while nn > 1:
        while nn % i == 0:
            factors.append(i)
            nn //= i
        i += 1

    # Reduce the list of factors to match the dimension (dim)
    while len(factors) > dim:
        # Combine the two smallest factors
        factors[1] *= factors[0]
        factors.pop(0)
        factors.sort()
    if len(factors) < dim:
        factors = [1]*(dim-len(factors)) + factors
    """

    factors = [1] * dim
    factors[0] = n

    while True:
        prev = factors.copy()
        factors.sort()
        largest = factors[-1]
        sqrt_large = int(math.sqrt(largest))
        for i in range(sqrt_large, 0, -1):
            if largest % i == 0:
                factor1, factor2 = i, largest // i
                break
        # If cannot further balance, break
        if factor1 == 1 or factor2 == largest or len(set(factors)) == 1:
            break
        factors[-1] = factor2
        factors[0] *= factor1
        if factors == prev:
            break

    factors.sort()

    assert reduce(mul, factors) == n and len(factors)==dim, f"factors, {factors}, dim {dim}"

    return factors

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
