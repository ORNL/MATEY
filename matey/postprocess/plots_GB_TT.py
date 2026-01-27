# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import argparse
import os, sys
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from einops import rearrange
from collections import OrderedDict
from torchinfo import summary
try:
    from data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT
    from models.avit import build_avit
    from models.svit import build_svit
    from models.vit import build_vit
    from utils.YParams import YParams
    from utils.visualization_utils import plot_visual_contourcomp, plot_visual_contourcompthree_cpu, plot_visual_contourcomp_omg_eps 
    from trustworthiness import turbulence_descriptors
    from utils.distributed_utils import parse_slurm_nodelist, splitsample, get_sequence_parallel_group, locate_group
    from data_utils.utils import construct_filterkernels, multimods_turbulencetransformer, generate_grid, mods_assemble
except:
    from .data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT
    from .models.avit import build_avit
    from .models.svit import build_svit
    from .models.vit import build_vit
    from .utils.YParams import YParams
    from utils.visualization_utils import plot_visual_contourcomp, plot_visual_contourcompthree_cpu, plot_visual_contourcomp_omg_eps 
    from .trustworthiness import turbulence_descriptors
    from .utils.distributed_utils import parse_slurm_nodelist, splitsample, get_sequence_parallel_group, locate_group
    from .data_utils.utils import construct_filterkernels, multimods_turbulencetransformer, generate_grid, mods_assemble
from datetime import timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rcParams.update({'font.size': 24})
plt.style.use('tableau-colorblind10')



labelsplot = ['Vx', 'Vy', 'Vw', 'pressure']
setname="valid"
it=82
caselabels=["Matey-Medium"]
casedirs=['/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/JHTDB/jh_gb111_med/plots_output']
stat_dict = {caselabel:{} for caselabel in caselabels}

###############################################################################################
###############################################################################################
turbulence_des = turbulence_descriptors.turbulence_descriptor()
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
iplot=-1
tar_total = np.zeros((1, 4, 1024, 1024, 1024), dtype='float32') # B, C, D, H, W
pred_total= np.zeros((1, 4, 1024, 1024, 1024), dtype='float32') # B, C, D, H, W
for icase, (caselabel, output_dir) in enumerate(zip(caselabels, casedirs)):
    for imod in range(3):
        iplot+=1
        file_pred = f"{output_dir}/pre_{setname}_t_{it}_{it}_x_0_1024_y_0_1024_z_0_1024mode_{imod}.npy"
        file_true = f"{output_dir}/tar_{setname}_t_{it}_{it}_x_0_1024_y_0_1024_z_0_1024mode_{imod}.npy"
        if imod>0:
            tar_total = np.load(file_true)
            pred_total= np.load(file_pred)
        else:
            try:
                for id in range(0,1024,256):
                    for ih in range(0,1024,256):
                        for iw in range(0,1024,256):
                            #pre_valid_t_82_82_x_768_1024_y_512_768_z_768_1024.npy
                            filestr=f"t_{it}_{it}_x_{ih}_{ih+256}_y_{iw}_{iw+256}_z_{id}_{id+256}"
                            #tar_total[:,:,id:id+256,ih:ih+256,iw:iw+256] = np.load(f"{output_dir}/tar_{setname}_{filestr}.npy")
                            pred_total[:,:,id:id+256,ih:ih+256,iw:iw+256] = np.load(f"{output_dir}/pre_{setname}_{filestr}.npy")
            except:
                iblk=-1
                for id in range(0,1024,128):
                    for ih in range(0,1024,128):
                        for iw in range(0,1024,128):
                            iblk+=1
                            filestr_time=f"t_{it}_{it}"
                            filestr_space_mod = f"x_0_1024_y_0_1024_z_0_1024mode_0_blk_{iblk}"
                            filestr=f"{setname}_{filestr_time}_{filestr_space_mod}"
                            tar_total[:,:,id:id+128,ih:ih+128,iw:iw+128] = np.load(f"{output_dir}/tar_{filestr}.npy")
                            pred_total[:,:,id:id+128,ih:ih+128,iw:iw+128] = np.load(f"{output_dir}/pre_{filestr}.npy")
        if imod==0:
            im=axs[0].contourf(tar_total[0,0,0,:,:], cmap="jet", levels=50, vmin=-1.4, vmax=2.4)
            axs[0].set_aspect('equal')
            axs[0].axis('off')
            axs[0].set_title("True Vx")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im.set_clim(-1.4, 2.4)
            cax.set_axis_off()

        irow=(iplot+1)#//2
        icol=(iplot+1)%2
        im=axs[irow].contourf(pred_total[0,0,0,:,:], cmap="jet", levels=50, vmin=-1.4, vmax=2.4)
        axs[irow].set_title(f"Mode {imod}")
        axs[irow].set_aspect('equal')
        axs[irow].axis('off')
        divider = make_axes_locatable(axs[irow])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im.set_clim(-1.4, 2.4)
        if irow!=3:
            cax.set_axis_off()
        else:
            cbar=plt.colorbar(im, cax=cax, orientation='vertical')

    
plt.subplots_adjust(left=0.01, bottom=0.02, right=0.93, top=0.90, wspace=0.02, hspace=0.2)
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/JHTDB_Ufield_TT.png")
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/JHTDB_Ufield_TT.pdf")





