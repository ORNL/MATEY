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



labelsplot = ['\rho', 'V_x', 'V_y', 'V_z']
setname="valid"
imod=1
caselabels=["Matey-Medium"]
casedirs=['/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/TG/tg11_medlt10_boosted/plots_output']
datasets=["P1F2R32",  "P1F3R64",  "P1F4R96"]
res_dict={
        "P1F2R32": [256, 512, 512],
        "P1F2R64": [384, 768, 768],
        "P1F2R96": [512, 1024, 1024],
        "P1F3R32": [256, 512, 512],
        "P1F3R64": [384, 768, 768],
        "P1F3R96": [512, 1024, 1024],
        "P1F4R32": [256, 512, 512],
        "P1F4R64": [384, 768, 768],
        "P1F4R96": [512, 1024, 1024]
    }
###############################################################################################
turbulence_des = turbulence_descriptors.turbulence_descriptor()
fig, axs = plt.subplots(6, 4, figsize=(16, 24))
iplot=-1
for icase, (caselabel, output_dir) in enumerate(zip(caselabels,casedirs)):
     for icol, it in enumerate([53, 253, 453, 653]):
            for idata, datastr in enumerate(datasets):
                D,H,W=res_dict[datastr]
                filestr_space=f"x_{0}_{H}_y_{0}_{W}_z_{0}_{D}"
                filestr_time=f"t_{it}_{it}"
                tar = np.load(f"{output_dir}/{datastr}/tar_{setname}_{filestr_time}_{filestr_space}mode_{imod}.npy")
                pred = np.load(f"{output_dir}/{datastr}//pre_{setname}_{filestr_time}_{filestr_space}mode_{imod}.npy")
                #B,C,D,H,W
                rho = pred[0,0,:,:,:].squeeze()
                uvec = pred[0,1:4,:,:,:].squeeze() #C,D,H,W
                rhotar = tar[0,0,:,:,:].squeeze()
                uvectar = tar[0,1:4,:,:,:].squeeze()#C,D,H,W
                rho = rho.transpose((1,2,0))
                rhotar = rhotar.transpose((1,2,0))
                uvec = uvec.transpose((2,3,1,0))
                uvectar = uvectar.transpose((2,3,1,0))

                irow=idata*2
                im=axs[irow, icol].contourf(rhotar[:,:,D//8//2], cmap="jet", levels=50)
                axs[irow, icol].set_aspect('equal')
                axs[irow, icol].axis('off')
                if irow==0:
                     axs[irow, icol].set_title(f"$t=%.2f$"%(it*0.04), fontsize=48)
                divider = make_axes_locatable(axs[irow, icol])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cax.set_axis_off()

                irow=idata*2+1
                im=axs[irow, icol].contourf(rho[:,:,D//8//2], cmap="jet", levels=50)
                #axs[irow, icol].set_title(caselabel)
                axs[irow, icol].set_aspect('equal')
                axs[irow, icol].axis('off')
                divider = make_axes_locatable(axs[irow, icol])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cax.set_axis_off()
                
plt.subplots_adjust(left=0.01, bottom=0.02, right=0.93, top=0.95, wspace=0.02, hspace=0.2)
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/TG_rhofield.png",dpi=300)
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/TG_rhofield.pdf")





