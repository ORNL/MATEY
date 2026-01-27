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
imod=0
it=82
casestrs=["", "", "", "_fine", "_fine"]
caselabels=["ViT-PS64", "Matey-ViT-PS32", "Matey-PS8", "Matey-Small", "Matey-Medium"]
casedirs=['/lustre/orion/lrn037/scratch/zhangp/fy25/hierarchical/matey/Dev_JHUTDB3D/basic_config/demo_jhutdb_baseline_ps64/plots_output/',
        '/lustre/orion/lrn037/scratch/zhangp/fy25/hierarchical/matey/Dev_JHUTDB3D/basic_config/demo_jhutdb_baseline/plots_output/',
        '/lustre/orion/lrn037/scratch/zhangp/fy25/hierarchical/matey/Dev_JHUTDB3D/basic_config/demo_jhutdb_hierarchical/plots_output/',
        '/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/JHTDB/jh_gb222/plots_output/',
        '/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/JHTDB/jh_gb111_med/plots_output'
        ]
cutoffs=[512//64, 512//32, 512//8, 512//2, 512//1]
stat_dict = {caselabel:{} for caselabel in caselabels}
for caselabel, casestr, output_dir in zip(caselabels, casestrs, casedirs):
    Omg_Epsi=np.load(os.path.join(output_dir,f"./Omg_Epsi_{setname}{casestr}_it{it}.npz"))
    """
    OMGtar=Omg_Epsi["OMGtar"]
    EPSItar=Omg_Epsi["EPSItar"]
    OMGpre=Omg_Epsi["OMGpre"] 
    EPSIpre=Omg_Epsi["EPSIpre"]
    """
    stat_dict[caselabel]["Omg_Epsi"]=Omg_Epsi
    dataspectrum=np.load(os.path.join(output_dir,f"./Ek_{setname}{casestr}_it{it}.npz"))
    """
    kvals_base_c=dataspectrum["kvals"]
    Ek_base_c=dataspectrum["Ek"]
    """
    stat_dict[caselabel]["dataspectrum"]=dataspectrum


lwd=2.0
colors = ['#377eb8', '#4daf4a','#ff7f00', 
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']
lines=["-","--","-.",":"]
###############################################################################################
###############################################################################################
turbulence_des = turbulence_descriptors.turbulence_descriptor()
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
iplot=-1
for icase, (caselabel, cutoff) in enumerate(zip(caselabels, cutoffs)):
    if icase in [0,2]:
        continue
    iplot+=1
    Omg_Epsi = stat_dict[caselabel]["Omg_Epsi"]
    OMGtar=Omg_Epsi["OMGtar"]
    EPSItar=Omg_Epsi["EPSItar"]
    OMGpre=Omg_Epsi["OMGpre"] 
    EPSIpre=Omg_Epsi["EPSIpre"]
    range_f1=(-5, 5)
    range_f2=(-4, 4)
    range_f1=(-2, 2)
    range_f2=(-2, 2)
    if iplot==0:
        # Compute joint PDF
        #range_f1=(min(np.log10(OMGtar/OMGtar.mean()).min(),   np.log10(OMGpre/OMGpre.mean()).min()), 10.0)
        #range_f2=(min(np.log10(EPSItar/EPSItar.mean()).min(), np.log10(EPSIpre/EPSIpre.mean()).min()), 10.0)
        pdf, f1_centers, f2_centers = turbulence_des.compute_joint_pdf(np.log10(OMGtar/OMGtar.mean()), np.log10(EPSItar/EPSItar.mean()), 
                                                                range_f1=range_f1, range_f2=range_f2, bins=200)
        im=axs[0,0].contourf(f1_centers, f2_centers, pdf.T, cmap="Reds", levels=50)
        isolines = axs[0,0].contour(f1_centers, f2_centers, pdf.T, levels=10, colors='k', linewidths=0.5)
        #axs[0,0].set_xlabel(r'log10($\Omega/<\Omega>$)')
        axs[0,0].set_ylabel(r'log10($\varepsilon/<\varepsilon>$)')
        axs[0,0].set_title('True Joint PDF')
        divider = make_axes_locatable(axs[0,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
    irow=(iplot+1)//2
    icol=(iplot+1)%2
    pdfpre, f1_centers, f2_centers = turbulence_des.compute_joint_pdf(np.log10(OMGpre/OMGpre.mean()), np.log10(EPSIpre/EPSIpre.mean()), 
                                                            range_f1=range_f1, range_f2=range_f2, bins=200)
    im=axs[irow,icol].contourf(f1_centers, f2_centers, pdfpre.T, cmap="Reds", levels=50)
    isolines = axs[irow,icol].contour(f1_centers, f2_centers, pdfpre.T, levels=10, colors='k', linewidths=0.5)
    if irow==1:
        axs[irow,icol].set_xlabel(r'log10($\Omega/<\Omega>$)')
    if icol==0:
        axs[irow,icol].set_ylabel(r'log10($\varepsilon/<\varepsilon>$)')
    axs[irow,icol].set_title(caselabel)
    divider = make_axes_locatable(axs[irow,icol])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')
  
plt.subplots_adjust(left=0.09, bottom=0.1, right=0.91, top=0.92, wspace=0.27, hspace=0.3)
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/jpdf_omg_epsi.png")
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/jpdf_omg_epsi.pdf")
###############################################################################################
###############################################################################################




