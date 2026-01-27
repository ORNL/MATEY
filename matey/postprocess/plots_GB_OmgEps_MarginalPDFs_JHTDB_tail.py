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
caselabels=["Matey-ViT-PS64", "Matey-ViT-PS32", "Matey-PS8", "Matey-Small", "Matey-Medium"]
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
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
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

    if iplot==0:
        #pdf, f1_centers= turbulence_des.compute_pdf(np.log10(OMGtar/OMGtar.mean()), bins=200)
        rangf=(0,200)
        pdf, f1_centers= turbulence_des.compute_pdf(OMGtar/OMGtar.mean(), bins=200, range_f1=rangf)

        axs[0].plot(f1_centers, pdf,"ko", markerfacecolor='none', label="True", linewidth=2.0)
        axs[0].set_xlabel(r'$\Omega/<\Omega>$')
        axs[0].set_ylabel('PDF')
        #pdf, f1_centers= turbulence_des.compute_pdf(np.log10(EPSItar/EPSItar.mean()), bins=200)
        pdf, f1_centers= turbulence_des.compute_pdf(EPSItar/EPSItar.mean(), bins=200, range_f1=rangf)

        axs[1].plot(f1_centers, pdf, "ko", markerfacecolor='none', label="True", linewidth=2.0)
        axs[1].set_xlabel(r'$\varepsilon/<\varepsilon>$')
        #axs[1].set_ylabel('PDF')
    if iplot==1:
        rangf=(0,150)
    else:
        rangf=(0,200)
    pdf, f1_centers= turbulence_des.compute_pdf(OMGpre/OMGpre.mean(), bins=200, range_f1=rangf)
    axs[0].plot(f1_centers, pdf, lines[iplot], color=colors[iplot], label=caselabel, linewidth=2.0)
    #axs[0].set_title("Enstrophy")
    axs[0].set_yscale("log")
    pdf, f1_centers= turbulence_des.compute_pdf(EPSIpre/EPSIpre.mean(), bins=200, range_f1=rangf)
    axs[1].plot(f1_centers, pdf, lines[iplot], color=colors[iplot], label=caselabel, linewidth=2.0)
    #axs[1].set_title("Dissipation")
    axs[1].set_yscale("log")
axs[1].legend(fontsize=22,handlelength=1,handletextpad=0.2)  
plt.subplots_adjust(left=0.12, bottom=0.18, right=0.98, top=0.9, wspace=0.2, hspace=0.05)
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/pdfs_omg_epsi_ylog_cutoff.png")
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/pdfs_omg_epsi_ylog_cutoff.pdf")
###############################################################################################
###############################################################################################




