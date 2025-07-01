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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
#plt.figure(figsize=(8,6))
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
iplot=-1
for icase, (caselabel, cutoff) in enumerate(zip(caselabels, cutoffs)):
    if icase in [0,2]:
        continue
    iplot+=1
    dataspectrum = stat_dict[caselabel]["dataspectrum"]
    kvals=dataspectrum["kvals"]
    Ek=dataspectrum["Ek"]
    kvalstar=dataspectrum["kvalstar"]
    Ektar=dataspectrum["Ektar"]
    #cutoff at 483
    Ek   =np.array([ek for ik, ek in zip(kvals, Ek) if ik<483])
    kvals=np.array([ik for ik in kvals if ik<483 ])
    Ektar   =np.array([ek for ik, ek in zip(kvalstar, Ektar) if ik<483])
    kvalstar=np.array([ik for ik in kvalstar if ik<483 ])
    if iplot==0:
        C = 2.0 
        kplot=np.arange(2,150)
        slope_53 = C * (kplot**(-5.0/3.0))
        ax.loglog(kplot, slope_53,"k:", label=r'Reference $k^{-5/3}$')
        """
        data = np.loadtxt('/lustre/orion/lrn037/proj-shared/zhangp/JHUTDB/isotropic1024fine/spectrum.txt', skiprows=2)
        kvals = data[:, 0]
        Ek = data[:, 1]
        """
        ax.loglog(kvalstar, Ektar, "ko", markerfacecolor='none', markersize=8, label='True spectrum', linewidth=lwd)#, alpha=0.6) 
    #plt.loglog([cutoff, cutoff],[1e-7, 1],"k:")
    ax.loglog(kvals[1:], Ek[1:], "-",color=colors[iplot], label=caselabel, linewidth=lwd)  
    if False:
        #FIXME: doesn't look good
        # Create an inset axes: width and height can be specified as percentages of the main axes
        axins = inset_axes(ax, width="30%", height="30%", loc="upper right")
        
        axins.set_clip_on(False)
        if iplot==0:
            C = 2.0 
            kplot=np.arange(2,150)
            slope_53 = C * (kplot**(-5.0/3.0))
            axins.loglog(kplot, slope_53,"k:", label=r'Reference $k^{-5/3}$')
            axins.loglog(kvalstar, Ektar, "ko", markerfacecolor='none', markersize=8, label='True spectrum', linewidth=lwd)#, alpha=0.6) 
        axins.loglog(kvals[1:], Ek[1:], "-",color=colors[iplot], label=caselabel, linewidth=lwd)  
        axins.set_xlim(100, 510)
        axins.set_ylim(1.2e-7, 1e-4)       
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$E(k)$')
ax.set_title('Energy Spectrum')
#plt.grid(True)  
ax.legend(fontsize=22)
ax.set_ylim(1e-7, 1.0)
plt.subplots_adjust(left=0.16, bottom=0.13, right=0.99, top=0.90, wspace=0.05, hspace=0.05)
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/energy_spectrum_GB.png")
plt.savefig("/lustre/orion/lrn037/scratch/zhangp/fy25/GB/matey/GB/energy_spectrum_GB.pdf")




