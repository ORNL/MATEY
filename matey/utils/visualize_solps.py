# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.io import loadmat
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_visual_contour_mapped(x_true, x, var_names, outputname, radCell, zCell, lt=1, sample=0):

    B, C, D, H, W = x.shape
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    cmap = cm.turbo 
    verts_list = [list(zip(radCell[:, i], zCell[:, i])) for i in range(radCell.shape[1])] 

    for iplot in range(C):
        # True
        ax = axs[0, iplot]
        data_true = x_true[0, iplot, :, :, :].squeeze().cpu().detach().numpy().reshape(-1)
        norm = colors.Normalize(vmin=np.min(data_true), vmax=np.max(data_true))
        colors_list = [cmap(norm(val)) for val in data_true]
        collection = PolyCollection(verts_list, facecolors=colors_list, edgecolors=None, linewidths=0)
        ax.add_collection(collection)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(radCell.min(), radCell.max())
        ax.set_ylim(zCell.min(), zCell.max())
        ax.set_title(var_names[iplot], fontsize=18)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
        if iplot == 0:
            ax.text(-0.55, 0.95, "True at \n%d" % (lt + sample - 1),
                    fontsize=16, transform=ax.transAxes,
                    verticalalignment='top',
                    clip_on=False)

        # Predicted
        ax = axs[1, iplot]
        data_pred = x[0, iplot, :, :, :].squeeze().cpu().detach().numpy().reshape(-1)
        norm = colors.Normalize(vmin=np.min(data_pred), vmax=np.max(data_pred))
        colors_list = [cmap(norm(val)) for val in data_pred]
        collection = PolyCollection(verts_list, facecolors=colors_list, edgecolors=None, linewidths=0)
        ax.add_collection(collection)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(radCell.min(), radCell.max())
        ax.set_ylim(zCell.min(), zCell.max())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
        if iplot == 0:
            ax.text(-0.55, 0.95, "Pred at \n%d" % (lt + sample - 1),
                    fontsize=16, transform=ax.transAxes,
                    verticalalignment='top',
                    clip_on=False)

    plt.subplots_adjust(left=0.02, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.05)
    plt.savefig(outputname,dpi=300)
    plt.close()

solps_d3d = '/global/cfs/projectdirs/amsc007/zhan1668/code/MATEY/examples/matey_SOLPS_leadtime_1.0.pt'
torchdata = torch.load(solps_d3d)
x_true = torchdata["target"]
x_pred = torchdata["output"]
print("x_true.shape", x_true.shape)
h, w= x_true.shape[-2:]

geometry_file= '/global/cfs/projectdirs/amsc007/zhan1668/MATEY/Datasets_pretraining/solps/SOLPS2DwION/D3D/174310_D/baserun/b2fgmtry.mat'
geom_data = loadmat(geometry_file, squeeze_me=True, struct_as_record=False)
print(geom_data.keys(), geom_data["Geo"])
Geo = geom_data["Geo"]
pr = Geo.pr 
pz = Geo.pz 

var_names = ['ne2d', 'te2d', 'ti2d']
plot_visual_contour_mapped(x_true, x_pred, var_names, "./matey_solps.png", pr, pz)

