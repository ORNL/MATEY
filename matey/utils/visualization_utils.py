import argparse
import os, sys
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, ticker
import torch
from einops import rearrange
from .distributed_utils import assemble_samples


def plot_visual_contour(x, var_names, outputname):
    B, C, H, W = x.shape
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for iplot in range(C):
        ax =axs[iplot]
        im=ax.contourf(x[0,iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
        ax.set_aspect('equal')
        ax.set_title(var_names[iplot])
        #ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')

    #fig.tight_layout()
    plt.subplots_adjust(left=0.02, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.05)
    plt.savefig(outputname)
    plt.close()


def plot_visual_contourcomp(x_true, x, var_names, outputname, debug=False, additional=None):
    B, C, H, W = x.shape
    if debug:
        fig, axs = plt.subplots(5 if additional is None else 6, 4, figsize=(16, 20))
        for iplot in range(C):
            for inp in range(4):
                ax=axs[inp, iplot]
                im=ax.contourf(x_true[inp,0, iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
                ax.set_aspect('equal')
                #ax.axis('off')
                ax.set_title(var_names[iplot])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax, orientation='vertical')

            ax =axs[4, iplot]
            im=ax.contourf(x[0,iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
            ax.set_aspect('equal')
            #ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical')
            if additional is not None:
                ax =axs[5, iplot]
                im=ax.contourf(additional[0,iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
                ax.set_aspect('equal')
                #ax.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax, orientation='vertical')

        #fig.tight_layout()
        plt.subplots_adjust(left=0.02, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.05)
        plt.savefig(outputname)
        plt.close()
        return

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for iplot in range(C):
        ax=axs[0, iplot]
        im=ax.contourf(x_true[0,iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
        ax.set_aspect('equal')
        #ax.axis('off')
        ax.set_title(var_names[iplot])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        if iplot>0:
            ax.set_yticklabels([])

        ax =axs[1, iplot]
        im=ax.contourf(x[0,iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
        ax.set_aspect('equal')
        #ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        if iplot>0:
            ax.set_yticklabels([])

    #fig.tight_layout()
    plt.subplots_adjust(left=0.02, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.05)
    plt.savefig(outputname)
    plt.close()

def plot_visual_contourcompthree(x_true, x, x_auto, var_names, outputname, iplot=3, lt=1, x_inp=None):
    T, C, H, W = x.shape
    if x_inp is None:
        fig, axs = plt.subplots(3, 4, figsize=(20, 12))
        inp=-1
    else:
        fig, axs = plt.subplots(4, 4, figsize=(20, 16))
        inp=0


    for iplot in range(C):
        if inp==0:
            ax=axs[inp, iplot]
            im=ax.contourf(x_inp[0, -1, iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(var_names[iplot], fontsize=18)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical')
            if iplot==0:
                ax.text(-100,100, "Input[-1]", fontsize=18)

        ax=axs[0+inp+1, iplot]
        im=ax.contourf(x_true[0,iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
        ax.set_aspect('equal')
        ax.axis('off')
        if inp<0:
            ax.set_title(var_names[iplot], fontsize=18)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        if iplot==0:
            ax.text(-100,100, "True at %d"%lt, fontsize=18)

        ax =axs[1+inp+1, iplot]
        im=ax.contourf(x[0,iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
        ax.set_aspect('equal')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        if iplot==0:
            ax.text(-120,100, "Pred at %d"%lt, fontsize=18)

        ax =axs[2+inp+1, iplot]
        im=ax.contourf(x_auto[0,iplot,:,:].cpu().squeeze().detach().numpy(), cmap="jet", levels=50)
        ax.set_aspect('equal')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        if iplot==0:
            ax.text(-120,100, "Auto-regress", fontsize=18)

    #fig.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.975, top=0.95, wspace=0.1, hspace=0.05)
    plt.savefig(outputname)
    plt.close()

def plot_visual_contourcomp_omg_eps(OMGtrue, EPSItrue, OMGpre, EPSIpre, var_names, outputname):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    for irow in range(2):
        if irow ==0:
            dataplot=[OMGtrue, OMGpre]
        else:
            dataplot=[EPSItrue, EPSIpre]
        strlabels=[", true", ", pred"]
        for icol in range(2):
            ax=axs[irow, icol]
            data = dataplot[icol][512,:,:]
            im=ax.contourf(data, cmap="jet", levels=50, locator=ticker.LogLocator())
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(var_names[irow]+strlabels[icol], fontsize=20)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical')
    #fig.tight_layout()
    plt.subplots_adjust(left=0.01, bottom=0.02, right=0.9, top=0.95, wspace=0.2, hspace=0.1)
    plt.savefig(outputname)
    plt.close()

def plot_visual_contourcompthree_cpu(x_true, x, var_names, outputname, iplot=3, lt=1):
    B, C, H, W = x.shape
    print(x.shape)
    fig, axs = plt.subplots(2, 4, figsize=(20, 12))
    inp=-1
    for iplot in range(C):
        ax=axs[0, iplot]
        im=ax.contourf(x_true[0,iplot,:,:], cmap="jet", levels=50)
        ax.set_aspect('equal')
        ax.axis('off')
        if inp<0:
            ax.set_title(var_names[iplot], fontsize=18)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        if iplot==0:
            ax.text(-300,500, "True", fontsize=18)

        ax =axs[1, iplot]
        im=ax.contourf(x[0,iplot,:,:], cmap="jet", levels=50)
        ax.set_aspect('equal')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        if iplot==0:
            ax.text(-300,500, "MATEY-Small", fontsize=18)

    #fig.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.975, top=0.95, wspace=0.1, hspace=0.05)
    plt.savefig(outputname)
    plt.close()

def plot_visual_contourindivi(x, outputname):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im=ax.contourf(x[:,:].squeeze().detach().numpy(), cmap="jet", levels=50)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    plt.savefig(outputname)
    plt.close()

def plot_loss_leadingtime(leadtime, samp, err_model_autoreg, err_model_LeadTim, err_reference, outputname):
    samp, err_model_autoreg, err_model_LeadTim, err_reference  = zip(*sorted(zip(samp, err_model_autoreg, err_model_LeadTim, err_reference)))
    print(leadtime, samp, err_model_autoreg, err_model_LeadTim, err_reference)
    colorlib=["k","b","g","r","m","y"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for isample in range(len(err_reference)):
        ax.plot(leadtime, err_model_LeadTim[isample], colorlib[isample%len(colorlib)]+"-",label="%d-Pred"%samp[isample])
        ax.plot(leadtime, err_model_autoreg[isample], colorlib[isample%len(colorlib)]+"--", label="%d-Auto-regress"%samp[isample])
        ax.plot(leadtime, err_reference[isample], colorlib[isample%len(colorlib)]+":", label="%d-Ref."%samp[isample])
        print(leadtime,err_reference[isample] )
    ax.legend(bbox_to_anchor=(1.0, 0.7))
    ax.set_title("Prediction loss vs. leading time")
    ax.set_xlabel("Leadtime")
    ax.set_ylabel("Normalized RMSE Loss")
    fig.tight_layout()
    plt.savefig(outputname)
    plt.close()

def checking_data_inp_tar(tar, inp, blockdict, global_rank, current_group, group_rank, group_size, device, outdir):
    tar = tar.to(device)
    inp = inp.to(device)
    plot_visual_contourcomp(tar[:,:,0,:,:], inp[:,0,:,0,:,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./parallelloading_checking_hw_rank{global_rank}.png"))       
    plot_visual_contourcomp(tar[:,:,:,0,:], inp[:,0,:,:,0,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./parallelloading_checking_dw_rank{global_rank}.png"))       
    if group_rank==0:
        tar_list = [torch.empty_like(tar) for _ in range(group_size)]
        inp_list = [torch.empty_like(inp) for _ in range(group_size)]
    else:
        tar_list = None
        inp_list = None
    global_dst = dist.get_global_rank(current_group, 0)
    dist.gather(tar, tar_list, dst=global_dst, group=current_group)
    dist.gather(inp, inp_list, dst=global_dst, group=current_group)
    if global_rank==0:
        nproc_blocks = blockdict["nproc_blocks"]
        tar_all = torch.stack(tar_list, dim=0)
        inp_all = torch.stack(inp_list, dim=0)
        p1,p2,p3=nproc_blocks
        tar_all = rearrange(tar_all,'(p1 p2 p3) b c d h w -> b c (p1 d) (p2 h) (p3 w)',    p1=p1, p2=p2, p3=p3)
        inp_all = rearrange(inp_all,'(p1 p2 p3) b t c d h w -> b t c (p1 d) (p2 h) (p3 w)',p1=p1, p2=p2, p3=p3)
        print("Loading checking", blockdict, inp_all.shape, tar_all.shape, flush=True)
        #################
        #plot_visual_contourcomp(tar_all[:,:,0,:,:], tar_all[:,:,0,:,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./parallelloading_checking_hw_full.png"))       
        plot_visual_contourcomp(tar_all[:,:,0,:,:], inp_all[:,0,:,0,:,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./parallelloading_checking_hw.png"))       
        plot_visual_contourcomp(tar_all[:,:,:,0,:], inp_all[:,0,:,:,0,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./parallelloading_checking_dw.png"))       
        #################

def checking_data_pred_tar(tar, pred, blockdict, global_rank, current_group, group_rank, group_size, device, outdir, istep=0, imod=0):
    if group_size>1:
        pred_all, tar_all = assemble_samples(tar, pred, blockdict, global_rank, current_group, group_rank, group_size, device)
    else:
        pred_all = pred
        tar_all = tar
    if global_rank==0:
        print("Pred checking", blockdict, pred_all.shape, tar_all.shape, flush=True)
        #################
        #plot_visual_contourcomp(tar_all[:,:,0,:,:], tar_all[:,:,0,:,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./datapred_checking_hw_full.png"))       
        plot_visual_contourcomp(tar_all[:,:,0,:,:], pred_all[:,:,0,:,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./datapred_checking_hw_istep{istep}_imod{imod}.png"))       
        plot_visual_contourcomp(tar_all[:,:,:,0,:], pred_all[:,:,:,0,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./datapred_checking_dw_istep{istep}_imod{imod}.png"))       
        #################

def checking_pred_full(pred, blockdict, global_rank, current_group, group_rank, group_size, device, outdir, istep=0, imod=0):
    pred = pred.to(device)
    #plot_visual_contour(pred[:,:,0,:,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./datapredfull_checking_hw_rank{global_rank}_istep{istep}_imod{imod}.png"))       
    #plot_visual_contour(pred[:,:,:,0,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./datapredfull_checking_dw_rank{global_rank}_istep{istep}_imod{imod}.png"))       
    if group_rank==0:
        pred_list = [torch.empty_like(pred) for _ in range(group_size)]
    else:
        pred_list = None
    global_dst = dist.get_global_rank(current_group, 0)
    dist.gather(pred, pred_list, dst=global_dst, group=current_group)
    if global_rank==0:
        nproc_blocks = blockdict["nproc_blocks"]
        pred_all = torch.stack(pred_list, dim=0)
        p1,p2,p3=nproc_blocks
        pred_all = rearrange(pred_all,'(p1 p2 p3) b c d h w -> b c (p1 d) (p2 h) (p3 w)',p1=p1, p2=p2, p3=p3)
        print("Pred full checking", blockdict, pred_all.shape, flush=True)
        #################
        plot_visual_contour(pred_all[:,:,0,:,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./datapredfull_checking_hw_istep{istep}_imod{imod}.png"))       
        plot_visual_contour(pred_all[:,:,:,0,:], ['Vx', 'Vy', 'Vw', 'pressure'],os.path.join(outdir, f"./datapredfull_checking_dw_istep{istep}_imod{imod}.png"))       
        #################