import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, math
from einops import rearrange
from operator import mul
from functools import reduce
import torch.distributed as dist

def normalize_spatiotemporal_persample(x, sequence_parallel_group=None):
    # input tensor shape: [T, B, C, D, H, W]
    ######## Normalize (time + space per sample)########
    with torch.no_grad():
        if sequence_parallel_group is not None:
            data_mean = torch.mean(x, dim=(0, -3, -2, -1), keepdim=True)
            data_square = torch.mean(torch.square(x), dim=(0, -3, -2, -1), keepdim=True)
            torch.set_printoptions(precision=15)
            dist.all_reduce(data_mean, op=dist.ReduceOp.SUM, group=sequence_parallel_group)
            dist.all_reduce(data_square,  op=dist.ReduceOp.SUM, group=sequence_parallel_group)
            world_size = dist.get_world_size(sequence_parallel_group)
            data_mean = data_mean/world_size
            data_square = data_square/world_size
            var = data_square - torch.square(data_mean)
            data_std  = torch.sqrt(torch.clamp(var,min=0.0))
        else:
            data_std, data_mean = torch.std_mean(x, dim=(0, -3, -2, -1), keepdim=True)

        #data_std = data_std + 1e-7 # Orig 1e-7
        data_std = torch.clamp_min(data_std, 1e-4)
    x = (x - data_mean) / (data_std)
    return x, data_mean, data_std

def get_top_variance_patchids(patch_size, data, gammaref, refine_ratio):
        #FIXME: need to figure out better way to extend to more levels
        assert len(patch_size)<3, "Implemented for two levels for now"
        #T,C,D,H,W
        space_dims= data.shape[2:]
        ps = patch_size[-1]
        ntokendim=[]
        #psz, psx, psy
        for idim, dim in enumerate(space_dims):
            ntokendim.append(dim//ps[idim])
        num_tokens=reduce(mul, ntokendim)
        xdata = torch.from_numpy(np.asarray(data))
        II_topk = torch.empty(num_tokens, dtype=torch.int).fill_(-1)
        #T,C,D,H,W-->T,C,ntz,ntx,nty,psz,psx,psy->c,ntx,ntx,nty->ntx,nty,ntz
        variance = xdata.unfold(2,ps[0],ps[0]).unfold(3,ps[1],ps[1]).unfold(4,ps[2],ps[2]).var(dim=(0,-3,-2,-1)).mean(dim=0)
        assert ntokendim==list(variance.shape)
        variance = rearrange(variance, 'ntz ntx nty -> (ntz ntx nty)')
        if gammaref is None:
            nrefines = int(refine_ratio * num_tokens)
            _, II_t = variance.topk(nrefines)
        else:
            varmax = variance.max()
            b = variance > varmax*gammaref
            II_t = b.nonzero().type(torch.int)[:,0]
        II_topk[:len(II_t)]=II_t
        return II_topk

def plot_checking(patch_size, field_names, time_idx, data):
    #T,C,D,H,W
    nvar = data.shape[1]
    space_dims= data.shape[2:]
    ps = patch_size[-1]
    ntokendim=[]
    #psz, psx, psy
    for idim, dim in enumerate(space_dims):
        ntokendim.append(dim//ps[idim])

    ncol=math.ceil(math.sqrt(nvar))
    nrow=nvar//ncol

    fig, axs = plt.subplots(nrow,ncol,figsize=(20, 20))
    for ivar in range(nvar):
        irow=ivar//ncol
        icol=ivar%ncol
        ax = axs[irow, icol]
        cs = ax.contourf(np.asarray(data[0, ivar, 0, :-20, :]).squeeze(), cmap="jet", levels=50)
        for iref in range(ntokendim[1]*ntokendim[2]):
            itoken = iref
            ix=itoken//ntokendim[2]
            iy=itoken%ntokendim[2]
            assert ix*ntokendim[2]+iy==itoken
            ax.plot([iy*ps[2],     (iy+1)*ps[2]], [ix*ps[1],         ix*ps[1]], "m--")
            ax.plot([iy*ps[2],     (iy+1)*ps[2]], [(ix+1)*ps[1], (ix+1)*ps[1]], "m--")
            ax.plot([iy*ps[2],         iy*ps[2]], [ix*ps[1],     (ix+1)*ps[1]], "m--")
            ax.plot([(iy+1)*ps[2], (iy+1)*ps[2]], [ix*ps[1],     (ix+1)*ps[1]], "m--")
            variance=data[:, ivar, 0, ix*ps[1]:(ix+1)*ps[1], iy*ps[2]:(iy+1)*ps[2]].var(axis=(0,-2,-1)).mean()
            ax.text(iy*ps[2],ix*ps[1],"%.2e"%variance)
        ax.set_aspect('equal')
        #ax.axis('off')
        ax.set_title(field_names[ivar])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cs, cax=cax, orientation='vertical')
    fig.tight_layout()
    plt.savefig(f"./imgs/check_all_{time_idx}.png")
    plt.close()
    sys.exit(-1)

def plot_refinedtokens(patch_size, field_names, time_idx, data, II_topk):
    #T,C,D,H,W
    nvar = data.shape[1]
    space_dims= data.shape[2:]
    ps = patch_size[-1]
    ntokendim=[]
    #psz, psx, psy
    for idim, dim in enumerate(space_dims):
        ntokendim.append(dim//ps[idim])
    
    ncol=math.ceil(math.sqrt(nvar))
    nrow=nvar//ncol
    for iref in range(len(II_topk)):
        itoken = II_topk[iref]
        if itoken<0:
            continue
        ####(iz,ix,iy) --> (iz+ix*nz+iy*nz*nx)
        itoken=itoken.cpu()
        iy = itoken//(ntokendim[0]*ntokendim[1])
        izx= itoken% (ntokendim[0]*ntokendim[1])
        ix = izx//ntokendim[0]
        iz = izx% ntokendim[0]
        assert iz+ix*ntokendim[0]+iy*ntokendim[0]*ntokendim[1]==itoken

        fig, axs = plt.subplots(nrow,ncol, figsize=(20, 20))
        for ivar in range(nvar):
            irow=ivar//ncol
            icol=ivar%ncol
            ax = axs[irow, icol]
            cs = ax.contourf(np.asarray(data[0, ivar, iz, :240, :]).squeeze(), cmap="jet", levels=50)
       
            ax.plot([iy*ps[2],     (iy+1)*ps[2]], [ix*ps[1],         ix*ps[1]], "m--")
            ax.plot([iy*ps[2],     (iy+1)*ps[2]], [(ix+1)*ps[1], (ix+1)*ps[1]], "m--")
            ax.plot([iy*ps[2],         iy*ps[2]], [ix*ps[1],     (ix+1)*ps[1]], "m--")
            ax.plot([(iy+1)*ps[2], (iy+1)*ps[2]], [ix*ps[1],     (ix+1)*ps[1]], "m--")
            variance=data[:, ivar, iz, ix*ps[1]:(ix+1)*ps[1], iy*ps[2]:(iy+1)*ps[2]].var(axis=(0,-2,-1)).mean()
            ax.text(iy*ps[2],ix*ps[1],"%.2e"%variance)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(field_names[ivar])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(cs, cax=cax, orientation='vertical')
            #ax.set_title(variance[II_topk[iref]].item())
        fig.tight_layout()
        plt.savefig(f"./imgs/check_ref_{time_idx}_iz{iz}.png")
        plt.close()

def figure_checking(x_0, xlocal, B, ntokenz, ntokenx, ntokeny, ps0, ps1, ps2, refineind):
    #FIXME: need to check B value; did not verify code
    return 
    x_0 = rearrange(x_0, '(b ntz ntx nty) t c d h w-> t b c (ntz d) (ntx h) (nty w)', ntz=ntokenz, ntx=ntokenx, nty=ntokeny)
    varmin = x_0.amin(dim=(0,1,-3,-2,-1))
    varmax = x_0.amax(dim=(0,1,-3,-2,-1))
    print(varmin, varmax)
    for ib in range(B):
        fig, axs = plt.subplots(2,2, figsize=(10, 10))
        for ivar in range(4):
            irow=ivar//2
            icol=ivar%2
            ax = axs[irow, icol]
            cs = ax.contourf(x_0[0, ib, ivar, 0, :, :].squeeze().cpu().detach().numpy().transpose(), cmap="jet", levels=50)
            for iref in range(len(refineind)//B):
                itoken = refineind[iref+ib*ntokenx*ntokeny+0]

      
                if itoken<0:
                    continue
                ####(iz,ix,iy) --> (iz+ix*nz+iy*nz*nx)
                ix=itoken.cpu()//ntokeny
                iy=itoken.cpu()%ntokeny
                assert ix*ntokeny+iy==itoken
                ax.plot([ix*ps0,         ix*ps0], [iy*ps1,     (iy+1)*ps1], "m--")
                ax.plot([(ix+1)*ps0, (ix+1)*ps0], [iy*ps1,     (iy+1)*ps1], "m--")
                ax.plot([ix*ps0,     (ix+1)*ps0], [iy*ps1,         iy*ps1], "m--")
                ax.plot([ix*ps0,     (ix+1)*ps0], [(iy+1)*ps1, (iy+1)*ps1], "m--")
            ax.set_aspect('equal')
            ax.axis('off')
        fig.tight_layout()
        plt.savefig(f"check_{ib}.png")
        plt.close()

    ibt = -1
    print(xlocal.size())
    for ib in range(B):
        fig, axs = plt.subplots(2,2, figsize=(10, 10))
        for iref in range(len(refineind)//B):
            itoken = refineind[iref+ib*ntokenx*ntokeny+0]
            if itoken<0:
                continue
            ix=itoken.cpu()//ntokeny
            iy=itoken.cpu()%ntokeny
            assert ix*ntokeny+iy==itoken
            ibt += 1
            for ivar in range(4):
                irow=ivar//2
                icol=ivar%2
                ax = axs[irow, icol]
                cs = ax.contourf(np.arange(ps0)+ps0*ix.item(), np.arange(ps1)+ps1*iy.item(),xlocal[ibt, 0, ivar,0, :, :].squeeze().cpu().detach().numpy().transpose(),
                                    vmin=varmin[ivar], vmax=varmax[ivar], cmap="jet", levels=50)
                ax.set_xlim(0, 256)
                ax.set_ylim(0, 256)
                ax.set_aspect('equal')
                ax.axis('off')
        fig.tight_layout()
        plt.savefig(f"check_local{ib}.png")
        plt.close()
    sys.exit(-1)
