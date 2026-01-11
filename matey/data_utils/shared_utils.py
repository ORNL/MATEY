import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, math
from einops import rearrange
from operator import mul
from functools import reduce
import torch.distributed as dist

def normalize_spatiotemporal_persample(x):
    # input tensor shape: [T, B, C, D, H, W]
    ######## Normalize (time + space per sample)########
    with torch.no_grad():
        data_std, data_mean = torch.std_mean(x, dim=(0, -3, -2, -1), keepdim=True)
        
        dist.all_reduce(data_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(data_std,  op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        data_mean = data_mean / world_size
        data_std  = data_std  / world_size

        #data_std = data_std + 1e-7 # Orig 1e-7
        data_std = torch.clamp_min(data_std, 1e-4)
    x = (x - data_mean) / (data_std)
    return x, data_mean, data_std

def mask_to_indices(b):
    """
    b: boolean mask,[B, N] 
    return: a padded index tensor [B, Kmax].
        Each row i contains the column indices j where b[i, j] is True,
        in increasing j order, padded with -1 to length Kmax = max_i sum(b[i]).
        b = tensor([[0,1,0,1],[1,0,0,0]], dtype=torch.bool)
        idx_pad = tensor([[1,3],[0,-1]])
    """
    B = b.shape[0]
    max_k  = b.sum(1).max().item()
    idx_pad = torch.full((B, max_k), -1, dtype=torch.long, device=b.device)
    slots = b.cumsum(dim=1)
    rows, cols = b.nonzero(as_tuple=True)         
    idx_pad[rows, slots[rows, cols] - 1] = cols
    return idx_pad

def unfold_patches(x, patch_size):
    """
    T,B,C,D,H,W-->T,B,C,ntz,ntx,nty,psz,psx,psy
    """
    pd, ph, pw = patch_size
    return x.unfold(3, pd, pd).unfold(4, ph, ph).unfold(5, pw, pw)

def gather_patches(x_pl, token_idx):
    """
    x_pl:T, B, C, ntz*ntx*nty, psz, psx, psy
    token_idx: B, k
    return: T, nbatch, C, psz, psx, psy
    """
    T, B, C, N, psz, psx, psy = x_pl.shape
    k = token_idx.shape[1]
    expand_shape = (T, B, C, k, psz, psx, psy)
    gather_idx = token_idx.clamp(min=0).unsqueeze(0).unsqueeze(2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(expand_shape)
    gathered = x_pl.gather(dim=3, index=gather_idx) #T, B, C, k, psz, psx, psy
    valid_mask       = token_idx >= 0 #B, k
    gathered = rearrange(gathered,'T B C k psz psx psy -> T (B k) C psz psx psy')
    gathered = gathered[:, valid_mask.flatten()] # keep only real tokens
    return gathered            

def get_top_variance_patchids_ilevel(ps, xdata, gammaref, refine_ratio, full=True):
    #T,B,C,D,H,W
    B, space_dims= xdata.shape[1], xdata.shape[3:]
    ntokendim=[]
    #psz, psx, psy
    for idim, dim in enumerate(space_dims):
        ntokendim.append(dim//ps[idim])
    num_tokens=reduce(mul, ntokendim)
    #T,B,C,D,H,W-->T,B,C,ntz,ntx,nty,psz,psx,psy->B,C,ntz,ntx,nty->B,ntz,nty,nty
    variance = xdata.unfold(3,ps[0],ps[0]).unfold(4,ps[1],ps[1]).unfold(5,ps[2],ps[2]).var(dim=(0,-3,-2,-1)).mean(dim=1)
    assert ntokendim==list(variance.shape[1:])
    variance = rearrange(variance, 'B ntz ntx nty -> B (ntz ntx nty)')
    if gammaref is None:
        nrefines = int(refine_ratio * num_tokens)
        _, II_t = variance.topk(nrefines, dim=1)
    else:
        varmax, _ = variance.max(dim=1, keepdim=True)
        b = variance > varmax*gammaref
        II_t=mask_to_indices(b)
    idx_pad = torch.full((B, num_tokens if full else II_t.shape[1]), -1, dtype=torch.long, device=xdata.device)
    idx_pad[:,:II_t.shape[1]]=II_t
    return idx_pad

def get_top_variance_patchids_new(patch_size, xdata, gammaref, refine_ratio, full=True):
    #used in the matey paper 2025 with 2D data and 2 levels
    #xdata: T,B,C,D,H,W
    #patch_size: e.g. [[1,8,8], [1,16,16], ...]
    if len(patch_size)>=3:
        print("Warning: not tested with 3 levels yet!!!!")
        
    if gammaref is None and refine_ratio is None:
        return [],[]
    refineind = []
    x_current = xdata
    blockdict_list=[]
    for ps in reversed(patch_size[1:]): 
        II_t = get_top_variance_patchids_ilevel(ps, x_current, gammaref, refine_ratio, full=full)
        #[B_parent, k]
        refineind.append(II_t)
        #prepare data for next levels
        #T,B_parent,C,D,H,W-->T,B_parent,C,ntz,ntx,nty,psz,psx,psy
        xplits = unfold_patches(x_current, ps)    
        ntz = xplits.shape[3]
        ntx = xplits.shape[4]
        nty = xplits.shape[5]
               
        xplits = rearrange(xplits, 'T B C ntz ntx nty psz psx psy -> T B C (ntz ntx nty) psz psx psy')
        x_current = gather_patches(xplits, II_t)    #T, B_current, C, psz, psx, psy, for next round 
        #B_current (=sum_i=0^{B_parent} k_i)

        #keep track of refined patches location and size
        #B0(=B) -> B1 (=sum_i=0^{B-1} k_i) -> B2 (=sum_i=0^{B1-1} k_i)->...
        valid = II_t >= 0    
        if len(blockdict_list) == 0:
            parent_off = torch.zeros(II_t.shape[0], 3, dtype=torch.long, device=II_t.device)  # [B_parent,3]
        else:
            # offset is [B_parent,3] from previous iteration’s kept patches
            parent_off = offset.expand(-1, II_t.size(1), -1)  # [B_parent,k,3]


        #unravel flat ids -> (iz,ix,iy)
        stride_x=nty
        stride_z=ntx*nty

        iz=II_t//stride_z
        rem=II_t%stride_z
        ix=rem//stride_x
        iy=rem%stride_x

        psz, psx, psy = ps

        #start coords in absolute indices
        #parent_off[...,0]=z, [...,1]=x, [...,2]=y
        start_z = iz * psz + parent_off[..., 0:1]
        start_x = ix * psx + parent_off[..., 1:2]
        start_y = iy * psy + parent_off[..., 2:3]

        coords = torch.cat((start_z, start_x, start_y), dim=-1) #[B_parent,k,3]
        coords[~valid] = -1 

        keep_mask = valid.flatten()
        offset = coords.reshape(-1, 3)[keep_mask] #[B_current,3]

        blockdict_list.append(
            {
            'zxy_start':offset, #B_current, 3
            'Lzxy': ps
            }
        )
        
    #refineind: index tensors (from coarse to fine),
    # [tensor([B0, k0]),tensor([B1, k1]), ..., tensor([B{nlevel-1}, k{nlevel-1}])]           
    return refineind, blockdict_list

def get_top_variance_patchids(patch_size, xdata, gammaref, refine_ratio, full=True):
        if len(patch_size)>=3:
           return get_top_variance_patchids_new(patch_size, xdata, gammaref, refine_ratio, full=full)
        
        #T,B,C,D,H,W
        B, space_dims= xdata.shape[1], xdata.shape[3:]
        ps = patch_size[-1]
        ntokendim=[]
        #psz, psx, psy
        for idim, dim in enumerate(space_dims):
            ntokendim.append(dim//ps[idim])
        num_tokens=reduce(mul, ntokendim)
        II_topk = torch.empty(num_tokens, dtype=torch.int).fill_(-1)
        #T,B,C,D,H,W-->T,B,C,ntz,ntx,nty,psz,psx,psy->B,C,ntz,ntx,nty->B,ntz,nty,nty
        variance = xdata.unfold(3,ps[0],ps[0]).unfold(4,ps[1],ps[1]).unfold(5,ps[2],ps[2]).var(dim=(0,-3,-2,-1)).mean(dim=1)
        assert ntokendim==list(variance.shape)[1:]
        variance = rearrange(variance, 'B ntz ntx nty -> B (ntz ntx nty)')
        if gammaref is None:
            nrefines = int(refine_ratio * num_tokens)
            _, II_t = variance.topk(nrefines, dim=1)
        else:
            varmax, _ = variance.max(dim=1, keepdim=True)
            b = variance > varmax*gammaref
            II_t=mask_to_indices(b)
        idx_pad = torch.full((B, num_tokens if full else II_t.shape[1]), -1, dtype=torch.long, device=xdata.device)
        idx_pad[:,:II_t.shape[1]]=II_t
        return idx_pad

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
