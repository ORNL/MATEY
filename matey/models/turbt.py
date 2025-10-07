import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .spacetime_modules import SpaceTimeBlock_all2all
from .basemodel import BaseModel
from ..data_utils.shared_utils import normalize_spatiotemporal_persample
from ..data_utils.utils import construct_filterkernel
from .spatial_modules import UpsampleinSpace, RMSInstanceNormSpace


import sys, copy
from operator import mul
from functools import reduce
import torch.distributed as dist

def build_turbt(params):
    """ Builds model from parameter file.
    'all2all'- spatiotemporal toekens all together
    sts_model: when True, we use two separte avit modules for coarse and refined tokens, respectively
    sts_train:
                when True, we use loss function with two parts: l_coarse/base + l_total, so that the coarse ViT approximates true solutions directly as well
    leadtime_max: when larger than 1, we use a `ltimeMLP` NN module to incoporate the impact of leadtime
    """
    model = TurbT(tokenizer_heads=params.tokenizer_heads,
                     embed_dim=params.embed_dim,
                     num_heads=params.num_heads,
                     processor_blocks=params.processor_blocks,
                     n_states=params.n_states,
                     sts_model=params.sts_model if hasattr(params, 'sts_model') else False,
                     sts_train=params.sts_train if hasattr(params, 'sts_train') else False,
                     leadtime=True if hasattr(params, 'leadtime_max') and params.leadtime_max>1 else False,
                     bias_type=params.bias_type,
                     replace_patch=params.replace_patch if hasattr(params, 'replace_patch') else True,
                     hierarchical=params.hierarchical if hasattr(params, 'hierarchical') else None,
                     notransposed=params.notransposed if hasattr(params, 'notransposed') else False
                    )
    return model

class TurbT(BaseModel):
    """
    Naive model that interweaves spatial and temporal attention blocks. Temporal attention
    acts only on the time dimension.

    Args:
        patch_size (tuple): Size of the input patch
        embed_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.
        sts_f
    """
    def __init__(self, tokenizer_heads=None, embed_dim=768,  num_heads=12, processor_blocks=8, n_states=6,
                 drop_path=.2, sts_train=False, sts_model=False, leadtime=False, bias_type="none", replace_patch=True, hierarchical=None, notransposed=False):
        super().__init__(tokenizer_heads=tokenizer_heads, n_states=n_states,  embed_dim=embed_dim, leadtime=leadtime, bias_type=bias_type, hierarchical=hierarchical, 
                         notransposed=notransposed, nlevels=hierarchical["nlevels"] if hierarchical is not None else 1)
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.module_blocks=nn.ModuleDict({})
        self.sts_model=sts_model
        self.sts_train = sts_train

        self.num_heads=num_heads
        self.processor_blocks=processor_blocks
        self.replace_patch=replace_patch
        assert not (self.replace_patch and self.sts_model)

        self.upscale_factors=[1]
        self.module_upscale = nn.ModuleDict({})
        self.module_upscale_space = nn.ModuleDict({})

        self.hierarchical=False
        self.datafilter_kernel=None
        if hierarchical is not None:
            self.hierarchical=True
            filtersize=hierarchical["filtersize"]
            self.datafilter_kernel=construct_filterkernel(filtersize)
            self.filtersize = filtersize
            self.nhlevels = hierarchical["nlevels"]
            #finest at self.nhlevels-1; coarsest at 0; upscale_factors: upscale ratio from previous level to current level
            self.upscale_factors=[1]+[self.filtersize  for _ in range(self.nhlevels-1)] 
            
        for imod, upscalefactor in enumerate(self.upscale_factors):
            if hierarchical["fixedupsample"]:
                self.module_upscale_space[str(imod)]=nn.Upsample(scale_factor=(upscalefactor, upscalefactor, upscalefactor), mode='trilinear',align_corners=True)
            elif hierarchical["linearupsample"]:
                 self.module_upscale_space[str(imod)]=torch.nn.Sequential(
                    nn.Upsample(scale_factor=(upscalefactor, upscalefactor, upscalefactor), mode='trilinear',align_corners=True),
                    nn.Conv3d(n_states, n_states, kernel_size=(upscalefactor, upscalefactor, upscalefactor), stride=1, padding="same", bias=True, padding_mode="reflect"),
                    RMSInstanceNormSpace(n_states, affine=True))
            else:
                #self.module_upscale[str(imod)]=PatchExpandinSpace(embed_dim, expand_ratio=upscalefactor)
                #self.module_upscale[str(imod)]=PatchUpsampleinSpace(embed_dim, expand_ratio=upscalefactor)
                self.module_upscale_space[str(imod)]=UpsampleinSpace(patch_size=[upscalefactor, upscalefactor, upscalefactor], channels=n_states)
            if imod ==0:
                self.module_blocks[str(imod)] = nn.ModuleList([SpaceTimeBlock_all2all(embed_dim, num_heads,drop_path=self.dp[i])
                                        #for i in range(processor_blocks)])
                                        for i in range(processor_blocks//self.nhlevels)])
            else:
                #FIXME: figure out how to do local attention in 3D physical space for corrections
                self.module_blocks[str(imod)] = nn.ModuleList([SpaceTimeBlock_all2all(embed_dim, num_heads,drop_path=self.dp[i])
                                        for i in range(processor_blocks//self.nhlevels)])
                
    def filterdata(self, data, blockdict=None):
        #T,B,C,D,H,W
        assert data.ndim==6, f"unkown tensor shape in filter_data, {data.shape}"
        with torch.no_grad():
            kernel = self.datafilter_kernel
            kernel_size = self.filtersize
            T,B,C,D,H,W = data.shape
            data = rearrange(data, 't b c d h w -> (t b c) d h w')
            # Apply the filter
            filtered = F.conv3d(data[:,None,:,:,:], kernel.to(data.device), stride=kernel_size)
            filtered = rearrange(filtered, '(t b c) c1 d h w -> t b (c c1) d h w', t=T, b=B, c=C) #c1=1
            """
            blockdict={}
            blockdict["Lzxy"] = [Lz/nproc_blocks[0], Lx/nproc_blocks[1], Ly/nproc_blocks[2]]
            blockdict["nproc_blocks"] = nproc_blocks
            blockdict["Ind_dim"] = [Dloc, Hloc, Wloc]
            idz, idx, idy = blockIDs[self.group_rank,:]
            blockdict["Ind_start"] = [idz*Dloc, idx*Hloc, idy*Wloc]
            Lz_loc, Lx_loc, Ly_loc = blockdict["Lzxy"]
            blockdict["zxy_start"]=[Lz_start+idz*Lz_loc, Lx_start+idx*Lx_loc, Ly_start+idy*Ly_loc]
            """
            if blockdict is not None:
                assert [D,H,W] == blockdict["Ind_dim"], f"(D,H,W),{(D,H,W)}, {blockdict['Ind_dim']}"
                blockdict["Ind_dim"] = [D//kernel_size, H//kernel_size, W//kernel_size] #total mode size 

        return filtered, blockdict  
    
    def sequence_factor_short(self, x, ilevel, tkhead_name, tspace_dims, nfact=2):
        B, C, TL = x.shape
        embed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed"]
        ########################################################################
        ntokendim   =[]
        ps_c  = embed_ensemble[-1].patch_size
        for idim, dim in enumerate(tspace_dims[1:]):
            ntokendim.append(dim//ps_c[idim])
        assert TL==tspace_dims[0]*reduce(mul, ntokendim), f"{TL}, {tspace_dims}, {ntokendim}"
        d, h, w=ntokendim
        if h//nfact<4:
            print(f"Warning (sequence_factor_short): in level {ilevel}, local blocks ({(d, h, w)}) are too small to be split into {nfact}, reset to {max(1, h//4)} for preserving 4 points", flush=True)
            nfact = max(1, h//4)
        if nfact<2:
            return x, nfact
        if d==1:
            nfactd=1
        else:
            nfactd=nfact
        ########################################################################
        x = rearrange(x, 'b c (t d h w) -> b c t d h w', t=tspace_dims[0], d=d, h=h, w=w)
        x = x.unfold(3, d//nfactd, d//nfactd).unfold(4, h//nfact, h//nfact).unfold(5, w//nfact, w//nfact) 
        #b,c,t,nfactd,nfact,nfact,d',h',w'  
        x = rearrange(x, 'b c t nd nh nw d h w -> (b nd nh nw) c (t d h w)')
        return x, nfact
    
    def sequence_factor_long(self, x, ilevel, tkhead_name, tspace_dims, nfact=2):
        if nfact<2:
            return x
        B, C, TL = x.shape
        embed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed"]
        ########################################################################
        ntokendim   =[]
        ps_c  = embed_ensemble[-1].patch_size
        for idim, dim in enumerate(tspace_dims[1:]):
            ntokendim.append(dim//ps_c[idim])
        d, h, w=ntokendim
        if d==1:
            nfactd=1
        else:
            nfactd=nfact
        assert TL*(nfactd*nfact*nfact)==tspace_dims[0]*reduce(mul, ntokendim), f"{TL}, {tspace_dims}, {ntokendim}, {nfact, nfactd}"
        ########################################################################
        x = rearrange(x, '(b nd nh nw) c (t d h w) -> b c t nd nh nw d h w', 
                      b=B//(nfactd*nfact*nfact), nd=nfactd, nh=nfact, nw=nfact, d=d//nfact, h=h//nfact, w=w//nfact)
        x = rearrange(x, 'b c t nd nh nw d h w -> b c t (nd d) (nh h) (nw w)')
        x = rearrange(x, 'b c t d h w -> b c (t d h w)')
        return x            
    
    def forward(self, x, state_labels, bcs, imod=None, sequence_parallel_group=None, leadtime=None, refineind=None, tkhead_name=None, blockdict=None):
        
        if imod<self.nhlevels-1:
            x, blockdict=self.filterdata(x, blockdict=blockdict)
        if imod>0:
            x_pred = self.forward(x, state_labels, bcs, imod=imod-1, sequence_parallel_group=sequence_parallel_group, leadtime=leadtime, 
                    refineind=refineind, tkhead_name=tkhead_name, blockdict=blockdict)
        
        max_allocated_after_reset = torch.cuda.max_memory_allocated()
        print(f"imod: {imod}, Max GPU memory allocated after reset: {max_allocated_after_reset / (1024**2):.2f} MB", flush=True)
               
        #x_input = x.clone()
        #T,B,C,D,H,W
        T, _, _, D, H, W = x.shape
        #self.debug_nan(x, message="input")
        if imod==self.nhlevels-1:
            x, data_mean, data_std = normalize_spatiotemporal_persample(x,sequence_parallel_group=sequence_parallel_group)
        #self.debug_nan(x, message="input after normalization")
        ################################################################################
        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP[imod](leadtime)
        else:
            leadtime=None
        ########Encode and get patch sequences [B, C_emb, T*ntoken_len_tot]########
        x, patch_ids, patch_ids_ref, mask_padding, _, _, tposarea_padding, _ = self.get_patchsequence(x, state_labels, tkhead_name, refineind=refineind, blockdict=blockdict, ilevel=imod)
        x = rearrange(x, 't b c ntoken_tot -> b c (t ntoken_tot)')
        ################################################################################
        use_zpos=True if D>1 else False
        if self.posbias[imod] is not None:
            posbias = self.posbias[imod](tposarea_padding, mask_padding=mask_padding, use_zpos=use_zpos) #b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> b c (t L)')
            x = x + posbias
            del posbias
        ######## Process ########
        local_att = imod>0
        if local_att:
            #each mode similar cost
            nfact=2**(2*imod)//blockdict["nproc_blocks"][-1]
            """
            #FIXME: temporary, currently hard-coded: pass as nfactor=4//ps 
            ps = self.tokenizer_ensemble_heads[imod][tkhead_name]["embed"][-1].patch_size
            nfact=4//ps[-1]
            """
            x, nfact =self.sequence_factor_short(x, imod, tkhead_name, [T, D, H, W], nfact=nfact)
        max_allocated_after_reset = torch.cuda.max_memory_allocated()
        print(f"imod: {imod}, before attention blocks: Max GPU memory allocated after reset: {max_allocated_after_reset / (1024**2):.2f} MB", flush=True)
        for iblk, blk in enumerate(self.module_blocks[str(imod)]):
            # print("Pei debugging", f"iblk {iblk}, imod {imod}, {x.shape}, CUDA {torch.cuda.memory_allocated()/1024**3} GB")
            if iblk==0:
                x = blk(x, sequence_parallel_group=sequence_parallel_group, bcs=bcs, leadtime=leadtime, mask_padding=mask_padding, local_att=local_att)
            else:
                x = blk(x, sequence_parallel_group=sequence_parallel_group, bcs=bcs, leadtime=None, mask_padding=mask_padding, local_att=local_att)
        #self.debug_nan(x_padding, message="attention block")
        max_allocated_after_reset = torch.cuda.max_memory_allocated()
        print(f"imod: {imod}, after attention. Max GPU memory allocated after reset: {max_allocated_after_reset / (1024**2):.2f} MB", flush=True)
        if local_att:
            # nfact=2**(2*imod)#//blockdict["nproc_blocks"][-1]
            #nfact=4//ps[-1]
            x=self.sequence_factor_long(x, imod, tkhead_name, [T, D, H, W], nfact=nfact)
        ################################################################################
        x = rearrange(x, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T)
        #################################################################################
        ######## Decode ########
        x = self.get_spatiotemporalfromsequence(x, patch_ids, patch_ids_ref, state_labels, [D, H, W], tkhead_name, ilevel=imod)
        max_allocated_after_reset = torch.cuda.max_memory_allocated()
        print(f"imod: {imod}, depatching. Max GPU memory allocated after reset: {max_allocated_after_reset / (1024**2):.2f} MB", flush=True)
        ########upsampling######
        x_correct = x[-1]
        del x
        if imod>0:
            x_filter =self.filterdata(x_correct[None,...])[0][-1]
            filtered_eps=self.module_upscale_space[str(imod)](x_filter)#a full set of variables for all systems
            x_correct = x_correct - filtered_eps
            #x_pred=(x_pred-data_mean[-1])/data_std[-1]#a subset of variables for a specific system
            x_pred=self.module_upscale_space[str(imod)](x_pred)
            #prediction at current level = Refine(pred from previous level) + prediction at current level
            #x_correct[:,var_index,...] = x_pred + x_correct[:,var_index,...] 
            x_correct = x_correct + x_pred 
            if imod==self.nhlevels-1:
                x_correct=x_correct[:,state_labels[0],...] * data_std[-1] + data_mean[-1]
        #since no T dim: b c d h w
        #x_correct=x_correct[:,var_index,...] * data_std[-1] + data_mean[-1]
        
        max_allocated_after_reset = torch.cuda.max_memory_allocated()
        print(f"imod: {imod}, modle end. Max GPU memory allocated after reset: {max_allocated_after_reset / (1024**2):.2f} MB", flush=True)
        
        return x_correct #B,C_all,D,H,W for imod<nlevels-1; B,C_sys,D,H,W
     


        


