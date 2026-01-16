import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .spacetime_modules import SpaceTimeBlock_all2all
from .basemodel import BaseModel
from ..data_utils.shared_utils import normalize_spatiotemporal_persample, get_top_variance_patchids 
from ..data_utils.utils import construct_filterkernel, construct_filterkernel2D
from .spatial_modules import UpsampleinSpace
import sys, copy
from operator import mul
from functools import reduce
from ..utils.forward_options import ForwardOptionsBase
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
                     #leadtime=True if hasattr(params, 'leadtime_max') else False,
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
                 drop_path=.2, sts_train=False, sts_model=False, leadtime=True, bias_type="none", replace_patch=True, hierarchical=None, notransposed=False):
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
        self.module_upscale_space2D = nn.ModuleDict({})

        self.hierarchical=False
        self.datafilter_kernel=None
        if hierarchical is not None:
            self.hierarchical=True
            filtersize=hierarchical["filtersize"]
            self.datafilter_kernel=construct_filterkernel(filtersize)
            self.datafilter_kernel2D=construct_filterkernel2D(filtersize)
            self.filtersize = filtersize
            self.nhlevels = hierarchical["nlevels"]
            #finest at self.nhlevels-1; coarsest at 0; upscale_factors: upscale ratio from previous level to current level
            self.upscale_factors=[1]+[self.filtersize  for _ in range(self.nhlevels-1)] 
            
        for imod, upscalefactor in enumerate(self.upscale_factors):
            if hierarchical["fixedupsample"]:
                self.module_upscale_space[str(imod)]=nn.Upsample(scale_factor=(upscalefactor, upscalefactor, upscalefactor), mode='trilinear',align_corners=True)
                self.module_upscale_space2D[str(imod)]=nn.Upsample(scale_factor=(1, upscalefactor, upscalefactor), mode='trilinear',align_corners=True)
            elif hierarchical["linearupsample"]:
                 self.module_upscale_space[str(imod)]=torch.nn.Sequential(
                    nn.Upsample(scale_factor=(upscalefactor, upscalefactor, upscalefactor), mode='trilinear',align_corners=True),
                    nn.Conv3d(n_states, n_states, kernel_size=(upscalefactor, upscalefactor, upscalefactor), stride=1, padding="same", bias=True, padding_mode="reflect"),
                    nn.InstanceNorm3d(n_states, affine=True))
                 self.module_upscale_space2D[str(imod)]=torch.nn.Sequential(
                    nn.Upsample(scale_factor=(1, upscalefactor, upscalefactor), mode='trilinear',align_corners=True),
                    nn.Conv3d(n_states, n_states, kernel_size=(1, upscalefactor, upscalefactor), stride=1, padding="same", bias=True, padding_mode="reflect"),
                    nn.InstanceNorm3d(n_states, affine=True))
            else:
                #self.module_upscale[str(imod)]=PatchExpandinSpace(embed_dim, expand_ratio=upscalefactor)
                #self.module_upscale[str(imod)]=PatchUpsampleinSpace(embed_dim, expand_ratio=upscalefactor)
                self.module_upscale_space[str(imod)]=UpsampleinSpace(patch_size=[upscalefactor, upscalefactor, upscalefactor], channels=n_states)
                self.module_upscale_space2D[str(imod)]=UpsampleinSpace(patch_size=[1, upscalefactor, upscalefactor], channels=n_states)
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
            kernel_size = self.filtersize
            T,B,C,D,H,W = data.shape
            data = rearrange(data, 't b c d h w -> (t b c) d h w')
            # Apply the filter
            if D==1:
                kernel = self.datafilter_kernel2D
                filtered = F.conv3d(data[:,None,:,:,:], kernel.to(data.device), stride=(1, kernel_size, kernel_size))
            else:
                kernel = self.datafilter_kernel
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
                if D==1:
                    blockdict["Ind_dim"] = [D, H//kernel_size, W//kernel_size] #total mode size 
                else:
                    blockdict["Ind_dim"] = [D//kernel_size, H//kernel_size, W//kernel_size] #total mode size 

        return filtered, blockdict  
    
    def upsampeldata(self, data, imod):
        B,C,D,H,W=data.shape
        if D==1:
            data_upsample=self.module_upscale_space2D[str(imod)](data)
        else:
            data_upsample=self.module_upscale_space[str(imod)](data)
        return data_upsample
    
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
            #print(f"Warning (sequence_factor_short): in level {ilevel}, local blocks ({(d, h, w)}) are too small to be split into {nfact}, reset to {max(1, h//4)} for preserving 4 points", flush=True)
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
        #print("Pei debugging", f"{B,C,TL}, {x.shape}, {ntokendim}, {tspace_dims}, {nfact, nfactd}, CUDA {torch.cuda.memory_allocated()/1024**3} GB")
        x = rearrange(x, '(b nd nh nw) c (t d h w) -> b c t nd nh nw d h w', 
                      b=B//(nfactd*nfact*nfact), nd=nfactd, nh=nfact, nw=nfact, d=d//nfactd, h=h//nfact, w=w//nfact)
        x = rearrange(x, 'b c t nd nh nw d h w -> b c t (nd d) (nh h) (nw w)')
        x = rearrange(x, 'b c t d h w -> b c (t d h w)')
        return x           
    
    def forward(self, x, state_labels, bcs, opts: ForwardOptionsBase):
        ##################################################################       
        #unpack arguments
        imod = opts.imod
        imod_bottom = opts.imod_bottom
        tkhead_name = opts.tkhead_name
        sequence_parallel_group = opts.sequence_parallel_group
        leadtime = opts.leadtime
        blockdict = opts.blockdict
        refine_ratio = opts.refine_ratio
        ##################################################################
        if refine_ratio is None:
            refineind=None
        else:
            raise ValueError("Adaptive tokenization is not set up/tested yet in TurbT")
        
        #imod: nhlevels-1, nhlevels-2,...,2,1,0
        if imod<self.nhlevels-1:
            x, blockdict=self.filterdata(x, blockdict=blockdict)
            opts.blockdict = blockdict
        #print(f"Pei debugging filter {imod}, {imod_bottom}, {x.shape}")   
        if imod>imod_bottom:
            opts.imod -= 1
            x_pred = self.forward(x, state_labels, bcs, opts)
        #x_input = x.clone()
        #T,B,C,D,H,W
        T, B, _, D, H, W = x.shape
        #self.debug_nan(x, message="input")
        if imod==self.nhlevels-1:
            x, data_mean, data_std = normalize_spatiotemporal_persample(x, sequence_parallel_group=sequence_parallel_group)
        #self.debug_nan(x, message="input after normalization")
        ################################################################################
        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP[imod](leadtime)
        else:
            leadtime=None
        ########Encode and get patch sequences [B, C_emb, T*ntoken_len_tot]########
        #print(f"Pei debugging imod {imod}, {imod_bottom}, {T, D, H, W}, {x.shape}")
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
        local_att = imod>imod_bottom
        if local_att:
            #each mode similar cost
            nfact=max(2**(2*(imod-imod_bottom))//blockdict["nproc_blocks"][-1], 1) if blockdict is not None else max(2**(2*(imod-imod_bottom)), 1)
            """
            #FIXME: temporary, currently hard-coded: pass as nfactor=4//ps 
            ps = self.tokenizer_ensemble_heads[imod][tkhead_name]["embed"][-1].patch_size
            nfact=4//ps[-1]
            """
            x, nfact=self.sequence_factor_short(x, imod, tkhead_name, [T, D, H, W], nfact=nfact)
            #print(f"Pei debugging imod {imod}, {T, D, H, W}, {x.shape} nfact, {nfact}")
        for iblk, blk in enumerate(self.module_blocks[str(imod)]):
            b_mod=x.shape[0]
            #print("Pei debugging", f"iblk {iblk}, imod {imod}, {T, D, H, W}, {x.shape}, {leadtime.shape}, {sequence_parallel_group}, CUDA {torch.cuda.memory_allocated()/1024**3} GB")
            if iblk==0:
                x = blk(x, sequence_parallel_group=sequence_parallel_group, bcs=bcs, leadtime=leadtime.repeat(b_mod//B, 1), mask_padding=mask_padding, local_att=local_att)
            else:
                x = blk(x, sequence_parallel_group=sequence_parallel_group, bcs=bcs, leadtime=None, mask_padding=mask_padding, local_att=local_att)
        #self.debug_nan(x_padding, message="attention block")
        if local_att:
            #nfact=2**(2*(imod-imod_bottom))//blockdict["nproc_blocks"][-1] if blockdict is not None else 2**(2*(imod-imod_bottom))
            #nfact=4//ps[-1]
            x=self.sequence_factor_long(x, imod, tkhead_name, [T, D, H, W], nfact=nfact)
        ################################################################################
        x = rearrange(x, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T)
        #################################################################################
        ######## Decode ########
        x = self.get_spatiotemporalfromsequence(x, patch_ids, patch_ids_ref, [D, H, W], tkhead_name, ilevel=imod)
        ########upsampling######
        x_correct = x[-1]
        del x
        if imod>imod_bottom:
            x_filter =self.filterdata(x_correct[None,...])[0][-1]
            #filtered_eps=self.module_upscale_space[str(imod)](x_filter)#a full set of variables for all systems
            filtered_eps=self.upsampeldata(x_filter, imod)
            x_correct = x_correct - filtered_eps
            #x_pred=(x_pred-data_mean[-1])/data_std[-1]#a subset of variables for a specific system
            #x_pred=self.module_upscale_space[str(imod)](x_pred)
            x_pred=self.upsampeldata(x_pred, imod)
            #prediction at current level = Refine(pred from previous level) + prediction at current level
            #x_correct[:,var_index,...] = x_pred + x_correct[:,var_index,...] 
            x_correct = x_correct + x_pred 
        if imod==self.nhlevels-1:
            x_correct=x_correct[:,state_labels[0],...] * data_std[-1] + data_mean[-1]
        #since no T dim: b c d h w
        #x_correct=x_correct[:,var_index,...] * data_std[-1] + data_mean[-1]
        return x_correct #B,C_all,D,H,W for imod<nlevels-1; B,C_sys,D,H,W
     


        


