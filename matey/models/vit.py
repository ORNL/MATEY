import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from .spacetime_modules import SpaceTimeBlock_all2all
from .basemodel import BaseModel
from ..data_utils.shared_utils import normalize_spatiotemporal_persample

def build_vit(params):
    """ Builds model from parameter file.
    'all2all'- spatiotemporal toekens all together
    sts_model: when True, we use two separte avit modules for coarse and refined tokens, respectively
    sts_train:
                when True, we use loss function with two parts: l_coarse/base + l_total, so that the coarse ViT approximates true solutions directly as well
    leadtime_max: when larger than 1, we use a `ltimeMLP` NN module to incoporate the impact of leadtime
    """
    model = ViT_all2all(tokenizer_heads=params.tokenizer_heads,
                     embed_dim=params.embed_dim,
                     num_heads=params.num_heads,
                     processor_blocks=params.processor_blocks,
                     n_states=params.n_states,
                     n_states_cond=params.n_states_cond if hasattr(params, 'n_states_cond') else None,
                     SR_ratio=params.SR_ratio if hasattr(params, 'SR_ratio') else [1,1,1],
                     sts_model=params.sts_model if hasattr(params, 'sts_model') else False,
                     sts_train=params.sts_train if hasattr(params, 'sts_train') else False,
                     leadtime=True if hasattr(params, 'leadtime_max') and params.leadtime_max>1 else False,
                     bias_type=params.bias_type,
                     replace_patch=params.replace_patch if hasattr(params, 'replace_patch') else True,
                     hierarchical=params.hierarchical if hasattr(params, 'hierarchical') else None
                    )
    return model

class ViT_all2all(BaseModel):
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
    def __init__(self, tokenizer_heads=None, embed_dim=768,  num_heads=12, processor_blocks=8, n_states=6, n_states_cond=None,
                 drop_path=.2, sts_train=False, sts_model=False, leadtime=False, bias_type="none", replace_patch=True, SR_ratio=[1,1,1], hierarchical=None):
        super().__init__(tokenizer_heads=tokenizer_heads, n_states=n_states, n_states_cond=n_states_cond, embed_dim=embed_dim, leadtime=leadtime, bias_type=bias_type,SR_ratio=SR_ratio, hierarchical=hierarchical)
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.blocks = nn.ModuleList([SpaceTimeBlock_all2all(embed_dim, num_heads,drop_path=self.dp[i])
                                     for i in range(processor_blocks)])
        self.sts_model=sts_model
        #if self.sts_model:
        #    self.blocks_sts = nn.ModuleList([SpaceTimeBlock_all2all(embed_dim, num_heads, drop_path=self.dp[i])
        #                    for i in range(processor_blocks)])
        self.sts_train = sts_train

        self.num_heads=num_heads
        self.processor_blocks=processor_blocks
        self.replace_patch=replace_patch
        assert not (self.replace_patch and self.sts_model)

    def expand_sts_model(self):
        """ Appends addition sts blocks"""
        with torch.no_grad():
            self.sts_model=True
            self.blocks_sts = nn.ModuleList([SpaceTimeBlock_all2all(self.embed_dim, self.num_heads, drop_path=self.dp[i])
                            for i in range(self.processor_blocks)])

    def add_sts_model(self, xbase, patch_ids, x_local, state_labels, bcs, tkhead_name, leadtime=None, t_pos_area=None, ilevel=0):
        #[T, B, C, D, H, W]
        T = xbase.shape[0]
        space_dims = x.shape[3:]
        ########################################################################
        embed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed"]
        debed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["debed"]
        ########################################################################
        #psz, psx, psy
        ntokenrefdim=[]
        ps=embed_ensemble[-1].patch_size
        ps_ref=embed_ensemble[0].patch_size
        for idim, ps_dim in enumerate(ps):
            ntokenrefdim.append(ps_dim//ps_ref[idim])
        ntokendim=[]
        for idim, dim in enumerate(space_dims):
            ntokendim.append(dim//ps[idim])
        ########################################################################
        # Process
        if self.posbias is not None:
            posbias = self.posbias(t_pos_area, use_zpos=True if space_dims[0]>1 else False) #b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> b c (t L)')
            x_local = x_local + posbias
        #FIXME: assume bcs always 0 for local patches
        #for iblk, blk in enumerate(self.blocks_sts):
        for iblk, blk in enumerate(self.blocks):
            if iblk==0:
                x_local = blk(x_local, bcs*0.0, leadtime=leadtime) 
            else:
                x_local = blk(x_local, bcs*0.0, leadtime=None)
        #self.debug_nan(x_local, message="x_local attention block")
        # Decode -
        x_local = rearrange(x_local, 'nrfb c (t d h w) -> (t nrfb) c d h w', t=T, d=ntokenrefdim[0], h=ntokenrefdim[1], w=ntokenrefdim[2])
        x_local = debed_ensemble[0](x_local, state_labels[0])
        x_local = rearrange(x_local, '(t nrfb) c d h w -> nrfb t c d h w', t=T)
        x = self.add_localpatches(xbase, x_local, patch_ids, ntokendim)
        return x

    def forward(self, x, state_labels, bcs, sequence_parallel_group=None, leadtime=None, refineind=None, returnbase4train=False, tkhead_name=None, blockdict=None, imod=0, cond_dict=None):
        conditioning = (cond_dict != None and bool(cond_dict) and self.conditioning)

        #T,B,C,D,H,W
        T, _, _, D, H, W = x.shape
        #self.debug_nan(x, message="input")
        x, data_mean, data_std = normalize_spatiotemporal_persample(x)
        #self.debug_nan(x, message="input after normalization")
        ################################################################################
        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP[imod](leadtime)
        else:
            leadtime=None
        ########Encode and get patch sequences [B, C_emb, T*ntoken_len_tot]########
        if  self.sts_model:
            raise ValueError("need to double check after multiple levels with imod")
            #x_padding: coarse tokens; x_local: refined local tokens
            x_padding, patch_ids, _, _, x_local, leadtime_local, tposarea_padding, tposarea_local = self.get_patchsequence(x, state_labels, tkhead_name, refineind=refineind, leadtime=leadtime, blockdict=blockdict)
            mask_padding = None
            x_local = rearrange(x_local, 'nrfb t c dhw_sts -> nrfb c (t dhw_sts)')
        else:
            x_padding, patch_ids, patch_ids_ref, mask_padding, _, _, tposarea_padding, _ = self.get_patchsequence(x, state_labels, tkhead_name, refineind=refineind, blockdict=blockdict, ilevel=imod)
        x_padding = rearrange(x_padding, 't b c ntoken_tot -> b c (t ntoken_tot)')

        # Repeat the steps for conditioning if present
        if conditioning:
            assert self.sts_model == False
            assert refineind == None
            c, _, _, _, _, _, _, _ = self.get_patchsequence(cond_dict["fields"], cond_dict["labels"], tkhead_name, refineind=refineind, blockdict=blockdict, ilevel=imod, conditioning=conditioning)
            c = rearrange(c, 't b c ntoken_tot -> b c (t ntoken_tot)')
        ################################################################################
        if self.posbias[imod] is not None:
            posbias = self.posbias[imod](tposarea_padding, mask_padding=mask_padding, use_zpos=True if D>1 else False) #b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> b c (t L)')
            x_padding = x_padding + posbias
        ######## Process ########
        for iblk, blk in enumerate(self.blocks):
            if conditioning:
                x_padding = x_padding + c

            if iblk==0:
                x_padding = blk(x_padding, sequence_parallel_group=sequence_parallel_group, bcs=bcs, leadtime=leadtime, mask_padding=mask_padding)
            else:
                x_padding = blk(x_padding, sequence_parallel_group=sequence_parallel_group, bcs=bcs, leadtime=None, mask_padding=mask_padding)
        #self.debug_nan(x_padding, message="attention block")
        ################################################################################
        x_padding = rearrange(x_padding, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T)
        ######## Decode ########
        if self.sts_model:
            raise ValueError("need to double check after multiple levels with imod")
            xbase = self.get_spatiotemporalfromsequence(x_padding, None, None, state_labels, [D, H, W], tkhead_name, ilevel=0)
            x = self.add_sts_model(xbase, patch_ids, x_local, state_labels, bcs, tkhead_name, leadtime=leadtime_local, t_pos_area=tposarea_local)
        else:
            x = self.get_spatiotemporalfromsequence(x_padding, patch_ids, patch_ids_ref, state_labels, [D, H, W], tkhead_name, ilevel=imod)
        ######### Denormalize ########
        #t b c d h w
        x = x[:,:,state_labels[0],...]
        x = x * data_std + data_mean 
        ################################################################################
        if returnbase4train:
            xbase = xbase * data_std + data_mean
            return x[-1], xbase[-1]
        return x[-1]
