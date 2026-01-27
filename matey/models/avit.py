# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from .spacetime_modules import SpaceTimeBlock
from .basemodel import BaseModel
from ..data_utils.shared_utils import normalize_spatiotemporal_persample, get_top_variance_patchids
from ..utils.forward_options import ForwardOptionsBase, TrainOptionsBase
from typing import Optional

def build_avit(params):
    """avit model
    sts_model: when True, we use two separte avit modules for coarse and refined tokens, respectively
    sts_train:
             when True, we use loss function with two parts: l_coarse/base + l_total, so that the coarse ViT approximates true solutions directly as well
    leadtime_max: when larger than 1, we use a `ltimeMLP` NN module to incoporate the impact of leadtime
    cond_input: when True, the model uses an additional inputs (scalar) to condition the predictions
    """
    model = AViT(tokenizer_heads=params.tokenizer_heads,
                embed_dim=params.embed_dim,
                space_type=params.space_type,
                time_type=params.time_type,
                num_heads=params.num_heads,
                processor_blocks=params.processor_blocks,
                n_states=params.n_states,
                n_states_cond=params.n_states_cond if hasattr(params, 'n_states_cond') else None,
                SR_ratio=getattr(params, 'SR_ratio', [1,1,1]),
                sts_model= getattr(params, 'sts_model', False),
                sts_train=getattr(params, 'sts_train', False),
                leadtime=hasattr(params, "leadtime_max") and params.leadtime_max >= 0,
                cond_input=getattr(params,'supportdata', False),
                n_steps=params.n_steps,
                bias_type=params.bias_type,
                hierarchical=getattr(params, 'hierarchical', None)
                )
    return model

class AViT(BaseModel):
    """
    Naive model that interweaves spatial and temporal attention blocks. Temporal attention
    acts only on the time dimension.

    Args:
        patch_size (tuple): Size of the input patch
        embed_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.
    """
    def __init__(self, tokenizer_heads=None, embed_dim=768,  space_type="axial_attention", time_type="attention", num_heads=12, processor_blocks=8, n_states=6, n_states_cond=None,
                drop_path=.2, sts_train=False, sts_model=False, leadtime=False, cond_input=False, n_steps=1, bias_type="none", SR_ratio=[1,1,1], hierarchical=None):
        super().__init__(tokenizer_heads=tokenizer_heads, n_states=n_states, n_states_cond=n_states_cond, embed_dim=embed_dim, leadtime=leadtime,
                         cond_input=cond_input, n_steps=n_steps, bias_type=bias_type, SR_ratio=SR_ratio, hierarchical=hierarchical)
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)

        #FIXME: need to test for other attention types under avit later
        assert space_type=="axial_attention" and time_type=="attention"

        self.blocks = nn.ModuleList([SpaceTimeBlock(space_type, time_type, embed_dim, num_heads, drop_path=self.dp[i])
                                     for i in range(processor_blocks)])
        self.sts_model=sts_model
        #if self.sts_model:
        #    self.blocks_sts = nn.ModuleList([SpaceTimeBlock(space_type, time_type, embed_dim, num_heads, drop_path=self.dp[i])
        #                    for i in range(processor_blocks)])
        self.sts_train = sts_train

        self.num_heads=num_heads
        self.n_steps=n_steps
        self.processor_blocks=processor_blocks
        self.space_type=space_type
        self.time_type=time_type

    def expand_sts_model(self):
        """ Appends addition sts blocks"""
        with torch.no_grad():
            self.sts_model=True
            self.blocks_sts = nn.ModuleList([SpaceTimeBlock(self.space_type, self.time_type, self.embed_dim, self.num_heads, drop_path=self.dp[i])
                            for i in range(self.processor_blocks)])

    def add_sts_model(self, x, x_pre, state_labels, bcs, tkhead_name, leadtime=None, refineind=None, blockdict=None, imod=0):
        #T,B,C,D,H,W
        T = x_pre.shape[0]
        space_dims = x.shape[3:]
        embed_ensemble = self.tokenizer_ensemble_heads[imod][tkhead_name]["embed"]
        debed_ensemble = self.tokenizer_ensemble_heads[imod][tkhead_name]["debed"]
        if len(self.tokenizer_heads_params[tkhead_name])>2 or refineind is None:
            #FIXME: no adaptive yet for more than 2 levels
            xlist = []
            for ilevel in range(len(self.tokenizer_heads_params[tkhead_name])-1):
                xin = self.get_structured_sequence(x_pre, ilevel, self.tokenizer_ensemble_heads[imod][tkhead_name]["embed"]) # xin in shape [T, B, C_emb, ntoken_z,  ntoken_x, ntoken_y]
                # [B, T, ntoken_z, ntoken_x, ntoken_y, 5]
                t_pos_area, _=self.get_t_pos_area(x_pre, ilevel, tkhead_name, blockdict=blockdict, ilevel=imod)
                if self.posbias[imod] is not None:
                    posbias = self.posbias[imod](t_pos_area, use_zpos=True if space_dims[0]>1 else False) # b t d h w c->b t d h w c_emb
                    posbias=rearrange(posbias,'b t d h w c -> t b c d h w')
                    xin = xin + posbias
                # Process
                for iblk, blk in enumerate(self.blocks):
                    if iblk==0:
                        xin = blk(xin, bcs, leadtime=leadtime) #, t_pos_area=t_pos_area)
                    else:
                        xin = blk(xin, bcs, leadtime=None) 

                # predicting the last step, but leaving it like this for compatibility to causal masking
                xin = rearrange(xin, 't b c d h w -> (t b) c d h w')
                xin = debed_ensemble[ilevel](xin) #, state_labels[0])
                xin = rearrange(xin, '(t b) c d h w -> t b c d h w', t=T)
                xlist.append(xin)
            x = x + torch.stack(xlist, dim=0).sum(dim=0)
        else:
            ##############tokenizie at the fine scale##############
            #in shape [T, B, C_emb, nt_z_ref, nt_x_ref, nt_y_ref]
            x_ref = self.get_structured_sequence(x_pre, 0, self.tokenizer_ensemble_heads[imod][tkhead_name]["embed"]) 
            t_pos_area_ref, _=self.get_t_pos_area(x_pre, 0, tkhead_name, blockdict=blockdict, ilevel=imod)
            t_pos_area_ref = rearrange(t_pos_area_ref, 'b t d h w c-> b t (d h w) c')
            xlocal, t_pos_area_local, patch_ids, leadtime = self.get_chosenrefinedpatches(x_ref, refineind, t_pos_area_ref, embed_ensemble, leadtime=leadtime)
            ############################################################
            psr0, psr1, psr2 = [ps_ci//ps_refi for ps_ci, ps_refi in zip(embed_ensemble[-1].patch_size, embed_ensemble[0].patch_size)] #sts
            xlocal = rearrange(xlocal, 'nrfb t c (d h w) -> t nrfb c d h w', d=psr0, h=psr1, w=psr2)
            t_pos_area_local = rearrange(t_pos_area_local, 'nrfb t (d h w) c -> nrfb t d h w c', d=psr0, h=psr1, w=psr2)
            ############################################################
            if self.posbias[imod] is not None:
                posbias = self.posbias(t_pos_area_local, use_zpos=True if space_dims[0]>1 else False) # b t d h w c->b t d h w c_emb
                posbias=rearrange(posbias,'b t d h w c -> t b c d h w')
                xlocal = xlocal + posbias
            # Process
            #FIXME: assume bcs always 0 for local patches
            if self.sts_model:
                for iblk, blk in enumerate(self.blocks):
                    if iblk==0:
                        xlocal = blk(xlocal, bcs*0.0, leadtime=leadtime)
                    else:
                        xlocal = blk(xlocal, bcs*0.0, leadtime=None)
            #self.debug_nan(xlocal, message="xlocal attention block")
            # Decode - It would probably be better to grab the last time here since we're only
            # predicting the last step, but leaving it like this for compatibility to causal masking
            xlocal = rearrange(xlocal, 't nrfb c d h w -> (t nrfb) c d h w')
            xlocal = debed_ensemble[0](xlocal)#, state_labels[0])
            xlocal = rearrange(xlocal, '(t nrfb) c d h w -> t nrfb c d h w', t=T)
            #self.debug_nan(xlocal, message="xlocal debed_ensemble")
            xlocal = rearrange(xlocal, 't nrfb c d h w -> nrfb t c d h w')
            xlocal = xlocal[:,:,state_labels[0],...]
            ntokendim=[]
            for idim, dim in enumerate(space_dims):
                ntokendim.append(dim//embed_ensemble[-1].patch_size[idim])
            x = self.add_localpatches(x, xlocal, patch_ids, ntokendim)
        return x

    def forward(self, x, state_labels, bcs, opts: ForwardOptionsBase, train_opts: Optional[TrainOptionsBase]=None):
        ##################################################################
        #unpack arguments
        imod = opts.imod
        tkhead_name = opts.tkhead_name
        sequence_parallel_group = opts.sequence_parallel_group
        leadtime = opts.leadtime
        blockdict = opts.blockdict
        cond_dict = opts.cond_dict
        refine_ratio = opts.refine_ratio
        cond_input = opts.cond_input
        isgraph = opts.isgraph
        ##################################################################
        conditioning = (cond_dict != None and bool(cond_dict) and self.conditioning)
        assert not isgraph, "graph is not supported in AViT"
        #T,B,C,D,H,W
        T, _, _, D, _, _ = x.shape
        if self.tokenizer_heads_gammaref[tkhead_name] is None and refine_ratio is None:
            refineind=None
        else:
            refineind = get_top_variance_patchids(self.tokenizer_heads_params[tkhead_name], x, self.tokenizer_heads_gammaref[tkhead_name], refine_ratio)

        x, data_mean, data_std = normalize_spatiotemporal_persample(x)
        #self.debug_nan(x, message="input")

        #[T, B, C_emb//4, D, H, W]
        x_pre = self.get_unified_preembedding(x, state_labels, self.space_bag[imod])

        # x in shape [T, B, C_emb, ntoken_z, ntoken_x, ntoken_y]
        x = self.get_structured_sequence(x_pre, -1, self.tokenizer_ensemble_heads[imod][tkhead_name]["embed"])
        # [B, T, ntoken_z, ntoken_x, ntoken_y, 5]
        t_pos_area, _=self.get_t_pos_area(x_pre, -1, tkhead_name, blockdict=blockdict, ilevel=imod)

        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP[imod](leadtime)
        else:
            leadtime=None
        if self.cond_input and cond_input is not None:
            leadtime = self.inconMLP[imod](cond_input) if leadtime is None else leadtime+self.inconMLP[imod](cond_input)
        if self.posbias[imod] is not None:
            posbias = self.posbias[imod](t_pos_area, use_zpos=True if D>1 else False) # b t d h w c -> b t d h w c_emb
            posbias=rearrange(posbias,'b t d h w c -> t b c d h w')
            x = x + posbias

        # Repeat the steps for conditioning if present
        if conditioning:
            assert len(self.tokenizer_heads_params[tkhead_name]) <= 1 # not tested with STS

            c_pre = self.get_unified_preembedding(cond_dict["fields"], cond_dict["labels"], self.space_bag_cond[imod])
            c = self.get_structured_sequence(c_pre, -1, self.tokenizer_ensemble_heads[imod][tkhead_name]["embed_cond"])

        # Process
        for iblk, blk in enumerate(self.blocks):
            if conditioning:
                x = x + c

            if iblk==0:
                x = blk(x, bcs, leadtime=leadtime, sequence_parallel_group=sequence_parallel_group)
            else:
                x = blk(x, bcs, leadtime=None, sequence_parallel_group=sequence_parallel_group)

        #self.debug_nan(x, message="attention block")

        # Decode 
        debed_ensemble = self.tokenizer_ensemble_heads[imod][tkhead_name]["debed"]
        x = rearrange(x, 't b c d h w -> (t b) c d h w')
        x = debed_ensemble[-1](x)#, state_labels[0])
        x = rearrange(x, '(t b) c d h w -> t b c d h w', t=T)

        x = x[:,:,state_labels[0],...]
        #self.debug_nan(x, message="debed_ensemble")

        xbase = x.clone()
        if len(self.tokenizer_heads_params[tkhead_name])>1:
            x = self.add_sts_model(x, x_pre, state_labels, bcs, tkhead_name, leadtime=leadtime, refineind=refineind, blockdict=blockdict)

        # Denormalize
        x = x * data_std + data_mean # All state labels in the batch should be identical
        if train_opts is not None and train_opts.returnbase4train:
            xbase = xbase * data_std + data_mean
            return x[-1], xbase[-1]
        return x[-1] # Just return last step - now just predict delta.
