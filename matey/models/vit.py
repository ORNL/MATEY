# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from .spacetime_modules import SpaceTimeBlock_all2all
from .time_modules import FourierEmbedding, PositionalEmbedding, Linear
from torch.nn.functional import silu
from .basemodel import BaseModel
from ..data_utils.shared_utils import normalize_spatiotemporal_persample, get_top_variance_patchids, normalize_spatiotemporal_persample_graph
from ..utils import ForwardOptionsBase, TrainOptionsBase, densenodes_to_graphnodes
from typing import Optional

def build_vit(params):
    """ Builds model from parameter file.
    'all2all'- spatiotemporal toekens all together
    sts_model: when True, we use two separte avit modules for coarse and refined tokens, respectively
    sts_train:
                when True, we use loss function with two parts: l_coarse/base + l_total, so that the coarse ViT approximates true solutions directly as well
    leadtime_max: when larger than 1, we use a `ltimeMLP` NN module to incoporate the impact of leadtime
    cond_input: when True, the model uses an additional inputs (scalar) to condition the predictions
    """
    model = ViT_all2all(tokenizer_heads=params.tokenizer_heads,
                     embed_dim=params.embed_dim,
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
                     replace_patch=getattr(params, 'replace_patch', True),
                     hierarchical=getattr(params, 'hierarchical', None),
                     use_linear=getattr(params, 'use_linear', False),
                     diffusion=getattr(params, 'diffusion', False)
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
                 drop_path=.2, sts_train=False, sts_model=False, leadtime=False, cond_input=False, n_steps=1, bias_type="none", replace_patch=True, SR_ratio=[1,1,1], hierarchical=None, use_linear=False, diffusion=False):
        super().__init__(tokenizer_heads=tokenizer_heads, n_states=n_states, n_states_cond=n_states_cond, embed_dim=embed_dim, leadtime=leadtime,
                         cond_input=cond_input, n_steps=n_steps, bias_type=bias_type,SR_ratio=SR_ratio, hierarchical=hierarchical, use_linear=use_linear)
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
        self.n_steps=n_steps
        self.processor_blocks=processor_blocks
        self.replace_patch=replace_patch
        assert not (self.replace_patch and self.sts_model)
        
        self.diffusion=diffusion
        if self.diffusion:
            #from https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L269
            #FIXME: @Paul, currently place holder pls check the proper setting of these variables
            model_channels      = 128          # Base multiplier for the number of channels.
            channel_mult_emb    = 4            # Multiplier for the dimensionality of the embedding vector.
            embedding_type      = 'positional' # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
            channel_mult_noise  = 1            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
            emb_channels = model_channels * channel_mult_emb
            noise_channels = model_channels * channel_mult_noise
            init = dict(init_mode='xavier_uniform')
            
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
            self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            # self.map_layer1 = Linear(in_features=emb_channels, out_features=embed_dim, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

            # self.affine = nn.ModuleDict({})
            # for imod in range(self.nhlevels):
            #     self.affine[str(imod)] = Linear(in_features=emb_channels, out_features=embed_dim, **init)
            self.affine = Linear(in_features=emb_channels, out_features=embed_dim, **init)

            # self.skip_projection = nn.ModuleDict({})
            # for imod in range(self.nhlevels):
            #     self.skip_projection[str(imod)] = Linear(in_features=embed_dim*2, out_features=embed_dim, **init)
            self.skip_projection = Linear(in_features=embed_dim*2, out_features=embed_dim, **init)

    def expand_sts_model(self):
        """ Appends addition sts blocks"""
        with torch.no_grad():
            self.sts_model=True
            self.blocks_sts = nn.ModuleList([SpaceTimeBlock_all2all(self.embed_dim, self.num_heads, drop_path=self.dp[i])
                            for i in range(self.processor_blocks)])

    def add_sts_model(self, xbase, patch_ids, x_local, bcs, tkhead_name, leadtime=None, t_pos_area=None, ilevel=0):
        #[T, B, C, D, H, W]
        T = xbase.shape[0]
        space_dims = xbase.shape[3:]
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
        if self.posbias[ilevel] is not None:
            posbias = self.posbias[ilevel](t_pos_area, use_zpos=True if space_dims[0]>1 else False) #b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> b c (t L)')
            x_local = x_local + posbias
        #FIXME: assume bcs always 0 for local patches
        #for iblk, blk in enumerate(self.blocks_sts):
        for iblk, blk in enumerate(self.blocks):
            if iblk==0:
                x_local = blk(x_local, bcs=bcs*0.0, leadtime=leadtime) 
            else:
                x_local = blk(x_local, bcs=bcs*0.0, leadtime=None)
                
        #self.debug_nan(x_local, message="x_local attention block")
        # Decode -
        x_local = rearrange(x_local, 'nrfb c (t d h w) -> (t nrfb) c d h w', t=T, d=ntokenrefdim[0], h=ntokenrefdim[1], w=ntokenrefdim[2])
        x_local = debed_ensemble[0](x_local) #, state_labels[0])
        x_local = rearrange(x_local, '(t nrfb) c d h w -> nrfb t c d h w', t=T)
        x = self.add_localpatches(xbase, x_local, patch_ids, ntokendim)
        return x

    def forward(self, data, state_labels, bcs, opts: ForwardOptionsBase, train_opts: Optional[TrainOptionsBase]=None):
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
        isgraph=opts.isgraph
        field_labels_out=opts.field_labels_out
        sigma = getattr(opts, 'sigma', None)
        diffusion_cond = getattr(opts, 'diffusion_cond', None)
        ##################################################################
        conditioning = (cond_dict != None and bool(cond_dict) and self.conditioning)

        if field_labels_out is None:
            field_labels_out = state_labels
        
        if self.diffusion:
            emb = self.map_noise(sigma)
            # print(f'noise_labels shape: {sigma.shape}, emb shape: {emb.shape}')
            emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
            # print(f'after swap emb shape: {emb.shape}')
            emb = silu(self.map_layer0(emb))
            # print(f'after first layer emb shape: {emb.shape}')
            emb = silu(self.map_layer1(emb)) #in shape
            # print(f'after second layer emb shape: {emb.shape}')
            emb = self.affine(emb)
            # print(f'after affine emb shape: {emb.shape}')
            if diffusion_cond is not None:
                diffusion_cond = rearrange(diffusion_cond, 'b t c d h w -> t b c d h w')

        if isgraph:
            x = data.x#[nnodes, T, C]
            edge_index = data.edge_index #
            batch = data.batch ##[N_total]
            T = x.shape[1] 
            x, data_mean, data_std = normalize_spatiotemporal_persample_graph(x, batch) #node features, mean_g:[G,C], std_g:[G,C]
            refineind=None
            x = (x, batch, edge_index)
        else:
            x = data
            #T,B,C,D,H,W
            T, _, _, D, H, W = x.shape
            if refine_ratio is None and  self.tokenizer_heads_gammaref[tkhead_name] is None:
                refineind=None
            else:
                refineind = get_top_variance_patchids(self.tokenizer_heads_params[tkhead_name], x, self.tokenizer_heads_gammaref[tkhead_name], refine_ratio)
            #self.debug_nan(x, message="input")
            x, data_mean, data_std = normalize_spatiotemporal_persample(x)
        #self.debug_nan(x, message="input after normalization")
        ################################################################################
        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP[imod](leadtime)
        else:
            leadtime=None
        if self.cond_input and cond_input is not None:
            leadtime = self.inconMLP[imod](cond_input) if leadtime is None else leadtime+self.inconMLP[imod](cond_input)
        ########Encode and get patch sequences [B, C_emb, T*ntoken_len_tot]########
        if  self.sts_model:
            assert not isgraph, "Not set sts_model yet"
            #x_padding: coarse tokens; x_local: refined local tokens
            x_padding, patch_ids, _, _, x_local, leadtime_local, tposarea_padding, tposarea_local = self.get_patchsequence(x, state_labels, tkhead_name, refineind=refineind, leadtime=leadtime, blockdict=blockdict)
            mask_padding = None
            x_local = rearrange(x_local, 'nrfb t c dhw_sts -> nrfb c (t dhw_sts)')
        else:
            x_padding, patch_ids, patch_ids_ref, mask_padding, _, _, tposarea_padding, _ = self.get_patchsequence(x, state_labels, tkhead_name, refineind=refineind, blockdict=blockdict, ilevel=imod, isgraph=isgraph)
        x_padding = rearrange(x_padding, 't b c ntoken_tot -> b c (t ntoken_tot)')

        if diffusion_cond is not None:
            diffusion_cond, _, _, _, _, _, _, _ = self.get_patchsequence(diffusion_cond, state_labels, tkhead_name, refineind=refineind, blockdict=blockdict, ilevel=imod, isgraph=isgraph)
            diffusion_cond = rearrange(diffusion_cond, 't b c ntoken_tot -> b c (t ntoken_tot)')

        # if self.diffusion:
        #     x_padding = x_padding + emb.unsqueeze(-1)

        # Repeat the steps for conditioning if present
        if conditioning:
            assert self.sts_model == False
            assert refineind == None
            assert not isgraph, "Not set conditioning yet"
            c, _, _, _, _, _, _, _ = self.get_patchsequence(cond_dict["fields"], cond_dict["labels"], tkhead_name, refineind=refineind, blockdict=blockdict, ilevel=imod, conditioning=conditioning)
            c = rearrange(c, 't b c ntoken_tot -> b c (t ntoken_tot)')
        ################################################################################
        if self.posbias[imod] is not None and tposarea_padding is not None:
            posbias = self.posbias[imod](tposarea_padding, mask_padding=mask_padding, use_zpos=True if D>1 else False) #b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> b c (t L)')
            x_padding = x_padding + posbias
        ######## Process ########
        #only send mask if mask_padding indicates padding tokens
        mask4attblk = None if (mask_padding is not None and mask_padding.all()) else mask_padding
        for iblk, blk in enumerate(self.blocks):
            if conditioning:
                x_padding = x_padding + c

            if iblk==0:
                if self.diffusion:
                    if diffusion_cond is not None:
                        # add diffusion_cond as additional tokens for the diffusion model to attend to
                        x_padding = torch.cat([x_padding, diffusion_cond], dim=2)
                    
                    # add noise embedding to input tokens as conditional information for diffusion model
                    x_padding = torch.cat([x_padding, emb.unsqueeze(-1)], dim=2)

                    x_input = x_padding.clone()
                
                x_padding = blk(x_padding, sequence_parallel_group=sequence_parallel_group, bcs=bcs, leadtime=leadtime, mask_padding=mask4attblk )
            else:
                if iblk==len(self.blocks)-1 and self.diffusion:
                    # for the last block, add skip connection from input tokens to facilitate training of diffusion model
                    x_padding = torch.cat([x_padding, x_input], dim=1)
                    local_batch = x_padding.shape[0]
                    x_padding = rearrange(x_padding, 'b c L -> (b L) c') 
                    x_padding = self.skip_projection(x_padding) # Residual connection from input to the last block in the level
                    x_padding = rearrange(x_padding, '(b L) c -> b c L', b = local_batch) 
                x_padding = blk(x_padding, sequence_parallel_group=sequence_parallel_group, bcs=bcs, leadtime=None, mask_padding=mask4attblk)
        #self.debug_nan(x_padding, message="attention block")
        ################################################################################

        if self.diffusion:
            x_padding = x_padding[:, :, :-1] # remove noise embedding from tokens after processing
        if diffusion_cond is not None:
            doubled_L = x_padding.shape[2]
            x_padding = x_padding[:, :, :doubled_L//2] # remove diffusion_cond part from tokens after processing

        x_padding = rearrange(x_padding, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T)
        ######## Decode ########
        if self.sts_model:
            xbase = self.get_spatiotemporalfromsequence(x_padding, None, None, [D, H, W], tkhead_name, ilevel=0)
            x = self.add_sts_model(xbase, patch_ids, x_local, bcs, tkhead_name, leadtime=leadtime_local, t_pos_area=tposarea_local)
        else:
            if isgraph:
                x_padding = rearrange(x_padding, 't b c ntoken_tot -> b ntoken_tot t c')
                #input:[B, Max_nodes, T, C] and mask: [B, Max_nodes]
                #output: [N_total, T, C] (only real nodes)
                x= densenodes_to_graphnodes(x_padding, mask_padding) #[nnodes, T, C]
                x_padding = (x, batch, edge_index)
                D, H, W = -1, -1, -1 #place holder

            x = self.get_spatiotemporalfromsequence(x_padding, patch_ids, patch_ids_ref, [D, H, W], tkhead_name, ilevel=imod, isgraph=isgraph)
            if isgraph:
                node_ft, batch, edge_index = x
                #node_ft: [nnodes, T, C]
                x = node_ft[:,:,field_labels_out[0]]
                N = x.shape[0]
                mask = torch.isin(state_labels[0], field_labels_out[0])
                #broadcast to node   
                mean_node = data_mean[batch].view(N, 1, -1)[:, :, mask]
                std_node  = data_std[batch].view(N, 1, -1)[:, :, mask]

                x = x * std_node + mean_node
                return x[:, -1, :] #[nnodes, C]
           
        ######### Denormalize ########
        #t b c d h w
        x = x[:,:,field_labels_out[0],...]
        x = x * data_std + data_mean 
        ################################################################################
        if train_opts is not None and train_opts.returnbase4train:
            xbase = xbase * data_std + data_mean
            return x[-1], xbase[-1]
        return x[-1]
