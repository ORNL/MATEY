import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from .spacetime_modules import SpaceTimeBlock_svit
from .basemodel import BaseModel
from ..data_utils.shared_utils import normalize_spatiotemporal_persample, get_top_variance_patchids, normalize_spatiotemporal_persample_graph
from ..utils import ForwardOptionsBase, TrainOptionsBase, densenodes_to_graphnodes
from typing import Optional

def build_svit(params):
    """ Builds model from parameter file.
    'time_space' - time and space sequentially, but with all2all attention inside each
    sts_model: when True, we use two separte avit modules for coarse and refined tokens, respectively
    sts_train:
                when True, we use loss function with two parts: l_coarse/base + l_total, so that the coarse ViT approximates true solutions directly as well
    leadtime_max: when larger than 1, we use a `ltimeMLP` NN module to incoporate the impact of leadtime
    cond_input: when True, the model uses an additional inputs (scalar) to condition the predictions
    """
    model = sViT_all2all(tokenizer_heads=params.tokenizer_heads,
                     embed_dim=params.embed_dim,
                     space_type=params.space_type,
                     time_type=params.time_type,
                     num_heads=params.num_heads,
                     processor_blocks=params.processor_blocks,
                     n_states=params.n_states,
                     n_states_cond=getattr(params, 'n_states_cond', None),
                     SR_ratio=getattr(params, 'SR_ratio', [1,1,1]),
                     sts_model= getattr(params, 'sts_model', False),
                     sts_train=getattr(params, 'sts_train', False),
                     leadtime=hasattr(params, "leadtime_max") and params.leadtime_max >= 0,
                     cond_input=getattr(params,'supportdata', False),
                     n_steps=params.n_steps,
                     bias_type=params.bias_type,
                     replace_patch=getattr(params, 'replace_patch', True),
                     hierarchical=getattr(params, 'hierarchical', None)
                    )
    return model

class sViT_all2all(BaseModel):
    """
    time and space sequentially, but with all2all attention inside each
    Args:
        patch_size (tuple): Size of the input patch
        embed_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.
        sts_f
    """
    def __init__(self, tokenizer_heads=None, embed_dim=768, space_type="all2all", time_type="all2all", num_heads=12, processor_blocks=8, n_states=6, n_states_cond=None,
                 drop_path=.2, sts_train=False, sts_model=False, leadtime=False, cond_input=False, n_steps=1, bias_type="none", replace_patch=True, SR_ratio=[1,1,1], hierarchical=None):
        super().__init__(tokenizer_heads=tokenizer_heads,  n_states=n_states, n_states_cond=n_states_cond, embed_dim=embed_dim, leadtime=leadtime,
                         cond_input=cond_input, n_steps=n_steps, bias_type=bias_type, SR_ratio=SR_ratio, hierarchical=hierarchical)
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)

        self.blocks = nn.ModuleList([SpaceTimeBlock_svit(space_type, time_type, embed_dim, num_heads, drop_path=self.dp[i])
                                     for i in range(processor_blocks)])
        self.sts_model=sts_model
        #if self.sts_model:
        #    self.blocks_sts = nn.ModuleList([SpaceTimeBlock_svit(space_type, time_type, embed_dim, num_heads, drop_path=self.dp[i])
        #                                     for i in range(processor_blocks)])
        self.sts_train = sts_train

        self.num_heads=num_heads
        self.n_steps=n_steps
        self.processor_blocks=processor_blocks
        self.space_type=space_type
        self.time_type=time_type
        self.replace_patch=replace_patch
        assert not (self.replace_patch and self.sts_model)

    def expand_sts_model(self):
        """ Appends addition sts blocks"""
        with torch.no_grad():
            self.sts_model=True
            self.blocks_sts = nn.ModuleList([SpaceTimeBlock_svit(self.space_type, self.time_type, self.embed_dim, self.num_heads, drop_path=self.dp[i])
                                             for i in range(self.processor_blocks)])

    def add_sts_model(self, xbase, patch_ids, x_local, bcs, tkhead_name, leadtime=None, t_pos_area=None, ilevel=0):
        #T,B,C,D,H,W
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
            posbias = self.posbias[ilevel](t_pos_area, use_zpos=True if space_dims[0]>1 else False) # b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> t b c L')
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
        x_local = rearrange(x_local, 't nrfb c (d h w) -> (t nrfb) c d h w', d=ntokenrefdim[0], h=ntokenrefdim[1], w=ntokenrefdim[2])
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
        ##################################################################
        conditioning = (cond_dict != None and bool(cond_dict) and self.conditioning)

        if field_labels_out is None:
            field_labels_out = state_labels

        if isgraph:
            x = data.x#nnodes, T, C
            edge_index = data.edge_index #
            batch = data.batch ##[N_total]
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
                refineind = get_top_variance_patchids(self.tokenizer_heads_params[tkhead_name], x,  self.tokenizer_heads_gammaref[tkhead_name], refine_ratio)
            #self.debug_nan(x, message="input")
            x, data_mean, data_std = normalize_spatiotemporal_persample(x)
        ################################################################################
        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP[imod](leadtime)
        else:
            leadtime=None
        if self.cond_input and cond_input is not None:
            leadtime = self.inconMLP[imod](cond_input) if leadtime is None else leadtime+self.inconMLP[imod](cond_input)
        ########Encode and get patch sequences [T, B, C_emb, ntoken_len_tot]########
        if  self.sts_model:
            assert not isgraph, "Not set sts_model yet"
            #x_padding: coarse tokens; x_local: refined local tokens
            x_padding, patch_ids, _, _, x_local, leadtime_local, tposarea_padding, tposarea_local = self.get_patchsequence(x, state_labels, tkhead_name, refineind=refineind, leadtime=leadtime, blockdict=blockdict)
            mask_padding = None
            x_local = rearrange(x_local, 'nrfb t c dhw_sts -> t nrfb c dhw_sts')
        else:
            x_padding, patch_ids, patch_ids_ref, mask_padding, _, _, tposarea_padding, _ = self.get_patchsequence(x, state_labels, tkhead_name, refineind=refineind, blockdict=blockdict, ilevel=imod, isgraph=isgraph)

        # Repeat the steps for conditioning if present
        if conditioning:
            assert self.sts_model == False
            assert refineind == None
            assert not isgraph, "Not set conditioning yet"
            c, _, _, _, _, _, _, _ = self.get_patchsequence(cond_dict["fields"], cond_dict["labels"], tkhead_name, refineind=refineind, blockdict=blockdict, ilevel=imod, conditioning=conditioning)
        ################################################################################
        if self.posbias[imod] is not None and tposarea_padding is not None:
            posbias = self.posbias[imod](tposarea_padding, mask_padding=mask_padding, use_zpos=True if D>1 else False) # b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> t b c L')
            x_padding = x_padding + posbias
        ######## Process ########
        #only send mask if mask_padding indicates padding tokens
        mask4attblk = None if (mask_padding is not None and mask_padding.all()) else mask_padding
        for iblk, blk in enumerate(self.blocks):
            if conditioning:
                x_padding = x_padding + c

            if iblk==0:
                x_padding = blk(x_padding, bcs, sequence_parallel_group=sequence_parallel_group, leadtime=leadtime, mask_padding=mask4attblk)
            else:
                x_padding = blk(x_padding, bcs, sequence_parallel_group=sequence_parallel_group, leadtime=None, mask_padding=mask4attblk)
        #self.debug_nan(x_padding, message="attention block")
        ################################################################################
        ######## Decode ########
        if self.sts_model:
            xbase = self.get_spatiotemporalfromsequence(x_padding, None, None, [D, H, W], tkhead_name)
            x = self.add_sts_model(xbase, patch_ids, x_local, bcs, tkhead_name, leadtime=leadtime_local, t_pos_area=tposarea_local)
        else:
            if isgraph:
                x_padding = rearrange(x_padding, 't b c ntoken_tot -> b ntoken_tot t c')
                #input:[B, Max_nodes, T, C] and mask: [B, Max_nodes]
                # #output: [N_total, T, C] (only real nodes)
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
                mean_node = data_mean[batch].view(N, 1, -1)[:, :, mask]#broadcast to node 
                std_node  = data_std[batch].view(N, 1, -1)[:, :, mask]

                x = x * std_node + mean_node
                return x[:, -1, :] #[nnodes, C]
        ######### Denormalize ########
        x = x[:,:,field_labels_out[0],...]
        x = x * data_std + data_mean 
        ################################################################################
        if train_opts is not None and train_opts.returnbase4train:
            xbase = xbase * data_std + data_mean
            return x[-1], xbase[-1]
        return x[-1]


