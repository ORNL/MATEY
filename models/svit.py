import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
try:
    from spatial_modules import hMLP_stem, hMLP_output, SubsampledLinear
    from spacetime_modules import SpaceTimeBlock_svit
    from time_modules import leadtimeMLP
    from positionbias_modules import positionbias_mod
    from ..data_utils.shared_utils import normalize_spatiotemporal_persample, figure_checking
except:
    from .spatial_modules import hMLP_stem, hMLP_output, SubsampledLinear
    from .spacetime_modules import SpaceTimeBlock_svit
    from .time_modules import leadtimeMLP
    from .positionbias_modules import positionbias_mod
    from data_utils.shared_utils import normalize_spatiotemporal_persample, figure_checking
import sys
import copy

def build_svit(params):
    """ Builds model from parameter file. 
    'time_space' - time and space sequentially, but with all2all attention inside each
    sts_model: when True, we use two separte avit modules for coarse and refined tokens, respectively
    sts_train: 
                when True, we use loss function with two parts: l_coarse/base + l_total, so that the coarse ViT approximates true solutions directly as well
    leadtime_max: when larger than 1, we use a `ltimeMLP` NN module to incoporate the impact of leadtime
    """
    model = sViT_all2all(patch_size=params.patch_size,
                     embed_dim=params.embed_dim,
                     space_type=params.space_type, 
                     time_type=params.time_type, 
                     num_heads=params.num_heads, 
                     processor_blocks=params.processor_blocks,
                     n_states=params.n_states,
                     sts_model=params.sts_model if hasattr(params, 'sts_model') else False,
                     sts_train=params.sts_train if hasattr(params, 'sts_train') else False,
                     leadtime=True if hasattr(params, 'leadtime_max') and params.leadtime_max>1 else False,
                     bias_type=params.bias_type)
    return model

class sViT_all2all(nn.Module):
    """
    time and space sequentially, but with all2all attention inside each
    Args:
        patch_size (tuple): Size of the input patch
        embed_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.  
        sts_f
    """
    def __init__(self, patch_size=(16, 16), embed_dim=768, space_type="all2all", time_type="all2all", num_heads=12, processor_blocks=8, n_states=6,
                 drop_path=.2, sts_train=False, sts_model=False, leadtime=False, bias_type="none"):
        super().__init__()
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.space_bag = SubsampledLinear(n_states, embed_dim//4)
        if all(isinstance(ps, int) for ps in patch_size) and len(patch_size)==2:
            patch_size = [patch_size]
        #patches at multiple scales/sizes
        self.patch_size = patch_size
        self.token_level = len(patch_size)
        self.embed_ensemble = nn.ModuleList()
        self.debed_ensemble = nn.ModuleList()
        for ps_scale in self.patch_size:
            self.embed_ensemble.append(hMLP_stem(patch_size=ps_scale, in_chans=embed_dim//4, embed_dim=embed_dim))
            self.debed_ensemble.append(hMLP_output(patch_size=ps_scale, embed_dim=embed_dim, out_chans=n_states))

        self.leadtime=leadtime
        if self.leadtime:
            self.ltimeMLP=leadtimeMLP(hidden_dim=embed_dim)

        self.blocks = nn.ModuleList([SpaceTimeBlock_svit(space_type, time_type, embed_dim, num_heads, drop_path=self.dp[i])
                                     for i in range(processor_blocks)])
        self.sts_model=sts_model
        #if self.sts_model:
        #    self.blocks_sts = nn.ModuleList([SpaceTimeBlock_svit(space_type, time_type, embed_dim, num_heads, drop_path=self.dp[i])
        #                                     for i in range(processor_blocks)])    
        self.sts_train = sts_train 
        self.posbias = positionbias_mod(bias_type, embed_dim)

        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.processor_blocks=processor_blocks
        self.space_type=space_type
        self.time_type=time_type

    def expand_conv_projections(self, refine_resol):
        """ Appends addition conv heads"""
        with torch.no_grad():
            #patches at multiple scales/sizes
            self.patch_size = refine_resol
            self.token_level = len(refine_resol)
            embed_dim=self.debed_ensemble[0].embed_dim
            n_states=self.debed_ensemble[0].out_chans

            embed_ensemble_new = nn.ModuleList()
            debed_ensemble_new = nn.ModuleList()
            #for ps_scale in self.patch_size:
            for ilevel in range(self.token_level):
                embed_ensemble_new.append(hMLP_stem(patch_size=self.patch_size[ilevel], in_chans=embed_dim//4, embed_dim=embed_dim))
                debed_ensemble_new.append(hMLP_output(patch_size=self.patch_size[ilevel], embed_dim=embed_dim, out_chans=n_states))
           
            if self.token_level>1:
                embed_ensemble_new[-1]=self.embed_ensemble[0]
                debed_ensemble_new[-1]=self.debed_ensemble[0]    
                
            self.embed_ensemble=embed_ensemble_new
            self.debed_ensemble=debed_ensemble_new    

    def expand_sts_model(self):
        """ Appends addition sts blocks"""
        with torch.no_grad():
            self.sts_model=True
            self.blocks_sts = nn.ModuleList([SpaceTimeBlock_svit(self.space_type, self.time_type, self.embed_dim, self.num_heads, drop_path=self.dp[i])
                                             for i in range(self.processor_blocks)])                   

    def expand_projections(self, expansion_amount):
        """ Appends addition embeddings for finetuning on new data """
        with torch.no_grad():
            # Expand input projections
            temp_space_bag = SubsampledLinear(dim_in = self.space_bag.dim_in + expansion_amount, dim_out=self.space_bag.dim_out)
            temp_space_bag.weight[:, :self.space_bag.dim_in] = self.space_bag.weight
            temp_space_bag.bias = self.space_bag.bias
            self.space_bag = temp_space_bag
            # expand output projections
            for ilevel in range(self.token_level):
                out_head = nn.ConvTranspose2d(self.debed_ensemble[ilevel].embed_dim//4, self.debed_ensemble[ilevel].out_chans+expansion_amount, 
                                            kernel_size=self.debed_ensemble[ilevel].ks[0], stride=self.debed_ensemble[ilevel].ks[0])
                temp_out_kernel = out_head.weight
                temp_out_bias = out_head.bias
                temp_out_kernel[:, :self.debed_ensemble[ilevel].out_chans, :, :] = self.debed_ensemble[ilevel].out_kernel
                temp_out_bias[:self.debed_ensemble[ilevel].out_chans] = self.debed_ensemble[ilevel].out_bias
                self.debed_ensemble[ilevel].out_kernel = nn.Parameter(temp_out_kernel)
                self.debed_ensemble[ilevel].out_bias = nn.Parameter(temp_out_bias)
    
    def expand_leadtime(self, expand_leadtime, embed_dim):
        if not expand_leadtime:
            return
        self.leadtime=expand_leadtime
        self.ltimeMLP=leadtimeMLP(hidden_dim=embed_dim)

    def freeze_middle(self):
        # First just turn grad off for everything
        for param in self.parameters():
            param.requires_grad = False
        # Activate for embed/debed layers
        for param in self.space_bag.parameters():
            param.requires_grad = True
        for ilevel in range(self.token_level):
            self.debed_ensemble[ilevel].out_kernel.requires_grad = True
            self.debed_ensemble[ilevel].out_bias.requires_grad = True
    
    def freeze_processor(self):
        # First just turn grad off for everything
        for param in self.parameters():
            param.requires_grad = False
        # Activate for embed/debed layers
        for param in self.space_bag.parameters():
            param.requires_grad = True
        for ilevel in range(self.token_level):
            for param in self.debed_ensemble[ilevel].parameters():
                param.requires_grad = True
            for param in self.embed_ensemble[ilevel].parameters():
                param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def debug_nan(self,x, message=""):
        #print(message, "min, max", torch.min(x), torch.max(x))
        if torch.isinf(x).any():
            print(f"INF detected in prediction, after {message}: ",x)

        if torch.isnan(x).any():
            print(f"NAN detected in prediction, after {message}: ",x)
            for name, param in self.named_parameters():
                if param.requires_grad and torch.isnan(param.data).any():
                    print("NAN detected in model parameters: ", name, param.data.numel())
                else:
                    continue
                    #print("No NAN in model parameters: ", name, param.data.numel())
            sys.exit(-1)

    def get_structured_sequence(self, x, state_labels, embed_index):
        ## input tensor x: [t, b, c, h, w]; state_labels[b, c]
        # embed_index: tokenization at different resolutions; 
        # state_labels: variable index to consider varying datasets 
        ## and return patch sequences in shape [t, b, c_emd, ntoken_x, ntoken_y]
        T, _, _, _, _ = x.shape
        # Sparse proj
        x = rearrange(x, 't b c h w -> t b h w c')
        x = self.space_bag(x, state_labels)
        self.debug_nan(x, message="space_bag")
        x = rearrange(x, 't b h w c -> (t b) c h w')
        x = self.embed_ensemble[embed_index](x)            
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)
        self.debug_nan(x, message="embed_ensemble")
        return x
    
    def get_refined_localpatches(self, x_0, refineind, leadtime=None):
        ## input tensor x_0 in shape [t,b,c,h,w]
        #  refineind: in shape[b, ntokenx*ntokeny]; leadtime: [b, 1] 
        #   (containing importnat token ids to be refined, followed by nonimportant ones as "-1")  
        ## return 
        #       x_local: [npatches, T, C, ps0, ps1]
        #       leadtime: [npatches, 1]
        #       patch_ids: [npatches] #selected token ids with sample pos inside batch considered
        
        T, B, C, H, W = x_0.shape
        ##############x refinement###################
        ps0=self.embed_ensemble[-1].patch_size[0]
        ps1=self.embed_ensemble[-1].patch_size[1]
        ntokenx = H//ps0
        ntokeny = W//ps1
        ####################################
        refineind1d = refineind.flatten()
        #images to batches of smaller patches
        x_0 = rearrange(x_0, 't b c (ntx h) (nty w) -> (b ntx nty) t c h w', ntx=ntokenx, nty=ntokeny)
        mask = refineind1d >= 0
        batch_offsets = torch.arange(B, device=x_0.device).repeat_interleave(ntokenx*ntokeny)* ntokenx * ntokeny
        patch_ids = refineind1d[mask] + batch_offsets[mask]
        x_local = x_0[patch_ids].clone() #(nrefines_batch, T, x_0.size()[2], ps0, ps1)
        if leadtime is not None:
            leadtime = leadtime.repeat_interleave(ntokenx*ntokeny, dim=0)[mask]
        self.debug_nan(x_local, message="xlocal ")
        return x_local, patch_ids, leadtime
    
    def add_localpatches(self, x, x_local, patch_ids):
        ###inputs: 
        # x in shape [T,B,C,H,W], 
        # x_local in shape[nrefines_batch, T, C, ps0, ps1]
        # patch_ids in shape [nrefines_batch]
        ###return x in the same shape [T,B,C,H,W]
        _, _, _, H, W = x.shape
        ps0=self.embed_ensemble[-1].patch_size[0]
        ps1=self.embed_ensemble[-1].patch_size[1]
        ntokenx = H//ps0
        ntokeny = W//ps1
        #images to small patches
        x = rearrange(x, 't b c (ntx h) (nty w) -> (b ntx nty) t c h w', ntx=ntokenx, nty=ntokeny)
        x[patch_ids] = x[patch_ids] + x_local.clone()
        #small patches to images
        x = rearrange(x, '(b ntx nty) t c h w -> t b c (ntx h) (nty w)', ntx=ntokenx, nty=ntokeny)
        self.debug_nan(x, message="add_localpatches")
        return x

    def get_merge_sequences(self, x_coarse, x_local, refineind, t_pos_area, t_pos_area_local):
        ###input tensors 
        #       x_coarse: [T, B, C_emb, ntoken_coarse]
        #       x_local : [T, npatches, C_emb, ntxy_ref]
        #       refineind: in shape[b, ntokenx*ntokeny]
        #      t_pos_area:  [B, T, ntoken_coarse, 4]
        #      t_pos_area_local:  [npatches, T, ntxy_ref, 4]
        ###return tnesors
        #       x_padding: [T, B, C_emb, ntoken_len_tot]
        #       patch_ids_ref: [npatches] (ids of effective tokens in x_local)
        #       mask_padding: [B, ntoken_len_tot]
        #       t_pos_area:  [B, T, ntoken_len_tot, 4]
        #######################################################
        T, B, C_emb, ntoken_coarse = x_coarse.shape
        _, _, _, ntxy_ref = x_local.shape
        nref_tokens = torch.max(torch.sum(refineind>=0, dim=1)).item()
        ntoken_len_tot = ntoken_coarse + nref_tokens*ntxy_ref

        x_local_padding = torch.full((B*nref_tokens, T, C_emb, ntxy_ref), 1e6, device=x_local.device)
        x_padding = torch.full((T, B, C_emb, ntoken_len_tot), 1e6, device=x_coarse.device)
        t_pos_area_local_padding = torch.full((B*nref_tokens, T, ntxy_ref, 4), 1e6, device=x_local.device)
        t_pos_area_padding = torch.full((B, T, ntoken_len_tot, 4), 1e6, device=x_coarse.device)

        mask_padding = torch.full((B, ntoken_len_tot), True, device=x_coarse.device)

        refineind1d = refineind[:, :nref_tokens].flatten()
        mask = refineind1d >= 0
        indexmask = mask.nonzero().squeeze() 
        patch_ids_ref = indexmask 
        x_local= rearrange(x_local, 't b c ntxy_ref -> b t c ntxy_ref')
        x_local_padding[patch_ids_ref] = x_local
        x_local_padding = rearrange(x_local_padding, '(b nref_tokens) t c ntxy_ref -> t b c (nref_tokens ntxy_ref)', b=B)
    
        x_padding[:, :, :, :ntoken_coarse] = x_coarse
        x_padding[:, :, :, ntoken_coarse:] = x_local_padding
        mask2d = (refineind[:, :nref_tokens]>=0).repeat_interleave(ntxy_ref, dim=-1)
        mask_padding[:,ntoken_coarse:]=mask2d

        #time, position, and area
        t_pos_area_padding[:, :, :ntoken_coarse, :] = t_pos_area
        t_pos_area_local_padding[patch_ids_ref] = t_pos_area_local
        t_pos_area_local_padding = rearrange(t_pos_area_local_padding, '(b nref_tokens) t ntxy_ref c -> b t (nref_tokens ntxy_ref) c', b=B)
        t_pos_area_padding[:, :, ntoken_coarse:, :] = t_pos_area_local_padding

        return x_padding, patch_ids_ref, mask_padding, t_pos_area_padding

    def get_t_pos_area(self, x, embed_index):
        #assuming pos_x: 0->1 and pos_y: 0->1
        #x: [T, B, C, H, W]
        #return: [B, T, ntokenx*ntokeny, 4]
        T, B, C, H, W = x.shape
        ps0=self.embed_ensemble[embed_index].patch_size[0]
        ps1=self.embed_ensemble[embed_index].patch_size[1]
        ntokenx = H//ps0
        ntokeny = W//ps1
        t_pos_area = torch.zeros(B, T, ntokenx*ntokeny, 4, device=x.device)
        #time position
        t_pos_area[:,:,:,0]=repeat(torch.arange(T), "t -> b t hw", b=B, hw=ntokenx*ntokeny)
        #space position
        dx = 1.0/H*ps0; dy=1.0/W*ps1
        x_seq = repeat(torch.arange(dx*0.5, 1.0, dx), "h -> b t h w", b=B, t=T, w=ntokeny)
        y_seq = repeat(torch.arange(dy*0.5, 1.0, dy), "w -> b t h w", b=B, t=T, h=ntokenx)
        t_pos_area[:,:,:,1]=rearrange(x_seq, "b t h w-> b t (h w)")
        t_pos_area[:,:,:,2]=rearrange(y_seq, "b t h w-> b t (h w)")
        #area
        t_pos_area[:,:,:,3]=dx*dy
        return t_pos_area, dx, dy

    def adjust_t_pos_area(self, t_pos_area_local, t_pos_area, refineind, dx, dy):
        """"
        #local adjustment for t_pos_area_local: 
        #   1) shift pos_x and pos_y to match coarse; 2) area scaling
        # t_pos_area: [B, T, ntoken_coarse, 4] (last dimension: time, x, y, area)
        # refineind: in shape[B, ntoken_coarse]
        # t_pos_area_local: None or [npatches, T, ntxy_ref, 4]
        """
        B, T, ntoken_coarse, C = t_pos_area.shape
        _, _, ntxy_ref, _      = t_pos_area_local.shape

        t_pos_area = rearrange(t_pos_area, 'b t ntoken_coarse c -> (b ntoken_coarse) t c ')
        refineind1d = rearrange(refineind, 'b  ntoken_coarse -> (b ntoken_coarse)')
        mask = refineind1d >= 0
        patch_ids_ref = mask.nonzero().squeeze() 
        t_pos_area = repeat(t_pos_area, "bntoken_coarse t c -> bntoken_coarse t ntxy_ref c", ntxy_ref=ntxy_ref)

        #area rescaling
        t_pos_area_local[:,:,:,-1] = t_pos_area_local[:,:,:,-1] * t_pos_area[patch_ids_ref,:, :, -1]
        #position shift, pos_x, pox_y
        t_pos_area_local[:,:,:,1] = (t_pos_area_local[:,:,:,1] -0.5)*dx + t_pos_area[patch_ids_ref,:, :, 1]
        t_pos_area_local[:,:,:,2] = (t_pos_area_local[:,:,:,2] -0.5)*dy + t_pos_area[patch_ids_ref,:, :, 2]
        return t_pos_area_local

    def get_patchsequence(self, x,  state_labels, refineind=None, leadtime=None):
        ### intput tensors
        #       x: [T, B, C, H, W]
        #       refineind: None or [B, ntoken_x*ntoken_y]
        ### if refineind is None: return 
        #x: [T, B, C_emb, ntoken_x*ntoken_y]
        #t_pos_area:[b, t, slen, 4]
        ### else: 
        #     if self.sts_model,return tensors
        #       x: [T, B, C_emb, ntoken_x*ntoken_y]
        #       patch_ids: [npatches] #selected token ids with sample pos inside batch considere
        #       x_local: [T, npatches, C_emb, ps0/pso_ref*ps1/ps1_ref] 
        #       leadtime_local: [npatches, 1]
        #       t_pos_area:[b, t, slen, 4]
        #       t_pos_area_local:[npatches, t, slen, 4]
        #     else: return tensors
        #       x_padding: [T, B, C_emb, ntoken_len_tot]
        #       patch_ids_ref: [npatches] (ids of effective tokens in x_local)
        #       patch_ids: [npatches] #selected token ids with sample pos inside batch considered
        T, B, C, H, W = x.shape
        x_0 = x.clone()
        ########################################################
        ##############tokenizie at the coarse scale##############
        # x in shape [T, B, C_emb, ntoken_x, ntoken_y]
        x = self.get_structured_sequence(x, state_labels, -1) 
        x = rearrange(x, 't b c h w -> t b c (h w)')
        t_pos_area, dx, dy=self.get_t_pos_area(x_0, -1)
        if refineind is None:
            return x, None, None, None, None, None, t_pos_area, None
        ########################################################
        ##############tokenizie at the fine scale##############
        x_local, patch_ids, leadtime_local = self.get_refined_localpatches(x_0, refineind, leadtime=leadtime)
        x_local = rearrange(x_local, 'nrfb t c h w -> t nrfb c h w')
        t_pos_area_local,_,_=self.get_t_pos_area(x_local, 0)
        t_pos_area_local=self.adjust_t_pos_area(t_pos_area_local, t_pos_area, refineind, dx, dy)
        x_local =self.get_structured_sequence(x_local, state_labels, 0)
        x_local = rearrange(x_local, 't nrfb c h w -> t nrfb c (h w)')
        if self.sts_model:
            return x, patch_ids, None, None, x_local, leadtime_local, t_pos_area, t_pos_area_local
        # xlocal in shape [T, B*nref_tokens, C_embm, ntxy_ref]
        x_padding, patch_ids_ref, mask_padding, t_pos_area_padding = self.get_merge_sequences(x, x_local, refineind, t_pos_area, t_pos_area_local)
        return x_padding, patch_ids, patch_ids_ref, mask_padding, None, None, t_pos_area_padding, None
    
    def get_spatiotemporalfromsequence(self, x_padding, patch_ids, patch_ids_ref, state_labels, H, W):
        #taking token sequences, x_padding, in shape [T, B, C_emb, ntoken_len_tot] as input
        #return [T, B, C, H, W]
        T, B, _, _ = x_padding.shape
        ########################################################################
        ps0=self.embed_ensemble[-1].patch_size[0]
        ps1=self.embed_ensemble[-1].patch_size[1]
        ntokenx = H//ps0
        ntokeny = W//ps1
        ntoken_coarse = ntokenx * ntokeny
        ps0_ref=self.embed_ensemble[0].patch_size[0]
        ps1_ref=self.embed_ensemble[0].patch_size[1]
        ntx_ref = ps0//ps0_ref
        nty_ref = ps1//ps1_ref
        ntxy_ref = ntx_ref * nty_ref
        ########################################################################
        ##############tokenzie at the coarse scale##############
        x_coarsen = x_padding[:, :, :, :ntoken_coarse]
        x_coarsen = rearrange(x_coarsen, 't b c (h w) -> (t b) c h w', h=ntokenx)
        x_coarsen = self.debed_ensemble[-1](x_coarsen, state_labels[0])            
        x_coarsen = rearrange(x_coarsen, '(t b) c h w -> t b c h w', t=T)
        if patch_ids is None:
            return x_coarsen
        ########################################################
        ##############tokenzie at the fine scale#############
        xlocal=x_padding[:,:,:,ntoken_coarse:]
        xlocal = rearrange(xlocal, 't b c (nref_tokens ntxy_ref) -> t (b nref_tokens) c ntxy_ref', ntxy_ref = ntxy_ref)
        xlocal = rearrange(xlocal, 't nrfb c (h w) -> (t nrfb) c h w', h=ntx_ref)
        xlocal = self.debed_ensemble[0](xlocal, state_labels[0])
        xlocal = rearrange(xlocal, '(t nrfb) c h w -> nrfb t c h w', t=T) 
        ########################################################
        #images to small patches
        x_coarsen = rearrange(x_coarsen, 't b c (ntx h) (nty w) -> (b ntx nty) t c h w', ntx=ntokenx, nty=ntokeny)
        x_coarsen[patch_ids] = x_coarsen[patch_ids] + xlocal[patch_ids_ref]
        #small patches to images
        x_coarsen = rearrange(x_coarsen, '(b ntx nty) t c h w -> t b c (ntx h) (nty w)', ntx=ntokenx, nty=ntokeny)
        return x_coarsen 
    
    def add_sts_model(self, xbase, patch_ids, x_local, state_labels, bcs, leadtime=None, t_pos_area=None):
        T, _, _, _, _ = xbase.shape
        ps0=self.embed_ensemble[-1].patch_size[0]    
        ps0_ref=self.embed_ensemble[0].patch_size[0]
        ntx_ref = ps0//ps0_ref

        # Process
        if self.posbias is not None:
            posbias = self.posbias(t_pos_area) # b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> t b c L')
            x_local = x_local + posbias
        #FIXME: assume bcs always 0 for local patches
        #for blk in self.blocks_sts:
        for blk in self.blocks:
            x_local = blk(x_local, bcs*0.0, leadtime=leadtime)#, t_pos_area=t_pos_area)
        self.debug_nan(x_local, message="x_local attention block")
        # Decode -
        x_local = rearrange(x_local, 't nrfb c (h w) -> (t nrfb) c h w', h=ntx_ref)
        x_local = self.debed_ensemble[0](x_local, state_labels[0])
        x_local = rearrange(x_local, '(t nrfb) c h w -> nrfb t c h w', t=T) 

        x = self.add_localpatches(xbase, x_local, patch_ids)
        return x
    
    def forward(self, x, state_labels, bcs, leadtime=None, refineind=None, returnbase4train=False):
        _, _, _, H, W = x.shape
        self.debug_nan(x, message="input")
        x, data_mean, data_std = normalize_spatiotemporal_persample(x)
        ################################################################################
        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP(leadtime)
        else:
            leadtime=None
        ########Encode and get patch sequences [T, B, C_emb, ntoken_len_tot]########
        if  self.sts_model:
            #x_padding: coarse tokens; x_local: refined local tokens
            x_padding, patch_ids, _, _, x_local, leadtime_local, tposarea_padding, tposarea_local = self.get_patchsequence(x, state_labels, refineind=refineind, leadtime=leadtime)
            mask_padding = None
        else:
            x_padding, patch_ids, patch_ids_ref, mask_padding, _, _, tposarea_padding, _ = self.get_patchsequence(x, state_labels, refineind=refineind)
        ################################################################################
        if self.posbias is not None:
            posbias = self.posbias(tposarea_padding, mask_padding=mask_padding) # b t L c->b t L c_emb
            posbias=rearrange(posbias,'b t L c -> t b c L')
            x_padding = x_padding + posbias   
        ######## Process ########
        for blk in self.blocks:
            x_padding = blk(x_padding, bcs, leadtime=leadtime, mask_padding=mask_padding)#, t_pos_area=tposarea_padding) #t_pos_area:[b, t, slen, 4]
        self.debug_nan(x_padding, message="attention block")
        ################################################################################
        ######## Decode ########
        if self.sts_model:
            xbase = self.get_spatiotemporalfromsequence(x_padding, None, None, state_labels, H, W)
            x = self.add_sts_model(xbase, patch_ids, x_local, state_labels, bcs, leadtime=leadtime_local, t_pos_area=tposarea_local)
        else:
            x = self.get_spatiotemporalfromsequence(x_padding, patch_ids, patch_ids_ref, state_labels, H, W)
        ######### Denormalize ########
        x = x * data_std + data_mean # All state labels in the batch should be identical
        ################################################################################
        if returnbase4train:
            xbase = xbase * data_std + data_mean
            return x[-1], xbase[-1]
        return x[-1] 


