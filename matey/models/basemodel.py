import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from .spatial_modules import hMLP_stem, hMLP_output, SubsampledLinear
from .time_modules import leadtimeMLP
from .input_modules import input_control_MLP
from .positionbias_modules import positionbias_mod
import sys
from operator import mul
from functools import reduce

class BaseModel(nn.Module):
    """
    Args:
        tokenizer_heads (list): List of dictionaries, each is tokenizer
        tokenizer_heads:
            - head_name: "tk-2D"
            patch_size: [[1, 16, 16]] 
            - head_name: "tk-3D"
            patch_size: [[16, 16, 16]] 
        embed_dim (int): Dimension of the embedding
        n_states (int): Number of input state variables.
    """
    def __init__(self, tokenizer_heads, n_states=6, n_states_cond=None, embed_dim=768, leadtime=False,cond_input=False, n_steps=1, bias_type="none", SR_ratio=[1,1,1], model_SR=False, hierarchical=None, notransposed=False, nlevels=1, smooth=False):
        super().__init__()
        self.space_bag = nn.ModuleList([SubsampledLinear(n_states, embed_dim//4) for _ in range(nlevels)])
        self.conditioning = (n_states_cond is not None and n_states_cond > 0)
        if self.conditioning:
            self.space_bag_cond = nn.ModuleList([SubsampledLinear(n_states_cond, embed_dim//4) for _ in range(nlevels)])
        self.tokenizer_heads_params = {}
        self.tokenizer_outheads_params = {}
        self.tokenizer_heads_gammaref={}
        self.tokenizer_ensemble_heads=nn.ModuleList()
        self.leadtime=leadtime
        self.ltimeMLP=nn.ModuleList()
        self.posbias =nn.ModuleList()
        self.cond_input=cond_input
        self.model_SR = model_SR
        if self.cond_input:
            self.inconMLP=nn.ModuleList()
        for _ in range(nlevels):
            tokenizer_ensemble_heads_level=nn.ModuleDict({})
            for tk in tokenizer_heads:
                head_name = tk["head_name"]
                patch_size = tk["patch_size"]
                if all(isinstance(ps, int) for ps in patch_size) and len(patch_size)==3:
                    patch_size = [patch_size]
                if self.model_SR:
                    # multiply the patch_size by the SR_ratio elementwise
                    output_patch_size = []
                    for ps in patch_size:
                        output_patch_size.append([int(x*y) for x, y in zip(ps, SR_ratio)])
                else:
                    # SR is handled outside the model (input is interpolated to high-res first)
                    output_patch_size = list(patch_size)
                self.tokenizer_heads_params[head_name]=patch_size
                self.tokenizer_outheads_params[head_name]=output_patch_size
                self.tokenizer_heads_gammaref[head_name]=tk.get("gammaref", None)
                tokenizer_ensemble_heads_level[head_name]=nn.ModuleDict({})
                #patches at multiple scales/sizes
                embed_ensemble = nn.ModuleList()
                debed_ensemble = nn.ModuleList()
                embed_ensemble_cond = nn.ModuleList()
                for ps_scale, ps_scale_out in zip(patch_size, output_patch_size):
                    embed_ensemble.append(hMLP_stem(patch_size=ps_scale, in_chans=embed_dim//4, embed_dim=embed_dim))
                    debed_ensemble.append(hMLP_output(patch_size=ps_scale_out, embed_dim=embed_dim, out_chans=n_states, notransposed=notransposed, smooth=smooth))
                    if self.conditioning:
                        embed_ensemble_cond.append(hMLP_stem(patch_size=ps_scale, in_chans=embed_dim//4, embed_dim=embed_dim))
                tokenizer_ensemble_heads_level[head_name]["embed"] = embed_ensemble
                tokenizer_ensemble_heads_level[head_name]["debed"] = debed_ensemble
                tokenizer_ensemble_heads_level[head_name]["embed_cond"] = embed_ensemble_cond
            self.tokenizer_ensemble_heads.append(tokenizer_ensemble_heads_level)
            if self.leadtime:
                self.ltimeMLP.append(leadtimeMLP(hidden_dim=embed_dim))
            if self.cond_input:
                self.inconMLP.append(input_control_MLP(hidden_dim=embed_dim,n_steps=n_steps))
            self.posbias.append(positionbias_mod(bias_type, embed_dim))
        self.embed_dim=embed_dim 
        
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
   
    def expand_projections(self, expansion_amount):
        """ Appends addition embeddings for finetuning on new data """
        if expansion_amount==0:
            return
        with torch.no_grad():
            # Expand input projections
            temp_space_bag = SubsampledLinear(dim_in = self.space_bag.dim_in + expansion_amount, dim_out=self.space_bag.dim_out)
            temp_space_bag.weight[:, :self.space_bag.dim_in] = self.space_bag.weight
            temp_space_bag.bias = self.space_bag.bias
            self.space_bag = temp_space_bag
            # expand output projections
            for ilevel in range(self.token_level):
                out_head = nn.ConvTranspose3d(self.debed_ensemble[ilevel].embed_dim//4, self.debed_ensemble[ilevel].out_chans+expansion_amount,
                                            kernel_size=(self.debed_ensemble[ilevel].kz[0],self.debed_ensemble[ilevel].ks[0],self.debed_ensemble[ilevel].ks[0]),
                                            stride=(self.debed_ensemble[ilevel].kz[0],self.debed_ensemble[ilevel].ks[0],self.debed_ensemble[ilevel].ks[0]))

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

    def get_unified_preembedding(self, x, state_labels, op):
        ## input tensor x: [t, b, c, d, h, w]; state_labels[b, c]
        # state_labels: variable index to consider varying datasets 
        # return [t, b, c_emb//4, d, h, w]
        # Sparse proj
        x = rearrange(x, 't b c d h w -> t b d h w c')
        x = op(x, state_labels)
        x = rearrange(x, 't b d h w c -> t b c d h w')
        #self.debug_nan(x)
        return x

    def get_structured_sequence(self, x, embed_index, tokenizer):
        ## input tensor x: [t, b, c_emb//4, d, h, w]
        # embed_index: tokenization at different resolutions; 
        ## and return patch sequences in shape [t, b, c_emd, ntoken_z, ntoken_x, ntoken_y]
        T = x.shape[0]
        x = rearrange(x, 't b c d h w -> (t b) c d h w')
        x = tokenizer[embed_index](x)
        x = rearrange(x, '(t b) c d h w -> t b c d h w', t=T)
        #self.debug_nan(x, message="embed_ensemble")
        return x

    def get_refined_localpatches(self, x_0, refineind, tkhead_name, leadtime=None):
        ## input tensor x_0 in shape [t,b,c,d, h,w]
        #  refineind: in shape[b, ntokenz*ntokenx*ntokeny]; leadtime: [b, 1]
        #   (containing important token ids to be refined, followed by nonimportant ones as "-1")
        ## return
        #       x_local: [npatches, T, C, ps0, ps1, ps2]
        #       leadtime: [npatches,1]
        #       patch_ids: [npatches] #selected token ids with sample pos inside batch considered
        B= x_0.shape[1]
        embed_ensemble = self.tokenizer_ensemble_heads[tkhead_name]["embed"]
        ##############x refinement###################
        space_dims= x_0.shape[3:]
        ntokendim=[]
        #psz, psx, psy
        ps=embed_ensemble[-1].patch_size
        for idim, dim in enumerate(space_dims):
            ntokendim.append(dim//ps[idim])
        ntokens=reduce(mul, ntokendim)
        ####################################
        refineind1d = refineind.flatten()
        #images to batches of smaller patches
        x_0 = rearrange(x_0, 't b c  (ntz d) (ntx h) (nty w)-> (b ntz ntx nty) t c d h w', ntz=ntokendim[0], ntx=ntokendim[1], nty=ntokendim[2])
        mask = refineind1d >= 0
        batch_offsets = torch.arange(B, device=x_0.device).repeat_interleave(ntokens)* ntokens
        patch_ids = refineind1d[mask] + batch_offsets[mask]
        x_local = x_0[patch_ids].clone() #(nrefines_batch, T, x_0.size()[2], ps0, ps1, ps2)
        if leadtime is not None:
            leadtime = leadtime.repeat_interleave(ntokens, dim=0)[mask]
        #self.debug_nan(x_local, message="xlocal ")
        return x_local, patch_ids, leadtime
    
    def add_localpatches(self, x, x_local, patch_ids, ntokendim):
        ###inputs:
        # x in shape [T,B,C,D,H,W]
        # x_local in shape[nrefines_batch, T, C, ps0, ps1, ps2]
        # patch_ids in shape [nrefines_batch]
        # ntokendim in shape [D//ps0, H//ps1, W//ps2]
        ###return x in the same shape [T,B,C,D, H,W]
        ########################################################################
        #images to small patches
        x = rearrange(x, 't b c (ntz d) (ntx h) (nty w) -> (b ntz ntx nty) t c d h w', ntz=ntokendim[0], ntx=ntokendim[1], nty=ntokendim[2])
        x[patch_ids] = x[patch_ids] + x_local.clone()
        #small patches to images
        x = rearrange(x, '(b ntz ntx nty) t c d h w -> t b c (ntz d) (ntx h) (nty w)', ntz=ntokendim[0], ntx=ntokendim[1], nty=ntokendim[2])
        #self.debug_nan(x, message="add_localpatches")
        return x
    
    def get_t_pos_area(self, x, embed_index, tkhead_name, blockdict=None, ilevel=0):
        #assuming pos_x: 0->1, pos_y: 0->1, and pos_z: 0->1 if blockdict is None
        #x: [T, B, C, D, H, W]
        #return: [B, T, ntokenz, ntokenx, ntokeny, 5]
        T, B = x.shape[:2]
        space_dims= x.shape[3:]
        expand_patterns={0: "d -> b t d h w",
                        1: "h -> b t d h w",
                        2: "w -> b t d h w"
                        }
        embed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed"]
        ntokendim=[]
        delta=[]
        #psz, psx, psy
        ps = embed_ensemble[embed_index].patch_size
        for idim, dim in enumerate(space_dims):
            ntokendim.append(dim//ps[idim])
            delta.append(1.0/dim*ps[idim])

        t_pos_area = torch.zeros(B, T, ntokendim[0], ntokendim[1], ntokendim[2], 2+len(space_dims), device=x.device)
        #time position
        t_pos_area[:,:,:,:,:,0]=repeat(torch.arange(T), "t -> b t d h w", b=B, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
        #space position
        for idim, dim in enumerate(space_dims):
            dim_seq = repeat(torch.arange(delta[idim]*0.5, 1.0, delta[idim]), expand_patterns[idim], b=B, t=T, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
            t_pos_area[:,:,:,:,:,idim+1]=dim_seq
        #area
        t_pos_area[:,:,:,:,:,-1]=reduce(mul, delta)
        if blockdict is not None:
            zxy_start = blockdict["zxy_start"] 
            Lzxy = blockdict["Lzxy"]
            assert len(zxy_start)==3 and len(Lzxy)==3, f"{zxy_start}, {Lzxy}"
            #correct position
            for idim in range(len(space_dims)):
                t_pos_area[...,idim+1]=t_pos_area[...,idim+1]*Lzxy[idim] + zxy_start[idim]
            #correct area
            t_pos_area[...,-1]=t_pos_area[...,-1]*reduce(mul, Lzxy)
        return t_pos_area, delta
    
    def get_merge_sequences(self, x_coarse, x_local, refineind, t_pos_area, t_pos_area_local):
        ###input tensors
        #       x_coarse: [T, B, C_emb, ntoken_coarse]
        #       x_local : [npatches, T, C_emb, ntzxy_sts]
        #       refineind: in shape[B, ntokenz*ntokenx*ntokeny()=ntoken_coarse)]
        #      t_pos_area:  [B, T, ntoken_coarse, 5]
        #      t_pos_area_local:  [npatches, T, ntzxy_sts, 5]
        ###return tnesors
        #       x_padding: [T, B, C_emb, ntoken_len_tot]
        #       patch_ids_ref: [npatches] (ids of effective tokens in x_local)
        #       mask_padding: [B, ntoken_len_tot]
        #       t_pos_area:  [B, T, ntoken_len_tot, 5]
        #######################################################
        T, B, C_emb, ntoken_coarse = x_coarse.shape
        ntzxy_sts = x_local.shape[-1]
        nref_tokens = torch.max(torch.sum(refineind>=0, dim=1)).item()
        ntoken_len_tot = ntoken_coarse + nref_tokens*ntzxy_sts

        x_local_padding = torch.full((B*nref_tokens, T, C_emb, ntzxy_sts), 1e12, device=x_local.device)
        x_padding = torch.full((T, B, C_emb, ntoken_len_tot), 1e12, device=x_coarse.device)
        t_pos_area_local_padding = torch.full((B*nref_tokens, T, ntzxy_sts, 5), 1e12, device=x_local.device)
        t_pos_area_padding = torch.full((B, T, ntoken_len_tot, 5), 1e12, device=x_coarse.device)
        mask_padding = torch.full((B, ntoken_len_tot), True, device=x_coarse.device) #True: meaningful patches; False: padding patches

        refineind1d = refineind[:, :nref_tokens].flatten()
        mask = refineind1d >= 0
        indexmask = mask.nonzero().squeeze()
        patch_ids_ref = indexmask

        ###x padding####
        x_local_padding[patch_ids_ref] = x_local
        x_local_padding = rearrange(x_local_padding, '(b nref_tokens) t c ntzxy_sts-> t b c (nref_tokens ntzxy_sts)', b=B)
        x_padding[:, :, :, :ntoken_coarse] = x_coarse
        x_padding[:, :, :, ntoken_coarse:] = x_local_padding

        ###time, position, and area###
        t_pos_area_padding[:, :, :ntoken_coarse, :] = t_pos_area
        t_pos_area_local_padding[patch_ids_ref] = t_pos_area_local
        t_pos_area_local_padding = rearrange(t_pos_area_local_padding, '(b nref_tokens) t ntzxy_sts c -> b t (nref_tokens ntzxy_sts) c', b=B)
        t_pos_area_padding[:, :, ntoken_coarse:, :] = t_pos_area_local_padding

        mask2d = (refineind[:, :nref_tokens]>=0).repeat_interleave(ntzxy_sts, dim=-1)
        mask_padding[:,ntoken_coarse:]=mask2d
        #######################################################
        if self.replace_patch:
            ##mask the picked tokens
            mask = refineind >=0
            batch_ind, _ = torch.nonzero(mask, as_tuple=True) #1d tensor of the batch indices of selected patches
            refined_patches = refineind[mask] #1d tensor of patch id of the selected patches 
            mask_padding[batch_ind, refined_patches]=False
            x_padding[:, batch_ind, :, refined_patches] = 1e12 
            t_pos_area_padding[batch_ind, :, refined_patches,:] = 1e12 

        return x_padding, patch_ids_ref, mask_padding, t_pos_area_padding

    def get_chosenrefinedpatches(self, x_refine, refineind, t_pos_area_refine, embed_ensemble, leadtime=None):
        """
        ###input tensors 
        #      x_refine :[T, B, C_emb, nt_z_ref, nt_x_ref, nt_y_ref] #ntzxy_ref= nt_z_ref*nt_x_ref*nt_y_ref
        #      refineind: in shape[B, ntokenz*ntokenx*ntokeny] 
        #      t_pos_area_refine:  [B, T, ntzxy_ref,  5]
        ###return tnesors
        #       x_local:    [npatches, T, C, ntzxy_sts]
        #       t_pos_area_local: [npatches, T, ntzxy_sts  5]
        #       leadtime: [npatches, 1]
        #       patch_ids:     [npatches] (ids of coarsen tokens chosen to refine)
        """
        #######################################################
        #embed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed"]
        #######################################################
        B, ncoarse = refineind.shape
        nt_z_ref, nt_x_ref, nt_y_ref = x_refine.shape[3:]
        #psz, psx, psy
        ps_c  = embed_ensemble[-1].patch_size
        ps_ref= embed_ensemble[0].patch_size
        psr0, psr1, psr2 = [ps_ci//ps_refi for ps_ci, ps_refi in zip(ps_c, ps_ref)] #sts
        #######################################################
        #[T, B, C_emb, nt_z_ref, nt_x_ref, nt_y_ref] --> T, B, C_emb, nt_z, nt_x, nt_y, psr0, psr1, psr2
        x_refine=x_refine.unfold(3, psr0, psr0).unfold(4, psr1, psr1).unfold(5, psr2, psr2)
        x_refine= rearrange(x_refine, 't b c nt_z nt_x nt_y psr0 psr1 psr2 -> b (nt_z nt_x nt_y) t c (psr0 psr1 psr2)')

        t_pos_area_refine = rearrange(t_pos_area_refine, 'b t (nt_z_ref nt_x_ref nt_y_ref) c -> b t c nt_z_ref nt_x_ref nt_y_ref', nt_z_ref=nt_z_ref, nt_x_ref=nt_x_ref, nt_y_ref=nt_y_ref)
        t_pos_area_refine = t_pos_area_refine.unfold(3, psr0, psr0).unfold(4, psr1, psr1).unfold(5, psr2, psr2)
        t_pos_area_refine = rearrange(t_pos_area_refine, 'b t c nt_z nt_x nt_y psr0 psr1 psr2 -> b (nt_z nt_x nt_y) t (psr0 psr1 psr2) c')
        #######################################################
        refineind1d = refineind.flatten() #equvivalent to rearrange(refineinx, 'b ntzxy -> (b ntzxy)')
        mask = refineind1d >= 0
        batch_offsets = torch.arange(B, device=x_refine.device).repeat_interleave(ncoarse)* ncoarse
        patch_ids = refineind1d[mask] + batch_offsets[mask]

        x_refine = rearrange(x_refine, 'b nt_zxy t c ntzxy_sts -> (b nt_zxy) t c ntzxy_sts')
        t_pos_area_refine= rearrange(t_pos_area_refine,'b nt_zxy t ntzxy_sts c -> (b nt_zxy) t ntzxy_sts c')
        x_local = x_refine[patch_ids]
        t_pos_area_local=t_pos_area_refine[patch_ids]

        if leadtime is not None:
            leadtime = leadtime.repeat_interleave(ncoarse, dim=0)[mask]
        return x_local, t_pos_area_local, patch_ids, leadtime

    def get_patchsequence(self, x,  state_labels, tkhead_name, refineind=None, leadtime=None, blockdict=None, ilevel=0, conditioning: bool = False):
        """
        ### intput tensors
        #       x: [T, B, C, D, H, W]
        #       refineind: None or [B, ntoken_z*ntoken_x*ntoken_y]
        ### if refineind is None: return
        #    x: [T, B, C_emb, ntoken_z*ntoken_x*ntoken_y] 
        #    t_pos_area: [B, T, ntoken_z*ntoken_x*ntoken_y, 5] (last dimension: time, x, y, z, volume)
        ### else:
        #     if self.sts_model,return tensors
        #       x: [T, B, C_emb, ntoken_z*ntoken_x*ntoken_y] 
        #       patch_ids: [npatches] #selected token ids with sample pos inside batch considere
        #       x_local: [npatches, T, C_emb, ps2/ps2_ref*ps0/pso_ref*ps1/ps1_ref] 
        #       leadtime_local: [npatches, 1]
        #       t_pos_area: [B,T, ntoken_z*ntoken_x*ntoken_y, 5]
        #       t_pos_area_local: [npatches, T, n_ref_z*n_ref_x*n_ref_y, 5]
        #     else: return tensors
        #       x_padding: [T, B, C_emb, ntoken_len_tot] 
        #       patch_ids_ref: [npatches] (ids of effective tokens in x_local)
        #       patch_ids: [npatches] #selected token ids with sample pos inside batch considered
        #       t_pos_area: [B, T, ntoken_len_tot, 5]
        """
        ########################################################
        #[T, B, C_emb//4, D, H, W]
        op = self.space_bag[ilevel] if not conditioning else self.space_bag_cond[ilevel]
        x_pre = self.get_unified_preembedding(x, state_labels, op)
        ##############tokenizie at the coarse scale##############
        # x in shape [T, B, C_emb, ntoken_z, ntoken_x, ntoken_y]
        tokenizer = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed" if not conditioning else "embed_cond"]
        x = self.get_structured_sequence(x_pre, -1, tokenizer)
        x = rearrange(x, 't b c d h w -> t b c (d h w)')
        t_pos_area, _ = self.get_t_pos_area(x_pre, -1, tkhead_name, blockdict=blockdict, ilevel=ilevel)
        t_pos_area = rearrange(t_pos_area, 'b t d h w c-> b t (d h w) c')
        if refineind is None:
            return x, None, None, None, None, None, t_pos_area, None
        ########################################################
        #FIXME: ("the following code breaks in MG test")
        ##############tokenizie at the fine scale##############
        #x in shape [T, B, C_emb, nt_z_ref, nt_x_ref, nt_y_ref]
        tokenizer = self.tokenizer_ensemble_heads[0][tkhead_name]["embed" if not conditioning else "embed_cond"]
        x_ref = self.get_structured_sequence(x_pre, 0, tokenizer)
        t_pos_area_ref, _ =self.get_t_pos_area(x_pre, 0, tkhead_name, blockdict=blockdict)
        t_pos_area_ref = rearrange(t_pos_area_ref, 'b t d h w c-> b t (d h w) c')
        x_local, t_pos_area_local, patch_ids, leadtime_local = self.get_chosenrefinedpatches(x_ref, refineind, t_pos_area_ref, tokenizer, leadtime=leadtime)
        #[npatches, T, C, ntzxy_sts] and [npatches, T, ntzxy_sts, 5]
        if self.sts_model:
            return x, patch_ids, None, None, x_local, leadtime_local, t_pos_area, t_pos_area_local
        ########################################################
        x_padding, patch_ids_ref, mask_padding, t_pos_area_padding = self.get_merge_sequences(x, x_local, refineind, t_pos_area, t_pos_area_local)
        #x_padding: [T, B, C_emb, ntoken_tot]
        #mask_padding: [B, ntoken_tot]
        return x_padding, patch_ids, patch_ids_ref, mask_padding, None, None, t_pos_area_padding, None

    def get_spatiotemporalfromsequence(self, x_padding, patch_ids, patch_ids_ref, space_dims, tkhead_name, ilevel=0):
        #taking token sequences, x_padding, in shape [T, B, C_emb, ntoken_tot] as input
        #patch_ids_ref: [npatches] (ids of effective tokens in x_local)
        #patch_ids: [npatches] #selected token ids with sample pos inside batch considered
        # Tspace_dims=[T, D, H, W]
        #return [T, B, C, D, H, W]
        T, B = x_padding.shape[:2]
        ########################################################################
        embed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["embed"]
        debed_ensemble = self.tokenizer_ensemble_heads[ilevel][tkhead_name]["debed"]
        ########################################################################
        ntokendim   =[]
        ntokenstsdim=[]
        ps_c  = embed_ensemble[-1].patch_size
        ps_ref= embed_ensemble[0].patch_size
        for idim, dim in enumerate(space_dims):
            ntokendim.append(dim//ps_c[idim])
            ntokenstsdim.append(ps_c[idim]//ps_ref[idim])
        ntoken_coarse=reduce(mul, ntokendim)
        ntzxy_sts=reduce(mul, ntokenstsdim)
        ########################################################################
        x_coarsen = x_padding[:, :, :, :ntoken_coarse]
        xlocal    = x_padding[:, :, :, ntoken_coarse:]
        ########################################################################
        x_coarsen = rearrange(x_coarsen, 't b c (d h w) -> t b c d h w', d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
        if patch_ids is None or len(patch_ids)==0: # no refinement; reconstruct from coarsen patches
            x_coarsen = rearrange(x_coarsen, 't b c d h w -> (t b) c d h w')
            x_coarsen = debed_ensemble[-1](x_coarsen)
            x_coarsen = rearrange(x_coarsen, '(t b) c d h w -> t b c d h w', t=T)
            return x_coarsen
        ########################################################
        #FIXME: "the following code breaks in MG test"
        if self.replace_patch:
            ######**********refined reconstruction**********######
            #T, B, C, ntz_ref, ntx_ref, nty_ref
            x_refined = repeat(x_coarsen, 't b c d h w -> t b c (d ntz_sts) (h ntx_sts) (w nty_sts)', ntz_sts=ntokenstsdim[0], ntx_sts=ntokenstsdim[1], nty_sts=ntokenstsdim[2]) 
            x_refined = rearrange(x_refined, 't b c (d ntz_sts) (h ntx_sts) (w nty_sts) -> (b d h w) t c (ntz_sts ntx_sts nty_sts)', ntz_sts=ntokenstsdim[0], ntx_sts=ntokenstsdim[1], nty_sts=ntokenstsdim[2]) 
            xlocal = rearrange(xlocal, 't b c (nref_tokens ntzxy_sts) -> (b nref_tokens) t c ntzxy_sts', ntzxy_sts = ntzxy_sts)
            x_refined[patch_ids] = xlocal[patch_ids_ref].clone()
            #refined patches with coarsen patches filled
            x_refined = rearrange(x_refined, '(b d h w) t c (ntz_sts ntx_sts nty_sts) -> (t b) c (d ntz_sts) (h ntx_sts) (w nty_sts)', b=B, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2], ntz_sts=ntokenstsdim[0], ntx_sts=ntokenstsdim[1], nty_sts=ntokenstsdim[2])
            #reconstruct from refined patches to solution fields
            x_reconst_ref = debed_ensemble[0](x_refined) #, state_labels[0])
            x_reconst_ref = rearrange(x_reconst_ref, '(t b) c d h w -> t b c d h w', t=T)
            ######**********add the coarsen reconstruction**********######
            #coarsen patches with important patches from refined
            x_coarsen=rearrange(x_coarsen, 't b c d h w -> (b d h w) t c')
            x_coarsen[patch_ids] = xlocal[patch_ids_ref].mean(dim=-1).clone()
            x_coarsen = rearrange(x_coarsen, '(b d h w) t c ->  (t b) c d h w', b=B, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
            x_reconst = debed_ensemble[-1](x_coarsen) #, state_labels[0])
            x_reconst = rearrange(x_reconst, '(t b) c d h w -> t b c d h w', t=T)
            ########################################################
            #sum two reconstructions
            #x_reconst = x_reconst + x_coarsen_ref
            ########################################################
            #print("Pei debugging0:", x_reconst, x_reconst_ref, flush=True)
            #print("Pei debugging1:", patch_ids, patch_ids_ref, flush=True)
            #merge two reconstructions
            #images to small patches
            x_reconst     = rearrange(x_reconst,     't b c (ntz d) (ntx h) (nty w) -> (b ntz ntx nty) t c d h w', ntz=ntokendim[0], ntx=ntokendim[1], nty=ntokendim[2])
            x_reconst_ref = rearrange(x_reconst_ref, 't b c (ntz d) (ntx h) (nty w) -> (b ntz ntx nty) t c d h w', ntz=ntokendim[0], ntx=ntokendim[1], nty=ntokendim[2])
            #print("Pei debugging0:", x_reconst.shape, x_reconst_ref.shape, patch_ids.shape, patch_ids_ref.shape, flush=True)
            x_reconst[patch_ids] = x_reconst_ref[patch_ids].clone()
            #small patches to images
            x_reconst     = rearrange(x_reconst,     '(b ntz ntx nty) t c d h w -> t b c (ntz d) (ntx h) (nty w)', ntz=ntokendim[0], ntx=ntokendim[1], nty=ntokendim[2])
        else:
            ########################################################
            ##############tokenzie at the fine scale#############
            xlocal = rearrange(xlocal, 't b c (nref_tokens ntzxy_sts) -> (b nref_tokens) t c ntzxy_sts', ntzxy_sts = ntzxy_sts)
            xlocal = xlocal[patch_ids_ref]
            xlocal = rearrange(xlocal, 'nrfb t c (d h w) -> (t nrfb) c d h w', d=ntokenstsdim[0], h=ntokenstsdim[1], w=ntokenstsdim[2])
            xlocal = debed_ensemble[0](xlocal) #, state_labels[0])
            xlocal = rearrange(xlocal, '(t nrfb) c d h w -> nrfb t c d h w', t=T) 
            ########################################################
            #images to small patches
            x_coarsen = rearrange(x_coarsen, 't b c (ntz d) (ntx h) (nty w) -> (b ntz ntx nty) t c d h w', ntz=ntokendim[0], ntx=ntokendim[1], nty=ntokendim[2])
            x_coarsen[patch_ids] = x_coarsen[patch_ids] + xlocal
            #small patches to images
            x_reconst = rearrange(x_coarsen, '(b ntz ntx nty) t c d h w -> t b c (ntz d) (ntx h) (nty w)', ntz=ntokendim[0], ntx=ntokendim[1], nty=ntokendim[2])
        return x_reconst

    def add_sts_model(self):
       pass
    
    def forward(self):
        raise NotImplementedError
       
