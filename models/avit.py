import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
try:
    from spatial_modules import hMLP_stem, hMLP_output, SubsampledLinear
    from spacetime_modules import SpaceTimeBlock
    from time_modules import leadtimeMLP
    from positionbias_modules import positionbias_mod
    from ..data_utils.shared_utils import normalize_spatiotemporal_persample, figure_checking
except:
    from .spatial_modules import hMLP_stem, hMLP_output, SubsampledLinear
    from .spacetime_modules import SpaceTimeBlock
    from .time_modules import leadtimeMLP
    from .positionbias_modules import positionbias_mod
    from data_utils.shared_utils import normalize_spatiotemporal_persample, figure_checking
import sys, copy

def build_avit(params):
    """avit model
    sts_model: when True, we use two separte avit modules for coarse and refined tokens, respectively
    sts_train: 
             when True, we use loss function with two parts: l_coarse/base + l_total, so that the coarse ViT approximates true solutions directly as well
    leadtime_max: when larger than 1, we use a `ltimeMLP` NN module to incoporate the impact of leadtime
    """
    model = AViT(patch_size=params.patch_size,
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

class AViT(nn.Module):
    """
    Naive model that interweaves spatial and temporal attention blocks. Temporal attention 
    acts only on the time dimension. 

    Args:
        patch_size (tuple): Size of the input patch
        embed_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.  
    """
    def __init__(self, patch_size=(16, 16), embed_dim=768,  space_type="axial_attention", time_type="attention", num_heads=12, processor_blocks=8, n_states=6,
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

        #FIXME: need to test for other attention types under avit later
        assert space_type=="axial_attention" and time_type=="attention"

        self.blocks = nn.ModuleList([SpaceTimeBlock(space_type, time_type, embed_dim, num_heads, drop_path=self.dp[i])
                                     for i in range(processor_blocks)])
        self.sts_model=sts_model
        if self.sts_model:
            self.blocks_sts = nn.ModuleList([SpaceTimeBlock(space_type, time_type, embed_dim, num_heads, drop_path=self.dp[i])
                            for i in range(processor_blocks)])    
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
            self.blocks_sts = nn.ModuleList([SpaceTimeBlock(self.space_type, self.time_type, self.embed_dim, self.num_heads, drop_path=self.dp[i])
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
        ## input tensor x: [t, b, c, h, w]; 
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
        #  refineind: in shape[b, ntokenx*ntokeny],  
        ## return 
        #       x_local: [npatches, T, C, ps0, ps1]
        #       leadtime: [npatches,1]
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

    def get_t_pos_area(self, x, embed_index):
        #assuming pos_x: 0->1 and pos_y: 0->1
        #x: [T, B, C, H, W]
        #return: [B, T, ntokenx, ntokeny, 4]
        T, B, C, H, W = x.shape
        ps0=self.embed_ensemble[embed_index].patch_size[0]
        ps1=self.embed_ensemble[embed_index].patch_size[1]
        ntokenx = H//ps0
        ntokeny = W//ps1
        t_pos_area = torch.zeros(B, T, ntokenx, ntokeny, 4, device=x.device)
        #time position
        t_pos_area[:,:,:,:,0]=repeat(torch.arange(T), "t -> b t h w", b=B, h=ntokenx, w=ntokeny)
        #space position
        dx = 1.0/H*ps0; dy=1.0/W*ps1
        x_seq = repeat(torch.arange(dx*0.5, 1.0, dx), "h -> b t h w", b=B, t=T, w=ntokeny)
        y_seq = repeat(torch.arange(dy*0.5, 1.0, dy), "w -> b t h w", b=B, t=T, h=ntokenx)
        t_pos_area[:,:,:,:,1]=x_seq
        t_pos_area[:,:,:,:,2]=y_seq
        #area
        t_pos_area[:,:,:,:,3]=dx*dy
        return t_pos_area, dx, dy

    def adjust_t_pos_area(self, t_pos_area_local, t_pos_area, refineind):
        """"
        #local adjustment for t_pos_area_local: 
        #   1) shift pos_x and pos_y to match coarse; 2) area scaling
        # t_pos_area: [B, T, ntoken_x, ntoken_y, 4] (last dimension: time, x, y, area)
        # refineind: in shape[B, ntoken_coarse]
        # t_pos_area_local: [npatches, T, ntx_ref, nty_ref, 4]
        """
        B, T, ntoken_x, ntoken_y, C = t_pos_area.shape
        _, _, ntx_ref, nty_ref, _      = t_pos_area_local.shape
        dx=1.0/ntoken_x; dy=1.0/ntoken_y

        t_pos_area = rearrange(t_pos_area, 'b t ntoken_x ntoken_y c -> (b ntoken_x ntoken_y) t c')
        refineind1d = rearrange(refineind, 'b  ntoken_coarse -> (b ntoken_coarse)')
        mask = refineind1d >= 0
        patch_ids_ref = mask.nonzero().squeeze() 
        t_pos_area = repeat(t_pos_area, "bntoken_coarse t c -> bntoken_coarse t ntx_ref nty_ref c", ntx_ref=ntx_ref, nty_ref=nty_ref)
    
        #area rescaling
        t_pos_area_local[:,:,:,:,-1] = t_pos_area_local[:,:,:,:,-1] * t_pos_area[patch_ids_ref,:,:,:,-1]
        #position shift, pos_x, pox_y
        t_pos_area_local[:,:,:,:,1] = (t_pos_area_local[:,:,:,:,1] -0.5)*dx + t_pos_area[patch_ids_ref,:,:,:,1]
        t_pos_area_local[:,:,:,:,2] = (t_pos_area_local[:,:,:,:,2] -0.5)*dy + t_pos_area[patch_ids_ref,:,:,:,2]
        return t_pos_area_local

    def add_sts_model(self, x, x_0, state_labels, bcs, t_pos_area, leadtime=None, refineind=None):
        T, B, _, H, W = x_0.shape
        if self.token_level>2 or refineind is None:
            #FIXME: no adaptive yet for more than 2 levels
            xlist = []
            for ilevel in range(self.token_level-1):
                xin = x_0.clone()
                xin = self.get_structured_sequence(xin, state_labels, ilevel) # xin in shape [T, B, C_emb, ntoken_x, ntoken_y]
                self.debug_nan(xin, message="after embed")
                # [B, T, ntoken_x, ntoken_y, 4]          
                t_pos_area, dx, dy=self.get_t_pos_area(x_0, ilevel)
                if self.posbias is not None:
                    posbias = self.posbias(t_pos_area) # b t h w c->b t h w c_emb
                    posbias=rearrange(posbias,'b t h w c -> t b c h w')
                    x = x + posbias

                # Process
                for blk in self.blocks:
                    xin = blk(xin, bcs, leadtime=leadtime) #, t_pos_area=t_pos_area)
                self.debug_nan(xin, message="after blk")

                # Decode - It would probably be better to grab the last time here since we're only
                # predicting the last step, but leaving it like this for compatibility to causal masking
                xin = rearrange(xin, 't b c h w -> (t b) c h w')
                xin = self.debed_ensemble[ilevel](xin, state_labels[0])
                xin = rearrange(xin, '(t b) c h w -> t b c h w', t=T)
                xlist.append(xin)
            x = x + torch.stack(xlist, dim=0).sum(dim=0)
        else:
            xlocal, patch_ids, leadtime = self.get_refined_localpatches(x_0, refineind, leadtime=leadtime)
            #checking if refinement mapping works as expected
            #self.figure_checking(x_0, xlocal, B, ntokenx, ntokeny, ps0, ps1, refineind)
            xlocal = rearrange(xlocal, 'nrfb t c h w -> t nrfb c h w')
            t_pos_area_local,_,_=self.get_t_pos_area(xlocal, 0)
            t_pos_area_local=self.adjust_t_pos_area(t_pos_area_local, t_pos_area, refineind)
            xlocal, data_mean_loc, data_std_loc = normalize_spatiotemporal_persample(xlocal)
            xlocal = self.get_structured_sequence(xlocal, state_labels, 0) 

            if self.posbias is not None:               
                posbias = self.posbias(t_pos_area_local) # b t h w c->b t h w c_emb
                posbias=rearrange(posbias,'b t h w c -> t b c h w')
                xlocal = xlocal + posbias
            # Process
            #FIXME: assume bcs always 0 for local patches
            if self.sts_model:
                for blk in self.blocks_sts:
                    xlocal = blk(xlocal, bcs*0.0, leadtime=leadtime)#, t_pos_area=t_pos_area_local)
            else:
                for blk in self.blocks:
                    xlocal = blk(xlocal, bcs*0.0, leadtime=leadtime)#, t_pos_area=t_pos_area_local)
            self.debug_nan(xlocal, message="xlocal attention block")

            # Decode - It would probably be better to grab the last time here since we're only
            # predicting the last step, but leaving it like this for compatibility to causal masking
            xlocal = rearrange(xlocal, 't nrfb c h w -> (t nrfb) c h w')
            xlocal = self.debed_ensemble[0](xlocal, state_labels[0])
            xlocal = rearrange(xlocal, '(t nrfb) c h w -> t nrfb c h w', t=T) 
            xlocal = xlocal * data_std_loc + data_mean_loc # All state labels in the batch should be identical
            self.debug_nan(xlocal, message="xlocal debed_ensemble")
            xlocal = rearrange(xlocal, 't nrfb c h w -> nrfb t c h w')

            x = self.add_localpatches(x, xlocal, patch_ids)
        return x

    def forward(self, x, state_labels, bcs, leadtime=None, refineind=None, returnbase4train=False):    
        T, B, C, H, W = x.shape
        x, data_mean, data_std = normalize_spatiotemporal_persample(x)
        x_0 = x.clone()
        self.debug_nan(x, message="input")

        # x in shape [T, B, C_emb, ntoken_x, ntoken_y]
        x = self.get_structured_sequence(x, state_labels, -1)  
        # [B, T, ntoken_x, ntoken_y, 4]          
        t_pos_area, dx, dy=self.get_t_pos_area(x_0, -1)

        if self.leadtime and leadtime is not None:
            leadtime = self.ltimeMLP(leadtime)
        else:
            leadtime=None

        if self.posbias is not None:
            posbias = self.posbias(t_pos_area) # b t h w c->b t h w c_emb
            posbias=rearrange(posbias,'b t h w c -> t b c h w')
            x = x + posbias
        # Process
        for blk in self.blocks:
            x = blk(x, bcs, leadtime=leadtime)#, t_pos_area=t_pos_area)

        self.debug_nan(x, message="attention block")

        # Decode - It would probably be better to grab the last time here since we're only
        # predicting the last step, but leaving it like this for compatibility to causal masking
        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.debed_ensemble[-1](x, state_labels[0])
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)

        self.debug_nan(x, message="debed_ensemble")

        xbase = x.clone()
        if self.token_level>1:
            x = self.add_sts_model(x, x_0, state_labels, bcs, t_pos_area, leadtime=leadtime, refineind=refineind)
           
        # Denormalize 
        x = x * data_std + data_mean # All state labels in the batch should be identical
        if returnbase4train:
            xbase = xbase * data_std + data_mean
            return x[-1], xbase[-1]
        return x[-1] # Just return last step - now just predict delta.

