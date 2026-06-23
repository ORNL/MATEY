import torch
import torch.nn as nn

from .avit import build_avit
from .svit import build_svit
from .vit import build_vit
from .turbt import build_turbt

def build_diffusion_model(params):
    model = EDMPrecond(params=params)
    return model


class EDMPrecond(nn.Module):
    def __init__(self,
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        params          = None,             # Additional parameters for the underlying model.
    ):
        super().__init__()

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        if params.model_type == 'avit':
            self.model = build_avit(params)
            self.tokenizer_heads_params = self.model.tokenizer_heads_params
        elif params.model_type == 'svit':
            self.model = build_svit(params)
            self.tokenizer_heads_params = self.model.tokenizer_heads_params
        elif params.model_type == 'vit_all2all':
            self.model = build_vit(params)
            self.tokenizer_heads_params = self.model.tokenizer_heads_params
        elif params.model_type == 'turbt':
            self.model = build_turbt(params)
            self.tokenizer_heads_params = self.model.tokenizer_heads_params
        else:
            raise ValueError(f"Unknown diffusion model type: {params.model_type}")


    def forward(self, x, sigma, field_labels, bcs, opts, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(1, -1, 1, 1, 1, 1) # reshape for broadcasting. shape: [1, B, 1, 1, 1, 1]
        # class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        opts.sigma = c_noise.flatten()
        # print(f"opts.imod: {opts.imod}, opts.imod_bottom: {opts.imod_bottom}")
        # print(f"x shape: {x.shape}, c_in shape: {c_in.shape}, c_skip shape: {c_skip.shape}, c_out shape: {c_out.shape}, c_noise shape: {opts.sigma.shape}")
        # print(f"class_labels shape: {opts.diffusion_cond.shape}")
        F_x = self.model((c_in * x).to(dtype), field_labels, bcs, opts)

        F_x = F_x.unsqueeze(0)
        # print(f"x shape: {x.shape}, F_x shape: {F_x.shape}, c_in shape: {c_in.shape}, c_skip shape: {c_skip.shape}, c_out shape: {c_out.shape}, c_noise shape: {opts.sigma.shape}")
        
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # print(f"x shape: {x.shape}, c_skip shape: {c_skip.shape}, c_out shape: {c_out.shape}, F_x shape: {F_x.shape}, D_x shape: {D_x.shape}")

        return D_x

