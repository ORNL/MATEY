import torch
import torch.distributed as dist
import torch.nn.functional as F
from .forward_options import ForwardOptionsBase, TrainOptionsBase
from contextlib import nullcontext

def preprocess_target(leadtime, ramping_warmup = False):
    """
    #Inputs:
    #  leadtime: (B, 1) with integer lead times (might be different across samples/ranks)
    #  ramping_warmup: If True, use a shorter rollout length during warmup.
    #Returns: 
    # rollout_steps: int, actual leadtime (rollout length) used in training/inference after synchronziation across ranks
    """
    min_lead = int(leadtime.min().item())
    #Global minimum leadtime based on end of data (across all ranks)
    if dist.is_initialized():
        min_lead_tensor =  leadtime.min()
        dist.all_reduce(min_lead_tensor, op=dist.ReduceOp.MIN)
        min_lead = int(min_lead_tensor.item())
    #max rollout length allowed, based on min leadtime and warmup
    if ramping_warmup:
        #Training:
        #FIXME: implement some warmup logic for ramping up rollout length
        #if self.params.auto_warmup and self.n_calls < 1000 and not self.params.resuming:
        max_rollout = max(1, int(min_lead * 0.5))
    else:
        max_rollout = max(1, min_lead)
    #set rollout_steps
    if dist.is_initialized():
        if dist.get_rank() == 0:
            rollout_steps = torch.randint(1, int(max_rollout+1), (1,)).to(leadtime.device)
        else:
            rollout_steps = torch.zeros(1, device=leadtime.device, dtype=torch.int64)

        dist.broadcast(rollout_steps, src=0)
        rollout_steps = rollout_steps.item()
    else:
        rollout_steps = torch.randint(1, int(max_rollout+1), (1,)).item()

    return rollout_steps

def autoregressive_rollout(model, inp, field_labels, bcs, opts: ForwardOptionsBase,  pushforward=True):  
    """
    #Performs an autoregressive rollout with randomly sampled rollout length.
    #Inputs:
    # inp: T,B,C,D,H,W.
    # field_labels: labels for input
    # opts: Forward options object (must contain .leadtime and .cond_input).
    # pushforward: If True, disables gradient computation, except for the last step.
    #Returns:
    # output: Model output after the final autoregressive step ([B, C, D, H, W])
    #  rollout_steps: Number of autoregressive steps performed.
    """
    rollout_steps = preprocess_target(opts.leadtime) 
    x_t = inp
    n_steps = inp.shape[0]
    ctx = torch.no_grad() if pushforward else nullcontext()
    cond_input = opts.cond_input.clone() if opts.cond_input is not None else None
    with ctx:
        for t in range(rollout_steps - 1):
            cond_input_t = cond_input[:, t:n_steps + t + 1] if cond_input is not None else None
            opts.cond_input = cond_input_t
            opts.leadtime = opts.leadtime * 0 + 1 #set leadtime to 1 for autoregressive training
            output_t = model(x_t, field_labels, bcs, opts)
            x_t = torch.cat([x_t[1:], output_t.unsqueeze(0)], dim=0)

    cond_input_t = cond_input[:, rollout_steps-1:n_steps+rollout_steps] if cond_input is not None else None
    opts.cond_input = cond_input_t
    output = model(x_t, field_labels, bcs, opts)# B,C,D,H,W

    return output, rollout_steps

def torch_diff(phi, dx=1.0, dy=1.0, dz=1.0):
    """
    Compute spatial gradients of a 5D tensor phi with shape (B, C, D, H, W).
    """
    # Compute gradients in all three directions at once
    grad_z, grad_x, grad_y = torch.gradient(phi, spacing=(dz, dx, dy), dim=(2, 3, 4), edge_order=1)
    return grad_x, grad_y, grad_z


def GradLoss(input, target):
    # Both input and target have shape (B, C, D, H, W)
    dx = dy = dz = 1.0
    channel_dim = input.shape[1]
    # Compute gradients for all channels at once
    dx_inp, dy_inp, dz_inp = torch_diff(input, dx, dy, dz)
    dx_tgt, dy_tgt, dz_tgt = torch_diff(target, dx, dy, dz)

    # Compute mean squared errors for all gradients
    loss = (
        F.mse_loss(dx_inp, dx_tgt) +
        F.mse_loss(dy_inp, dy_tgt) +
        F.mse_loss(dz_inp, dz_tgt)
    )*channel_dim

    return loss