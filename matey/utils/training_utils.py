import torch
import torch.distributed as dist
import torch.nn.functional as F
from .forward_options import ForwardOptionsBase, TrainOptionsBase
from contextlib import nullcontext
from torch_geometric.nn import global_mean_pool
from .visualization_utils import checking_data_pred_tar

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
    # inp: T,B,C,D,H,W. or Graph
    # field_labels: labels for input
    # opts: Forward options object (must contain .leadtime and .cond_input).
    # pushforward: If True, disables gradient computation, except for the last step.
    #Returns:
    # output: Model output after the final autoregressive step ([B, C, D, H, W])
    #  rollout_steps: Number of autoregressive steps performed.
    """
    rollout_steps = preprocess_target(opts.leadtime) 
    x_t = inp
    if opts.isgraph:
        n_steps = x_t.x.shape[1] #[nnodes, T, C]
        #FIXME: I realize it takes more to make this function work for graphs and will open a seperate PR on this
        raise ValueError("Autoregressive rollout is not supported yet for graphs")
    else:
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

def compute_loss_and_logs(output, tar, graphdata, logs, loss_logs, dset_type, params):
    """
    compute loss and update logging dicts.
    output: Model prediction [B,C,D,H,W] for tensor inputs or [nnodes, C_tar] for graph
    tar: target same shape as output
    logs: dict; Running log dictionary (updated in-place).
    loss_logs :dict; Dataset-type keyed loss log dict (updated in-place).
    loss :  loss tensor (already scaled by accum_grad and including grad_loss if used).
    """
    residuals = output - tar
    if output.ndim == 2:
        ###full resolution###
         #[nnodes, C_tar] 
        # Differentiate between log and accumulation losses
        raw_loss = global_mean_pool(residuals.pow(2), graphdata.batch)/global_mean_pool(1e-7 + tar.pow(2), graphdata.batch) #B,C
        # Scale loss for accum
        loss = raw_loss.mean() /params.accum_grad
        spatial_dims = None
    else:
        ###full resolution###
        spatial_dims = tuple(range(output.ndim))[2:] # B,C,D,H,W
        #Differentiate between log and accumulation losses
        #B,C,D,H,W->B,C
        raw_loss = residuals.pow(2).mean(spatial_dims)/ (1e-7 + tar.pow(2).mean(spatial_dims))
        # Scale loss for accum
        loss = raw_loss.mean()/params.accum_grad
        #Optional spatial gradient loss
        alpha = getattr(params, "grad_loss_alpha", None)
        if alpha is not None and alpha > 0.0:
            #expects B,C,D,H,W
            grad_loss = GradLoss(output, tar)/params.accum_grad 
            loss += params.grad_loss_alpha * grad_loss
    # Logging
    with torch.no_grad():
        logs['train_l1'] += F.l1_loss(output, tar)
        log_nrmse = raw_loss.sqrt().mean()
        logs['train_nrmse'] += log_nrmse 
        loss_logs[dset_type] += loss.item()
        logs['train_rmse'] += residuals.pow(2).mean(spatial_dims).sqrt().mean()
            
    return loss, log_nrmse

def update_loss_logs_inplace_eval(output, tar, graphdata, logs, loss_dset_logs, loss_l1_dset_logs, loss_rmse_dset_logs, dset_type):
    """
    compute loss and update logging dicts.
    output: Model prediction [B,C,D,H,W] for tensor inputs or [nnodes, C_tar] for graph
    tar: target same shape as output
    logs: dict; Running log dictionary (updated in-place).
    loss_logs :dict; Dataset-type keyed loss log dict (updated in-place).
    """
    residuals = output - tar
    if output.ndim == 2:
        #[nnodes, C_tar] 
        # Differentiate between log and accumulation losses
        raw_loss = global_mean_pool(residuals.pow(2), graphdata.batch)/global_mean_pool(1e-7 + tar.pow(2), graphdata.batch) #B,C
        raw_loss = raw_loss.sqrt().mean()
        raw_rmse_loss = residuals.pow(2).mean(dim=0).sqrt().mean()
    else:
        ###full resolution###
        spatial_dims = tuple(range(output.ndim))[2:]
        # Differentiate between log and accumulation losses
        raw_loss = residuals.pow(2).mean(spatial_dims)/(1e-7+ tar.pow(2).mean(spatial_dims))
        raw_loss = raw_loss.sqrt().mean()
        raw_rmse_loss = residuals.pow(2).mean(spatial_dims).sqrt().mean()
    raw_l1_loss = F.l1_loss(output, tar)
    logs['valid_nrmse'] += raw_loss
    logs['valid_l1']    += raw_l1_loss
    logs['valid_rmse']  += raw_rmse_loss
    loss_dset_logs[dset_type]      += raw_loss
    loss_l1_dset_logs[dset_type]   += raw_l1_loss
    loss_rmse_dset_logs[dset_type] += raw_rmse_loss
    return
