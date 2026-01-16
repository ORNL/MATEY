import torch
import torch.distributed as dist
from .forward_options import ForwardOptionsBase, TrainOptionsBase
from contextlib import nullcontext

def preprocess_target(leadtime, ramping_warmup = False):
    """
    #inputs:
    #  leadtime: B, 1
    #return 
    # rollout_steps: int, actual leadtime used in training/inference after synchronziation across ranks
    """
    min_lead = int(leadtime.min().item())
    # global minimum leadtime based on end of data
    if dist.is_initialized():
        min_lead_tensor =  leadtime.min()
        dist.all_reduce(min_lead_tensor, op=dist.ReduceOp.MIN)
        min_lead = min_lead_tensor
    # max rollout length allowed, based on min leadtime and warmup
    if ramping_warmup:
        # Training:
        # FIXME: implement some warmup logic for ramping up rollout length
        #if self.params.auto_warmup and self.n_calls < 1000 and not self.params.resuming:
        max_rollout = max(1, int(min_lead * 0.5))
    else:
        max_rollout = max(1, min_lead)
    # set rollout_steps
    if dist.is_initialized():
        if dist.get_rank() == 0:
            rollout_steps = torch.randint(1, int(max_rollout.item()+1), (1,)).to(leadtime.device)
        else:
            rollout_steps = torch.zeros(1, device=leadtime.device, dtype=torch.int64)

        dist.broadcast(rollout_steps, src=0)
        rollout_steps = rollout_steps.item()
    else:
        rollout_steps = torch.randint(1, int(max_rollout.item()+1), (1,)).item()

    return rollout_steps

def autoregressive_rollout(model, inp, field_labels, bcs, opts: ForwardOptionsBase,  pushforward=True):  
    rollout_steps = preprocess_target(opts.leadtime) 
    x_t = inp
    n_steps = inp.shape[0]
    ctx = torch.no_grad() if pushforward else nullcontext()
    cond_input = opts.cond_input.clone()
    with ctx:
        for t in range(rollout_steps - 1):
            cond_input_t = cond_input[:, t:n_steps + t + 1] if cond_input is not None else None
            opts.cond_input = cond_input_t
            opts.leadtime = opts.leadtime * 0 + 1 #set leadtime to 1 for autoregressive training
            output_t = model(x_t, field_labels, bcs, opts)
            x_t = torch.cat([x_t[1:], output_t.unsqueeze(0)], dim=0)

    cond_input_t = cond_input[:, rollout_steps:n_steps+rollout_steps+1] if cond_input is not None else None
    opts.cond_input = cond_input_t
    output = model(x_t, field_labels, bcs, opts)# B,C,D,H,W

    return output, rollout_steps