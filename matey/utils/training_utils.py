import torch
import torch.distributed as dist


def autoregressive_rollout(model, inp, field_labels, bcs, imod, leadtime, input_control, tkhead_name, blockdict, tar, 
                           n_steps, inference=False, pushforward=True, sequence_parallel_group=None):
    device = inp.device
    min_lead = int(leadtime.min().item())
    # global minimum leadtime based on end of data
    if dist.is_initialized():
        min_lead_tensor =  leadtime.min()
        dist.all_reduce(min_lead_tensor, op=dist.ReduceOp.MIN)
        min_lead = min_lead_tensor
    # max rollout length allowed, based on min leadtime and warmup
    if inference:
        # Inference always uses the full leadtime
        max_rollout = max(1, min_lead)
    else:
        # Training:
        # FIXME: implement some warmup logic for ramping up rollout length
        #if self.params.auto_warmup and self.n_calls < 1000 and not self.params.resuming:
        #    max_rollout = max(1, int(min_lead * 0.5))
        max_rollout = max(1, min_lead)
    # set rollout_steps
    if dist.is_initialized():
        if dist.get_rank() == 0:
            rollout_steps = torch.tensor(
                [1 if max_rollout == 1 else torch.randint(1, int(max_rollout.item()), (1,)).item()],
                device=device,
                dtype=torch.int64,
            )
        else:
            rollout_steps = torch.zeros(1, device=device, dtype=torch.int64)

        dist.broadcast(rollout_steps, src=0)
        rollout_steps = rollout_steps.item()

    else:
        rollout_steps = 1 if max_rollout == 1 else torch.randint(1, int(max_rollout.item()), (1,)).item()
    outputs = []
    x_t = inp
    if rollout_steps == 1:
        pushforward = False  # no need for pushforward if only one step
    if inference or not pushforward:
        # Normal autoregressive rollout
        for t in range(rollout_steps):
            if input_control is not None:
                control_t = input_control[:, t:n_steps+t+1]
            else:
                control_t = None
            # Set leadtime to 1 for autoregressive training
            leadtime = torch.ones(leadtime.shape, device=leadtime.device, dtype=leadtime.dtype).view(-1, 1) #B,1
            output_t = model(
                x_t, field_labels, bcs, imod=imod,
                sequence_parallel_group=sequence_parallel_group,
                leadtime=leadtime, cond_input=control_t,
                tkhead_name=tkhead_name, blockdict=blockdict
            )
            outputs.append(output_t)
            x_t = torch.cat([x_t[1:], output_t.unsqueeze(0)], dim=0)
        tar = tar[:, rollout_steps-1:rollout_steps, :].squeeze(1) # B,C,D,H,W
        output = output_t # B,C,D,H,W
        # We could return all timesteps if desired if not using pushforward
        # output = torch.stack(outputs, dim=1) # B,T,C,D,H,W

    else:
        # Pushforward rollout
        with torch.no_grad():
            for t in range(rollout_steps - 1):
                if input_control is not None:
                    control_t = input_control[:, t:n_steps+t+1]
                else:
                    control_t = None
                # Set leadtime to 1 for autoregressive training
                leadtime = torch.ones(leadtime.shape, device=leadtime.device, dtype=leadtime.dtype).view(-1, 1) #B,1
                output_t = model(
                    x_t, field_labels, bcs, imod=imod,
                    sequence_parallel_group=sequence_parallel_group,
                    leadtime=leadtime, cond_input=control_t,
                    tkhead_name=tkhead_name, blockdict=blockdict
                )
                outputs.append(output_t)
                x_t = torch.cat([x_t[1:], output_t.unsqueeze(0)], dim=0)

        # last step with grad
        if input_control is not None:
            control_t = input_control[:, rollout_steps:n_steps+rollout_steps+1]
        else:
            control_t = None
        output_t = model(
            x_t, field_labels, bcs, imod=imod,
            sequence_parallel_group=sequence_parallel_group,
            leadtime=leadtime, cond_input=control_t,
            tkhead_name=tkhead_name, blockdict=blockdict
        )

        tar = tar[:, rollout_steps-1:rollout_steps, :].squeeze(1) # B,C,D,H,W
        output = output_t # B,C,D,H,W

    return output, tar, rollout_steps