# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
import inspect
from functools import cache

@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args

def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        return _get_default_args(func._init_fn)

def ringX_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):

    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    global_ranks = [dist.get_global_rank(process_group, i) for i in range(world_size)]
    out, lse, lse_max = None, None, None
    q_buffers = [torch.empty_like(q).contiguous() for _ in range(2)]
    def flash_forward(q, k, v, causal):
        params = get_default_args(_flash_attn_forward).copy()
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                     {
                         "window_size_left": window_size[0],
                         "window_size_right": window_size[1],
                     }
            )
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        outputs = _flash_attn_forward(**params)
        if len(outputs) == 8:
            out, _, _, _, _, lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            out, lse, _, _ = outputs

        return out, lse
    
    q_buffers[0].copy_(q)
    current_buffer_idx = 0
    res_rank = global_ranks[world_size - 1]
    broadcast_work = dist.broadcast(q_buffers[current_buffer_idx], src=res_rank, group=process_group, async_op=True)

    prev_num = None
    prev_den = None
    prev_lse_max = None
    prev_reduce_num_work = None
    prev_reduce_den_work = None
    prev_rank = None

    for i in range(world_size - 1, -1, -1):
        if i < world_size - 1:
            prev_reduce_num_work.wait()
            prev_reduce_den_work.wait()

            if rank == prev_rank:
                out = prev_num.div_(prev_den.clamp(min=1e-8)).to(q.dtype)
                lse = (torch.log(prev_den) + prev_lse_max).squeeze(dim=-1).transpose(1, 2).contiguous()

            prev_num = None
            prev_den = None
            prev_lse_max = None
            prev_reduce_num_work = None
            prev_reduce_den_work = None
            prev_rank = None

        broadcast_work.wait()
        q_buffer = q_buffers[current_buffer_idx]

        if i > 0:
            next_idx = 1 - current_buffer_idx
            q_buffers[next_idx].copy_(q)
            res_rank_next = global_ranks[i - 1]
            next_broadcast_work = dist.broadcast(q_buffers[next_idx], src=res_rank_next, group=process_group, async_op=True)
        else:
            next_broadcast_work = None

        if not causal or rank <= i:
            loc_out, loc_lse = flash_forward(q_buffer, k, v, causal=(causal and rank == i))
            loc_out = loc_out.to(torch.float32)
            loc_lse = loc_lse.transpose(-2, -1).unsqueeze(dim=-1).contiguous()
            lse_max = loc_lse.clone().contiguous()
        else:
            lse_max.fill_(-torch.finfo(q.dtype).max)

        dist.all_reduce(lse_max, op=dist.ReduceOp.MAX, group=process_group)

        if not causal or rank <= i:
            den = torch.exp(loc_lse - lse_max)
            num = loc_out * den
        else:
            den.zero_()
            num.zero_()

        reduce_num_work = dist.reduce(num, dst=global_ranks[i], op=dist.ReduceOp.SUM, group=process_group, async_op=True)
        reduce_den_work = dist.reduce(den, dst=global_ranks[i], op=dist.ReduceOp.SUM, group=process_group, async_op=True)

        prev_num = num
        prev_den = den
        prev_lse_max = lse_max
        prev_reduce_num_work = reduce_num_work
        prev_reduce_den_work = reduce_den_work
        prev_rank = i

        current_buffer_idx = 1 - current_buffer_idx
        broadcast_work = next_broadcast_work

    if prev_reduce_num_work is not None:
        prev_reduce_num_work.wait()
    if prev_reduce_den_work is not None:
        prev_reduce_den_work.wait()

    if prev_rank is not None and rank == prev_rank:
        out = prev_num.div_(prev_den.clamp(min=1e-8)).to(q.dtype)
        lse = (torch.log(prev_den) + prev_lse_max).squeeze(dim=-1).transpose(1, 2).contiguous()

    return out, lse

def ringX_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    dq, dk, dv = None, None, None
    dq_buffer = torch.empty_like(q)
    dk_buffer = torch.empty_like(k)
    dv_buffer = torch.empty_like(v)
    kv = torch.cat([k,v], dim=0)
    kv_buffer = [torch.empty_like(kv) for _ in range(2)]
    k_size0 = k.shape[0]
    dkv_sum = [torch.empty_like(kv, dtype=torch.float32).contiguous() for _ in range(2)]
    bcast_handles = [None]*world_size
    reduce_handles = [None]*world_size
 
    def flash_backward(dout, q, k, v, out, softmax_lse, causal):
        params = get_default_args(_flash_attn_backward).copy()
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                     {
                         "window_size_left": window_size[0],
                         "window_size_right": window_size[1],
                     }
            )
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
        params.update(
            {
                "dout": dout,
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "softmax_lse": softmax_lse,
                "dq": dq_buffer,
                "dk": dk_buffer,
                "dv": dv_buffer,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
                "rng_state": rng_state,
            }
        )
        _flash_attn_backward(**params)

    kv_buffer[0][:k_size0].copy_(k)
    kv_buffer[0][k_size0:].copy_(v)
    res_rank_0 = dist.get_global_rank(process_group, 0)
    bcast_handles[0] = dist.broadcast(kv_buffer[0], src=res_rank_0, group=process_group, async_op=True)

    for i in range(1, world_size):
        prev_idx = (i-1)%2
        curr_idx = i%2 
        bcast_handles[i-1].wait()
        
        flash_backward(dout, q, kv_buffer[prev_idx][:k_size0], kv_buffer[prev_idx][k_size0:], out, softmax_lse, causal=False) 
        if dq is None: 
            dq = dq_buffer.to(torch.float32)
        else:
            dq += dq_buffer

        dkv_sum[prev_idx][:k_size0].copy_(dk_buffer)
        dkv_sum[prev_idx][k_size0:].copy_(dv_buffer)
        res_rank_i_1 = dist.get_global_rank(process_group, i - 1)
        reduce_handles[i-1] = dist.reduce(dkv_sum[prev_idx], dst=res_rank_i_1, op=dist.ReduceOp.SUM, group=process_group, async_op=True)

        kv_buffer[curr_idx][:k_size0].copy_(k)
        kv_buffer[curr_idx][k_size0:].copy_(v)
        res_rank_i = dist.get_global_rank(process_group, i)
        bcast_handles[i] = dist.broadcast(kv_buffer[curr_idx], src=res_rank_i, group=process_group, async_op=True)
            
        reduce_handles[i-1].wait()
        if rank == (i-1): 
            dk = dkv_sum[prev_idx][:k_size0].clone()
            dv = dkv_sum[prev_idx][k_size0:].clone()

    last_iter = world_size - 1
    prev_idx = last_iter%2 
    bcast_handles[last_iter].wait()
    flash_backward(dout, q, kv_buffer[prev_idx][:k_size0], kv_buffer[prev_idx][k_size0:], out, softmax_lse, causal=False) 
    dq += dq_buffer
    dkv_sum[prev_idx][:k_size0].copy_(dk_buffer)
    dkv_sum[prev_idx][k_size0:].copy_(dv_buffer)
    res_rank_last = dist.get_global_rank(process_group, last_iter)
    dist.reduce(dkv_sum[prev_idx], dst=res_rank_last, op=dist.ReduceOp.SUM, group=process_group)
    if rank == last_iter:
        dk = dkv_sum[prev_idx][:k_size0].clone()
        dv = dkv_sum[prev_idx][k_size0:].clone() 

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class RingXAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ringX_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ringX_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def ringX_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingXAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


# ---------------------------------------------------------------------------
# Variable-seqlen variant (non-causal): each rank may hold a different per-rank
# sq / skv. One small all_gather of [sq, skv] up front; receive buffers are
# allocated per iteration sized to the owner rank's seqlen. Pipelining and
# numerics match ringX_attn_func above.
# ---------------------------------------------------------------------------


def _gather_seqlens(q, k, process_group, world_size):
    meta = torch.tensor([q.shape[1], k.shape[1]], dtype=torch.int64, device=q.device)
    meta_all = [torch.empty_like(meta) for _ in range(world_size)]
    dist.all_gather(meta_all, meta, group=process_group)
    sq_list = [int(m[0].item()) for m in meta_all]
    skv_list = [int(m[1].item()) for m in meta_all]
    return sq_list, skv_list


def ringX_varlen_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    sq_list=None,
):
    if causal:
        raise NotImplementedError(
            "ringX_varlen_attn supports causal=False only; causal+varlen needs "
            "global cumulative offsets (switch to _flash_attn_varlen_forward)."
        )

    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    global_ranks = [dist.get_global_rank(process_group, i) for i in range(world_size)]

    if sq_list is None:
        sq_list, _ = _gather_seqlens(q, k, process_group, world_size)

    batch, _, num_heads, head_dim = q.shape

    def make_q_buf(seqlen):
        return torch.empty(batch, seqlen, num_heads, head_dim, dtype=q.dtype, device=q.device)

    def flash_forward(q_, k_, v_):
        params = get_default_args(_flash_attn_forward).copy()
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        params.update(
            {
                "q": q_,
                "k": k_,
                "v": v_,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": False,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        outputs = _flash_attn_forward(**params)
        if len(outputs) == 8:
            out, _, _, _, _, lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            out, lse, _, _ = outputs
        return out, lse

    first_owner = world_size - 1
    q_buf_curr = make_q_buf(sq_list[first_owner])
    if rank == first_owner:
        q_buf_curr.copy_(q)
    bcast_curr = dist.broadcast(
        q_buf_curr, src=global_ranks[first_owner], group=process_group, async_op=True
    )

    out, lse = None, None
    prev_num = prev_den = prev_lse_max = None
    prev_red_num = prev_red_den = None
    prev_owner = None

    for i in range(world_size - 1, -1, -1):
        if prev_red_num is not None:
            prev_red_num.wait()
            prev_red_den.wait()
            if rank == prev_owner:
                out = prev_num.div_(prev_den.clamp(min=1e-8)).to(q.dtype)
                lse = (torch.log(prev_den) + prev_lse_max).squeeze(dim=-1).transpose(1, 2).contiguous()
            prev_num = prev_den = prev_lse_max = None
            prev_red_num = prev_red_den = None
            prev_owner = None

        bcast_curr.wait()
        q_buf = q_buf_curr

        if i > 0:
            next_owner = i - 1
            q_buf_next = make_q_buf(sq_list[next_owner])
            if rank == next_owner:
                q_buf_next.copy_(q)
            bcast_next = dist.broadcast(
                q_buf_next, src=global_ranks[next_owner], group=process_group, async_op=True
            )
        else:
            q_buf_next = None
            bcast_next = None

        loc_out, loc_lse = flash_forward(q_buf, k, v)
        loc_out = loc_out.to(torch.float32)
        loc_lse = loc_lse.transpose(-2, -1).unsqueeze(dim=-1).contiguous()
        lse_max = loc_lse.clone().contiguous()

        dist.all_reduce(lse_max, op=dist.ReduceOp.MAX, group=process_group)

        den = torch.exp(loc_lse - lse_max)
        num = loc_out * den

        red_num = dist.reduce(
            num, dst=global_ranks[i], op=dist.ReduceOp.SUM, group=process_group, async_op=True
        )
        red_den = dist.reduce(
            den, dst=global_ranks[i], op=dist.ReduceOp.SUM, group=process_group, async_op=True
        )

        prev_num, prev_den, prev_lse_max = num, den, lse_max
        prev_red_num, prev_red_den = red_num, red_den
        prev_owner = i

        q_buf_curr = q_buf_next
        bcast_curr = bcast_next

    if prev_red_num is not None:
        prev_red_num.wait()
        prev_red_den.wait()
        if rank == prev_owner:
            out = prev_num.div_(prev_den.clamp(min=1e-8)).to(q.dtype)
            lse = (torch.log(prev_den) + prev_lse_max).squeeze(dim=-1).transpose(1, 2).contiguous()

    return out, lse


def ringX_varlen_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    skv_list=None,
):
    if causal:
        raise NotImplementedError("ringX_varlen_attn supports causal=False only.")

    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    global_ranks = [dist.get_global_rank(process_group, i) for i in range(world_size)]

    if skv_list is None:
        _, skv_list = _gather_seqlens(q, k, process_group, world_size)

    batch = k.shape[0]
    num_heads = k.shape[2]
    head_dim = k.shape[3]
    dq_buffer = torch.empty_like(q)

    def make_kv_buf(seqlen):
        return torch.empty(
            2 * batch, seqlen, num_heads, head_dim, dtype=k.dtype, device=k.device
        )

    def make_dkv_sum(seqlen):
        return torch.empty(
            2 * batch, seqlen, num_heads, head_dim, dtype=torch.float32, device=k.device
        )

    def flash_backward(dout_, q_, k_, v_, out_, softmax_lse_, dq_out, dk_out, dv_out):
        params = get_default_args(_flash_attn_backward).copy()
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        rng_state = torch.empty((2,), dtype=torch.int64, device=q_.device)
        params.update(
            {
                "dout": dout_,
                "q": q_,
                "k": k_,
                "v": v_,
                "out": out_,
                "softmax_lse": softmax_lse_,
                "dq": dq_out,
                "dk": dk_out,
                "dv": dv_out,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": False,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
                "rng_state": rng_state,
            }
        )
        _flash_attn_backward(**params)

    first = 0
    kv_buf_curr = make_kv_buf(skv_list[first])
    if rank == first:
        kv_buf_curr[:batch].copy_(k)
        kv_buf_curr[batch:].copy_(v)
    bcast_curr = dist.broadcast(
        kv_buf_curr, src=global_ranks[first], group=process_group, async_op=True
    )

    dq = None
    dk = dv = None
    prev_dkv_sum = None
    prev_red = None
    prev_owner = None

    for i in range(world_size):
        bcast_curr.wait()
        kv_buf = kv_buf_curr

        if i < world_size - 1:
            nxt = i + 1
            kv_buf_next = make_kv_buf(skv_list[nxt])
            if rank == nxt:
                kv_buf_next[:batch].copy_(k)
                kv_buf_next[batch:].copy_(v)
            bcast_next = dist.broadcast(
                kv_buf_next, src=global_ranks[nxt], group=process_group, async_op=True
            )
        else:
            kv_buf_next = None
            bcast_next = None

        dk_buf = torch.empty(
            batch, skv_list[i], num_heads, head_dim, dtype=k.dtype, device=k.device
        )
        dv_buf = torch.empty_like(dk_buf)
        flash_backward(
            dout, q,
            kv_buf[:batch], kv_buf[batch:],
            out, softmax_lse,
            dq_out=dq_buffer, dk_out=dk_buf, dv_out=dv_buf,
        )

        if dq is None:
            dq = dq_buffer.to(torch.float32)
        else:
            dq += dq_buffer

        if prev_red is not None:
            prev_red.wait()
            if rank == prev_owner:
                dk = prev_dkv_sum[:batch].clone()
                dv = prev_dkv_sum[batch:].clone()
            prev_dkv_sum = None
            prev_red = None
            prev_owner = None

        dkv_sum_i = make_dkv_sum(skv_list[i])
        dkv_sum_i[:batch].copy_(dk_buf)
        dkv_sum_i[batch:].copy_(dv_buf)
        red_i = dist.reduce(
            dkv_sum_i, dst=global_ranks[i], op=dist.ReduceOp.SUM,
            group=process_group, async_op=True,
        )

        prev_dkv_sum = dkv_sum_i
        prev_red = red_i
        prev_owner = i

        kv_buf_curr = kv_buf_next
        bcast_curr = bcast_next

    if prev_red is not None:
        prev_red.wait()
        if rank == prev_owner:
            dk = prev_dkv_sum[:batch].clone()
            dv = prev_dkv_sum[batch:].clone()

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class RingXVarlenAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()

        world_size = dist.get_world_size(group=group)
        sq_list, skv_list = _gather_seqlens(q, k, group, world_size)

        out, softmax_lse = ringX_varlen_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            sq_list=sq_list,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.sq_list = sq_list
        ctx.skv_list = skv_list
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ringX_varlen_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            skv_list=ctx.skv_list,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def ringX_varlen_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingXVarlenAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
