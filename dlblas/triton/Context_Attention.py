# adapted from https://github.com/DeepLink-org/dlBLAS/blob/main/tests/kernels/test_context_flashattention_nopad.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/context_flashattention_nopad.py
import math
import torch
import triton
import functools
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import triton.language as tl


def is_muxi():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'maca'

@functools.lru_cache
def is_tesla():
    try:
        return 'Tesla' in torch.cuda.get_device_name(0)
    except Exception:
        return False

@functools.lru_cache
def is_npu():
    try:
        return torch.npu.is_available()
    except Exception:
        return False

def get_tl_exp():
    if is_npu():
        from triton.language.math import exp as tl_exp
    elif triton.__version__ >= "3.0.0":
        from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
    else:
        from triton.language.math import fast_expf as tl_exp
    return tl_exp


TESLA = is_tesla or is_muxi()

tl_exp = get_tl_exp()


@triton.jit
def _fwd_kernel(
    Q_nope,
    Q_rope,
    KV_nope,
    KV_rope,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录当前输入的真实长度
    Out,
    Req_to_tokens,
    B_req_idx,
    stride_q_bs,
    stride_q_h,
    stride_q_d,
    stride_q_rope_bs,
    stride_q_rope_h,
    stride_q_rope_d,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_kv_rope_bs,
    stride_kv_rope_h,
    stride_kv_rope_d,
    stride_obs,
    stride_oh,
    stride_od,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    kv_group_num,
    b_prompt_cache_len,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # cur_kv_head = cur_head // kv_group_num # cur_kv_head 永远是0
    cur_kv_head = 0

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_q_bs + cur_head * stride_q_h +
             offs_d[None, :] * stride_q_d)
    off_q_rope = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_q_rope_bs + cur_head * stride_q_rope_h +
                  offs_rope_d[None, :] * stride_q_rope_d)

    q = tl.load(Q_nope + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    q_rope = tl.load(Q_rope + off_q_rope, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum((start_m + 1) * BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc,
            other=0,
        )
        off_kv = kv_loc[None, :] * stride_kv_bs + cur_kv_head * stride_kv_h + offs_d[:, None] * stride_kv_d
        off_kv_rope = (kv_loc[None, :] * stride_kv_rope_bs + cur_kv_head * stride_kv_rope_h +
                       offs_rope_d[:, None] * stride_kv_rope_d)
        kv = tl.load(KV_nope + off_kv, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        kv_rope = tl.load(KV_rope + off_kv_rope, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, kv)
        qk += tl.dot(q_rope, kv_rope)

        qk *= sm_scale
        qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :], qk, float('-100000000.0'))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.trans(kv)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh +
             offs_d[None, :] * stride_od)
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return


#   "qk_nope_head_dim": 128,
#   "qk_rope_head_dim": 64,


@torch.no_grad()
def context_attention_fwd(
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    o,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    max_input_len,
    req_to_token_indexs,
    softmax_scale,
):

    BLOCK = 128 if not TESLA else 64
    q_nope_dim = q_nope.shape[-1]
    q_rope_dim = q_rope.shape[-1]
    assert q_nope_dim == kv_nope.shape[-1]
    assert q_rope_dim == kv_rope.shape[-1]
    assert q_nope_dim in {16, 32, 64, 128, 256, 512}
    assert q_rope_dim in {16, 32, 64, 128, 256}

    if q_nope_dim >= 512:
        BLOCK = 64 if not TESLA else 32
    else:
        BLOCK = 128 if not TESLA else 64

    if q_nope.dtype == torch.float32:
        BLOCK = BLOCK // 4

    sm_scale = softmax_scale
    batch, head = b_seq_len.shape[0], q_nope.shape[1]
    kv_group_num = q_nope.shape[1]  # deepseekv2 的 group 就是q的head数量，类似于MQA

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,
    num_warps = 4 if q_nope_dim <= 64 else 8

    _fwd_kernel[grid](
        q_nope,
        q_rope,
        kv_nope,
        kv_rope,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        req_to_token_indexs,
        b_req_idx,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        q_rope.stride(2),
        kv_nope.stride(0),
        kv_nope.stride(1),
        kv_nope.stride(2),
        kv_rope.stride(0),
        kv_rope.stride(1),
        kv_rope.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        kv_group_num=kv_group_num,
        b_prompt_cache_len=b_prompt_cache_len,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=q_nope_dim,
        BLOCK_ROPE_DMODEL=q_rope_dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


class Model(nn.Module):
    def __init__(self, Z, H, N_CTX, D_HEAD, ROPE_HEAD, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.Z = Z
        self.H = H
        self.N_CTX = N_CTX
        self.D_HEAD = D_HEAD
        self.ROPE_HEAD = ROPE_HEAD
        self.dtype = dtype
        self.device = device

    def forward(self, q, q_rope, kv, kv_rope):
        Z = self.Z
        H = self.H
        N_CTX = self.N_CTX
        D_HEAD = self.D_HEAD
        ROPE_HEAD = self.ROPE_HEAD
        dtype = self.dtype
        device = self.device

        req_to_token_indexs = torch.zeros((10, Z * N_CTX), dtype=torch.int32, device=device)
        b_start_loc = torch.zeros((Z, ), dtype=torch.int32, device=device)
        b_seq_len = torch.ones((Z,), dtype=torch.int32, device=device)
        b_req_idx = torch.ones((Z,), dtype=torch.int32, device=device)
        b_prompt_cache_len = torch.zeros(1, dtype=torch.int32, device=device)
        b_prompt_cache_len[0] = 0
        prompt_cache_len = 0
        b_seq_len[0] = N_CTX
        b_req_idx[0] = 0
        req_to_token_indexs[0][:prompt_cache_len + N_CTX] = torch.tensor(
            np.arange(prompt_cache_len + N_CTX),
            dtype=torch.int32
        ).cuda()
        softmax_scale = 1 / math.sqrt(D_HEAD + ROPE_HEAD)
        
        o1 = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device=device).normal_(mean=0.7, std=0.2)
        context_attention_fwd(
            q,
            q_rope,
            kv,
            kv_rope,
            o1,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            b_prompt_cache_len,
            N_CTX,
            req_to_token_indexs,
            softmax_scale,
        )
        return o1
