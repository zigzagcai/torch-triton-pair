# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/benchmarks/partial_rotary_emb.py
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    # out1.copy_(x1 * cos - x2 * sin)
    # out2.copy_(x2 * cos + x1 * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# @torch.compile()
def partial_rotary_emb(q, k_pe, kv, cos, sin):
    bsz, q_len, num_heads, q_head_dim = q.shape
    assert bsz == k_pe.shape[0] and q_len == k_pe.shape[1] and 1 == k_pe.shape[2]
    qk_rope_head_dim = k_pe.shape[3]
    qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    assert bsz == kv.shape[0] and q_len == kv.shape[1] and num_heads == kv.shape[2]
    v_head_dim = kv.shape[3] - qk_nope_head_dim
    assert q_len == cos.shape[1] and qk_rope_head_dim == cos.shape[2]
    assert cos.shape == sin.shape

    q = q.transpose(1, 2)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim).transpose(1, 2)
    kv = kv.transpose(1, 2)
    k_nope, v = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

    q_out = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    q_out[:, :, :, :qk_nope_head_dim] = q_nope
    q_out[:, :, :, qk_nope_head_dim:] = q_pe
    k = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    k[:, :, :, :qk_nope_head_dim] = k_nope
    k[:, :, :, qk_nope_head_dim:] = k_pe

    if q_head_dim != v_head_dim:
        v = F.pad(v, [0, q_head_dim - v_head_dim])

    q_out = q_out.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    kv_out = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
    return q_out, kv_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_pe, kv, cos, sin):
        return partial_rotary_emb(q, k_pe, kv, cos, sin)


num_heads = 128
nope_head_dim = 128
rope_head_dim = 64
v_head_dim = 128
q_head_dim = nope_head_dim + rope_head_dim
bsz, q_len = 1, 4096
device_ = torch.device('cuda')


def get_inputs():
    q = torch.randn((bsz, q_len, num_heads, q_head_dim), dtype=torch.bfloat16, device=device_)
    k_pe = torch.randn((bsz, q_len, 1, rope_head_dim), dtype=torch.bfloat16, device=device_)
    kv = torch.randn(
        (bsz, q_len, num_heads, nope_head_dim + v_head_dim),
        dtype=torch.bfloat16,
        device=device_,
    )
    cos = torch.randn((bsz, q_len, rope_head_dim), dtype=torch.bfloat16, device=device_)
    sin = torch.randn((bsz, q_len, rope_head_dim), dtype=torch.bfloat16, device=device_)
    return [q, k_pe, kv, cos, sin]

def get_init_inputs():
    return []  # No special initialization inputs needed