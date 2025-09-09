# adapted from https://github.com/DeepLink-org/dlBLAS/blob/main/tests/kernels/test_context_flashattention_nopad.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/context_flashattention_nopad.py
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def torch_att(q, q_rope, kv, kv_rope, bs, seqlen, num_head, q_head_dim, rope_head_dim, device):

    xq = torch.cat([q, q_rope], dim=2).view(bs, seqlen, num_head, -1)
    xk = torch.cat([kv, kv_rope], dim=2).view(bs, seqlen, 1, -1)
    xv = kv.view(bs, seqlen, 1, -1)

    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).to(device=device)
    mask[mask == 0.0] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(q_head_dim + rope_head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, q_head_dim)
    return output


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
        return torch_att(q, q_rope, kv, kv_rope, self.Z, self.N_CTX, self.H, self.D_HEAD, self.ROPE_HEAD, self.device)


Z = 1
H = 6
N_CTX = 500
D_HEAD = 128
ROPE_HEAD = 64
dtype = torch.float16
device = 'cuda'


def get_inputs():
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device=device).normal_(mean=0.3, std=0.2)
    q_rope = torch.empty((Z * N_CTX, H, ROPE_HEAD), dtype=dtype, device=device).normal_(mean=0.3, std=0.2)
    kv = torch.empty((Z * N_CTX, 1, D_HEAD), dtype=dtype, device=device).normal_(mean=0.3, std=0.2)
    kv_rope = torch.empty((Z * N_CTX, 1, ROPE_HEAD), dtype=dtype, device=device).normal_(mean=0.3, std=0.2)
    return [q, q_rope, kv, kv_rope]


def get_init_inputs():
    return [Z, H, N_CTX, D_HEAD, ROPE_HEAD, dtype, device]
