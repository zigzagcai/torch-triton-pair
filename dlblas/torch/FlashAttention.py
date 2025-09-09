# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/flash_attention_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3).cpu(),
            key.permute(0, 2, 1, 3).cpu(),
            value.permute(0, 2, 1, 3).cpu(),
        ).permute(0, 2, 1, 3)


seq_len = 25600
heads = 32
dim = 64
dtype = torch.float16
device = torch.device('cuda')

def get_inputs():
    query = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device)
    key = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device)
    value = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device)
    return [query, key, value]

def get_init_inputs():
    return []  # No special initialization inputs needed