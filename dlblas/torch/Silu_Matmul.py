# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/tests/kernels/test_activation.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/activation.py
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gate, up = x.chunk(2, -1)
        gate = torch.nn.functional.silu(gate)
        return gate * up


seqlen = 256
feat_size = 4096


def get_inputs():
    x = torch.rand(seqlen, feat_size, dtype=torch.float16, device='cuda')
    return [x]


def get_init_inputs():
    return []