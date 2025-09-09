# adapted from https://github.com/DeepLink-org/dlBLAS/blob/main/tests/kernels/test_grouped_gemm.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/grouped_gemm.py
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, group_A, group_B):
        group_size = len(group_A)
        return [torch.matmul(group_A[i], group_B[i]) for i in range(group_size)]


group_m = [1024, 512, 256, 128]
group_n = [1024, 512, 256, 128]
group_k = [1024, 512, 256, 128]
group_A = []
group_B = []
assert len(group_m) == len(group_n)
assert len(group_n) == len(group_k)
group_size = len(group_m)
DEVICE = 'cuda'


def get_inputs():
    for i in range(group_size):
        M = group_m[i]
        N = group_n[i]
        K = group_k[i]
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)
    return (group_A, group_B)


def get_init_inputs():
    return []