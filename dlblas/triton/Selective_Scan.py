# https://github.com/DeepLink-org/dlBLAS/blob/main/benchmarks/selective_scan.py
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import einsum
from dlblas.kernels.selective_scan import SelectiveScan

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, delta, A, B2, C, initial_state):
        return SelectiveScan.apply(x, delta, A, B2, C, initial_state)