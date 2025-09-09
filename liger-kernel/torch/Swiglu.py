# https://github.com/linkedin/Liger-Kernel/blob/main/test/transformers/test_swiglu.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation function: x * silu(gate(x))."""
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)

class Model(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = True):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.w13 = nn.Linear(dim, hidden_dim * 2, bias=bias)  # W13 for gate & value
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)      # W2 for output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SwiGLU: W2(swiglu(W13(x)))"""
        return self.w2(swiglu(self.w13(x)))

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [dim, dim*2, False]