# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/tests/kernels/test_layernorm_gated.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/layernorm_gated.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def torch_rmsnorm_gated(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, '... (g d) -> ... g d', d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, '... g d -> ... (g d)') * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pt, weight_pt, bias_pt, z_pt, group_size):
        return torch_rmsnorm_gated(x_pt, weight_pt, bias_pt, z=z_pt, eps=1e-5, group_size=group_size, norm_before_gate=True, upcast=False)

def get_inputs():
    device_ = 'cuda'
    group_size = 64
    # set seed
    torch.random.manual_seed(0)
    batch = 16
    seqlen = 1024
    d = 2048
    dtype, wtype = torch.float32, torch.float32
    x = torch.randn(batch, seqlen, d, dtype=dtype, device=device_, requires_grad=True)
    z = torch.randn(batch, seqlen, d, dtype=dtype, device=device_, requires_grad=True)
    weight = torch.randn(d, dtype=wtype, device=device_, requires_grad=True)
    bias = torch.randn(d, dtype=wtype, device=device_, requires_grad=True)
    x_pt = x.detach().clone().requires_grad_()
    z_pt = z.detach().clone().requires_grad_() if z is not None else None
    weight_pt = weight.detach().clone().requires_grad_()
    bias_pt = bias.detach().clone().requires_grad_() if bias is not None else None
    return [x_pt, weight_pt, bias_pt, z_pt, group_size]

def get_init_inputs():
    return []