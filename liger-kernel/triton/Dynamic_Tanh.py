# adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/transformers/test_dyt.py
import torch
import torch.nn as nn
import operator
import importlib
import functools
import triton
import triton.language as tl
from packaging.version import Version
from typing import Callable


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper

def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


# @triton.autotune([triton.Config({"BLOCK_N":bn}, num_stages=ns, num_warps=nw)
#                   for bn in [1024, 2048, 4096]
#                   for ns in [1,2,4]
#                   for nw in [4, 8, 16, 32]
#                   ],
#                   key=['N'])
@triton.jit
def _dyt_fwd_kernel(X, Y, Alpha, Gamma, Beta, HAVE_BETA: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr = 1024):
    col = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    row_id = tl.cast(tl.program_id(1), tl.int64)

    X += row_id * N
    Y += row_id * N
    alpha = tl.load(Alpha).to(tl.float32)

    gamma = tl.load(Gamma + col, mask=mask, other=0.0).to(tl.float32)

    x = tl.load(X + col, mask=mask, other=0.0).to(tl.float32)

    tanh_x = tanh(alpha * x)
    y = tanh_x * gamma
    if HAVE_BETA:
        beta = tl.load(Beta + col, mask=mask, other=0.0).to(tl.float32)
        y += beta
    tl.store(Y + col, y, mask=mask)


# @triton.autotune([triton.Config({"BLOCK_N":bn}, num_stages=ns, num_warps=nw)
#                   for bn in [1024, 2048, 4096]
#                   for ns in [1,2,4]
#                   for nw in [4, 8, 16]
#                   ],
#                   key=['N'])
@triton.jit
def _dyt_bwd_kernel(
    DY, DX, DA, DG, DB, X, Alpha, Gamma, HAVE_BETA: tl.constexpr, M, N: tl.constexpr, BLOCK_N: tl.constexpr = 1024
):
    col = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    start_row_id = tl.cast(tl.program_id(1), tl.int64)

    alpha = tl.load(Alpha).to(tl.float32)
    da = 0.0
    gamma = tl.load(Gamma + col, mask=mask, other=0.0).to(tl.float32)
    dg = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAVE_BETA:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row_id in range(start_row_id, M, tl.num_programs(1)):
        x = tl.load(X + row_id * N + col, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row_id * N + col, mask=mask, other=0.0).to(tl.float32)
        tanh_x = tanh(alpha * x)
        if HAVE_BETA:
            db += dy
        dg += dy * tanh_x
        tmp = (1 - tanh_x * tanh_x) * dy * gamma
        da += tl.sum(x * tmp, 0)
        dx = alpha * tmp
        tl.store(DX + row_id * N + col, dx, mask=mask)

    tl.store(DG + start_row_id * N + col, dg, mask=mask)
    if HAVE_BETA:
        tl.store(DB + start_row_id * N + col, db, mask=mask)
    tl.store(DA + start_row_id * tl.cdiv(N, 512) + tl.program_id(0), da)


def liger_dyt_fwd(x, alpha, gamma, beta):
    assert x.is_contiguous()
    HAVE_BETA = True if beta is not None else False
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape

    y = torch.empty_like(x)

    if N >= 4096:
        kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 2048), "num_warps": 4, "num_stages": 1}
    else:
        kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 1024), "num_warps": 4, "num_stages": 1}

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)
    _dyt_fwd_kernel[(grid)](
        x,
        y,
        alpha,
        gamma,
        beta,
        HAVE_BETA,
        N,
        **kwargs,
    )
    return y.view(input_shape)


def liger_dyt_bwd(dy, x, alpha, gamma, beta):
    assert dy.is_contiguous()
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape
    HAVE_BETA = True if beta is not None else False

    device = "cuda"
    if device == "cuda":
        NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    elif device == "xpu":
        NUM_SMS = torch.xpu.get_device_properties(x.device).gpu_subslice_count

    da = torch.zeros(NUM_SMS, triton.cdiv(N, 512), dtype=torch.float32, device=x.device)
    dg = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device)
    db = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device) if HAVE_BETA else None
    dx = torch.empty_like(dy)

    kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 1024), "num_warps": 8, "num_stages": 2}
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), NUM_SMS)
    _dyt_bwd_kernel[grid](dy, dx, da, dg, db, x, alpha, gamma, HAVE_BETA, M, N, **kwargs)
    if HAVE_BETA:
        db = db.sum(0).to(x.dtype)
    dg = dg.sum(0).to(gamma.dtype)
    da = da.sum().to(x.dtype).unsqueeze(0)
    return dx.view(input_shape), da, dg, db


class LigerDyTFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, alpha, gamma, beta):
        y = liger_dyt_fwd(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dy):
        x, alpha, gamma, beta = ctx.saved_tensors
        dx, dalpha, dgamma, dbeta = liger_dyt_bwd(dy, x, alpha, gamma, beta)
        return dx, dalpha, dgamma, dbeta


class Model(nn.Module):
    def __init__(self, hidden_size, beta=True, init_alpha=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_alpha = init_alpha
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = None
        if beta:
            self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return LigerDyTFunction.apply(x, self.alpha, self.gamma, self.beta)

    def extra_repr(self):
        return f"{self.hidden_size}, init_alpha={self.init_alpha}, beta={self.beta}"
