import torch
import importlib
import operator
import triton
import triton.language as tl
import functools

from typing import Optional, Callable
from packaging.version import Version


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)

def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

def is_hip() -> bool:
    return torch.version.hip is not None

def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def get_amp_custom_fwd_bwd() -> Callable:
    device = infer_device()
    if compare_version("torch", operator.ge, "2.4.0"):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd


amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()

@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,  # set it as constexpr since reduction is always known at compile time
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    weight_ptr: Pointer to weight tensor.
    loss_ptr: Pointer to tensor to store the loss.
    z_loss_ptr: Pointer to tensor to store the z loss. No operation if RETURN_Z_LOSS is 0.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (float): The number of non-ignored elements in the batch.
    sum_non_ignore_weight (float): The sum of non-ignored target's weights in the batch.
    weight_sum (float): The sum of weight tensor.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    reduction (str): The string for the reduction to apply
    softcap (float): The upper threshold for scaling logits to the range (-softcap, +softcap).
    RETURN_Z_LOSS (int): The boolean value to decide whether storing z loss to z_loss_ptr or not. It must be 0 or 1.
    BLOCK_SIZE (int): The block size for Triton operations.
    HAS_WEIGHT (bool): The boolean value to determine whether assigning weight to each of the classes.
    HAS_SOFTCAPPING (bool): The boolean value to determine whether applying soft-capping or not.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)  # we need to store the original value of X_y for the loss calculation
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0))
            else:
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # log (sum(e^(X_i))) = log (sum(e ^ (max(X) * e ^ (X_i - max(X)))))
    #                    = log (e^(max(X)) * sum(e ^ (X_i - max(X))))
    #                    = max(X) + log (sum(e ^ (X_i - max(X)))) = m + log d
    lse = m + tl.log(d)

    # 4. [Online Softmax] Second pass: compute gradients
    # For 'mean' reduction, gradients are normalized by number of non-ignored elements (N)
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # For label smoothing:
    # dx_i = (softmax(x_i) - label_smoothing / V) / N, V = n_cols, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    # With Z loss:
    # dx_i = ((1 + 2 * lse_square_scale * lse) * softmax(x_i) - label_smoothing / V) / N, i != y
    # dx_y = dx_i - (1 - label_smoothing) / N
    # For 'sum' reduction, no normalization is applied:
    # dx_y = softmax(x_y) - 1
    # dx_i = softmax(x_i), for i ≠ y

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            intermediate = tanh(X_block / softcap)
            X_block = softcap * intermediate

        if not HAS_WEIGHT:
            # softmax(x_i)
            X_block = tl.exp(X_block - m) / d
            # derivative of z-loss: 2 * lse_square_scale * lse * softmax(x_i)
            X_block += 2 * lse_square_scale * lse * X_block
            # smoothing term
            X_block += -eps
            # special handle dx_y
            X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
            # reduction scale
            if reduction == "mean":
                X_block = X_block / n_non_ignore
        else:
            weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
            softmax_X = tl.exp(X_block - m) / d
            # derivative of original_loss
            dloss_ori = (1 - label_smoothing) * softmax_X
            # specially handle dx_y
            dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
            dloss_ori = dloss_ori * weight_y
            # derivative of smooth_loss
            dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
            # derivative of z-loss
            dz_loss = 2 * lse_square_scale * lse * softmax_X
            # reduction scale
            if reduction == "mean":
                dloss_ori = dloss_ori / sum_non_ignore_weight
                dloss_smooth = dloss_smooth / sum_non_ignore_weight
                # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
                dz_loss = dz_loss / n_non_ignore
            # derivative of total_loss
            X_block = dloss_ori + dloss_smooth + dz_loss

        # chain rule softcapping
        # d(softcap * tanh(x / softcap)) = (1 - tanh^2(x / softcap))
        if HAS_SOFTCAPPING:
            X_block = X_block * (1 - intermediate * intermediate)

        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    #      = X_y - m - log d = X_y - lse
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = lse - ori_X_y
    if HAS_WEIGHT:
        loss = weight_y * loss

    # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (sum(-eps * x_i) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
    # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    # An auxiliary loss, z_loss
    # Refer to Page14 Loss function section in the paper PaLM: https://www.jmlr.org/papers/v24/22-1144.html
    z_loss = lse_square_scale * lse * lse
    # Normalize the loss by the number of non-ignored elements if reduction is "mean"
    if reduction == "mean":
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
        z_loss = z_loss / n_non_ignore
    loss += z_loss

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_input = torch.zeros_like(_input, device=device)

    # we use fp32 for loss and gradients accumulator
    if accum_dtype is None:
        grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    else:
        grad_weight = torch.zeros_like(weight, dtype=accum_dtype, device=device) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=accum_dtype, device=device) if bias is not None else None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # when doing matmul, use the original precision
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # Compute predicted probabilities for token scaling if needed
        if use_token_scaling:
            # Compute softmax probabilities for scaling
            # We need to compute this before the cross entropy kernel modifies logits_chunk
            logits_for_softmax = logits_chunk.detach().clone()  # Detach to avoid gradient flow
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)

            # Compute softmax to get predicted probabilities
            probs = torch.softmax(logits_for_softmax, dim=-1)

            # Get predicted probabilities for token scaling, handling ignored targets
            valid_target_mask = target_chunk != ignore_index
            valid_targets = target_chunk[valid_target_mask]

            if len(valid_targets) > 0:
                # Gather probabilities only for valid targets
                valid_probs = probs[valid_target_mask]
                pred_probs_valid = torch.gather(valid_probs, -1, valid_targets.unsqueeze(-1)).squeeze(-1)

                # Create full tensor with zeros for ignored targets
                pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype, device=probs.device)
                pred_probs[valid_target_mask] = pred_probs_valid
            else:
                # All targets are ignored
                pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype, device=probs.device)

            # Store the scaling factors
            scaling_factors = pred_probs.detach()  # Detach to ensure no gradient flow

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None

        # ensure _input and target are contiguous
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            weight_ptr=ce_weight,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
            weight_sum=ce_weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # Apply token scaling if requested
        if use_token_scaling:
            loss_1d_slice = loss_1d_slice * scaling_factors
            if return_z_loss:
                z_loss_1d_slice = z_loss_1d_slice * scaling_factors

        loss_1d[start_idx:end_idx] = loss_1d_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice
        grad_logits_chunk = logits_chunk  # chunk_size x V

        # Apply token scaling to gradients if requested
        if use_token_scaling:
            # Expand scaling factors to match gradient dimensions
            scaling_factors_expanded = scaling_factors.unsqueeze(-1)  # chunk_size x 1
            grad_logits_chunk = grad_logits_chunk * scaling_factors_expanded

        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()

        if bias is not None:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # Need extra calculations for backward if reduction=='none'. Not supporting reduction='none' now.
    # if reduction == "none":
    #     loss = loss_1d
    #     z_loss = z_loss_1d if return_z_loss else None

    if reduction == "none":
        # Return per-token losses
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None

    # Cast back to original dtype
    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return loss, z_loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # handle grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
        accum_dtype=None,
        use_token_scaling: bool = False,
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ce_weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index: the index to ignore in the target
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: reduction to apply
        accum_dtype (torch.dtype): the dtype of intermediate result buffers for weight and bias gradient accumulations.
            Recommended to set `accum_dtype` to higher precision, e.g. `torch.float32`, if the training is unstable with original dtype. Default: `None`, performing accumulations in original dtype
        use_token_scaling (bool): whether to scale each token's loss by its predicted probability (detached).
            When True, each token's loss is multiplied by the model's predicted probability for that token's true class.
            Default: False.
        """

        loss, z_loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ce_weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
            accum_dtype=accum_dtype,
            use_token_scaling=use_token_scaling,
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # use_token_scaling
        )

class LigerFusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
        }, f"reduction must be 'mean' or 'sum'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.accum_dtype = accum_dtype
        self.use_token_scaling = use_token_scaling

    def forward(self, lin_weight, _input, target, bias=None):
        loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ce_weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
            self.accum_dtype,
            self.use_token_scaling,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss


class Model(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype=None,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ce_weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
            accum_dtype=accum_dtype,
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y, self.lin.bias)