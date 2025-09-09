import torch
import triton
import triton.language as tl
import functools
import importlib
import operator

from typing import Optional, Callable
from packaging.version import Version


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
def _jsd_kernel(
    X_ptr,  # input in logspace, X = log Q
    X_stride,
    Y_ptr,  # ground truth in logspace, Y = log P
    Y_stride,
    loss_ptr,
    loss_stride,
    dX_ptr,
    dX_stride,
    label_ptr,
    beta: tl.constexpr,
    n_non_ignore: int,
    ignore_index: tl.constexpr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    # JSD(P || Q) = (KL(P || M) + KL(Q || M)) / 2, M = (1/2) * (P + Q) = (1/2) * (e ^ Y + e ^ X)
    #             = sum(P * log P + Q * log Q - 2 * M * log M) / 2
    #             = sum(e ^ Y * Y + e ^ X * X - 2 * M * log M) / 2
    # grad_x_i = 0.5 * Q * (X - log_M)
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        if beta == 0.0:  # forward KL
            Y_max = tl.max(Y, axis=0)
            Y_shifted = Y - Y_max
            Y_prob = tl.exp(Y_shifted) * tl.exp(Y_max)  # Compensate for the shift
            loss = Y_prob * (Y - X)
            dX = -Y_prob
        elif beta == 1.0:  # reverse KL
            X_max = tl.max(X, axis=0)
            X_shifted = X - X_max
            X_prob = tl.exp(X_shifted) * tl.exp(X_max)  # Compensate for the shift
            loss = X_prob * (X - Y)
            dX = loss + X_prob
        else:
            max_val = tl.maximum(tl.max(X, axis=0), tl.max(Y, axis=0))
            X_shifted = X - max_val
            Y_shifted = Y - max_val

            # Pre-compute exp(max_val) since it's used twice
            exp_max = tl.exp(max_val)

            # Compute exp terms with compensation
            Q = tl.exp(X_shifted) * exp_max  # = exp(X)
            P = tl.exp(Y_shifted) * exp_max  # = exp(Y)

            # Pre-compute common terms
            beta_P = beta * P
            one_minus_beta_Q = (1 - beta) * Q
            M = beta_P + one_minus_beta_Q
            log_M = tl.log(M)  # No need to compensate as M is already in original scale

            loss = beta_P * Y + one_minus_beta_Q * X - M * log_M
            dX = one_minus_beta_Q * (X - log_M)

        # Pre-compute scaling factor
        scale = 1.0 / n_non_ignore
        loss = loss * scale
        dX = dX * scale

        tl.store(loss_ptr + offsets, loss, mask=mask)
        tl.store(dX_ptr + offsets, dX, mask=mask)

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2


def fused_linear_jsd_forward(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    shift_labels,
    jsd_beta,
    ignore_index,
    has_label,
    temperature,
):
    device = student_input.device
    dtype = student_input.dtype

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = student_input.shape
    V = student_weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight = torch.zeros_like(student_weight, device=device) if student_weight.requires_grad else None
    grad_input = torch.zeros_like(student_input)
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros((BT, V), dtype=torch.float32, device=device)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        # chunk both inputs, shape: chunk_size x H
        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]

        # shape: chunk_size x V
        # For anything starting from logits to the final JSD loss, we do computation
        # in FP32 to avoid losing numerical stability.
        student_logits_chunk = (student_input_chunk @ student_weight.t()).to(torch.float32)
        teacher_logits_chunk = (teacher_input_chunk @ teacher_weight.t()).to(torch.float32)
        chunk_n_rows = student_logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size
        # log-softmax with temperature
        student_logits_chunk = student_logits_chunk / temperature
        teacher_logits_chunk = teacher_logits_chunk / temperature
        student_prob_chunk = torch.log_softmax(student_logits_chunk, dim=-1)
        teacher_prob_chunk = torch.log_softmax(teacher_logits_chunk, dim=-1)

        # ensure _input and target are contiguous
        student_prob_chunk = student_prob_chunk.contiguous()
        teacher_prob_chunk = teacher_prob_chunk.contiguous()

        # Here we calculate the gradient of prob_chunk in place so we can save memory.
        _jsd_kernel[(chunk_n_rows,)](
            X_ptr=student_prob_chunk,
            X_stride=student_prob_chunk.stride(-2),
            Y_ptr=teacher_prob_chunk,
            Y_stride=teacher_prob_chunk.stride(-2),
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-2),
            dX_ptr=student_prob_chunk,
            dX_stride=student_prob_chunk.stride(-2),
            label_ptr=(
                shift_labels[start_idx:end_idx] if has_label else torch.empty(1, device=device)
            ),  # dummy ptr if no label
            beta=jsd_beta,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            n_cols=V,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_LABEL=has_label,
        )
        loss_1d[start_idx:end_idx] = loss_1d_slice
        # gradients of prob_chunk in place, shape: chunk_size x V
        # gradients of logits_chunk in place, shape: chunk_size x V
        student_logits_chunk = (
            student_prob_chunk
            - torch.softmax(student_logits_chunk, dim=-1)
            * student_prob_chunk.sum(dim=-1, keepdim=True).broadcast_to(student_prob_chunk.shape)
        ) / temperature
        # now we traverse back to grad w.r.t. input to `lm_head` and grad
        # w.r.t. `lm_head` which should be computed in original dtype
        student_logits_chunk = student_logits_chunk.to(dtype)
        grad_input[start_idx:end_idx] = student_logits_chunk @ student_weight

        if grad_weight is not None:
            grad_weight.add_(student_logits_chunk.t() @ student_input_chunk)

    loss = torch.sum(loss_1d)
    return loss, grad_input, grad_weight


def fused_linear_jsd_backward(grad_output, grad_input, grad_weight):
    # If JSD is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
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

    return grad_input, grad_weight


class LigerFusedLinearJSDFunction(torch.autograd.Function):
    """
    Fusing the last linear layer with generalized JSD

    Handle the forward and backward pass of the final linear layer via JSD by avoiding
    the materialization of the large logits tensor. Since JSD is the last layer, we can
    compute the gradient at the forward pass.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
        jsd_beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        """
        Args:

            student_input (torch.tensor): input of the last projection layer in student model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            student_weight (torch.tensor): the last projection layer in student model, with shape (V, H), where V is vocab size
            teacher_input (torch.tensor): input of the last projection layer in teacher model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            teacher_weight (torch.tensor): the last projection layer in teacher model, with shape (V, H), where V is vocab size
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT) where each value is in [0, V-1].
            jsd_beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
            ignore_index (int): the index to ignore. Default: -100
            temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

        Returns:
            loss (torch.Tensor): generalized JSD
        """
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (teacher_input.shape[0],), (
                f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, grad_input, grad_weight = fused_linear_jsd_forward(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            shift_labels,
            jsd_beta,
            ignore_index,
            has_label,
            temperature,
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
        )
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        grad_input, grad_weight = fused_linear_jsd_backward(grad_output, grad_input, grad_weight)
        return (grad_input, grad_weight, None, None, None, None, None, None)


class LigerFusedLinearJSD(torch.nn.Module):
    r"""Fusing the last linear layer with generalized JSD

    Handle the forward and backward pass of the final linear layer via JSD by avoiding
    the materialization of the large logits tensor.

    Args:
        jsd_beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
        ignore_index (int): The index to ignore in the target. Default: `-100`
        temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

    Shape:
        - student_input: :math:`(BT, H)`, where B is batch size, T is sequence length, H is hidden dimension.
        - student_weight: :math:`(V, H)`, where V is vocab size.
        - teacher_input: :math:`(BT, H')`, where H' is hidden dimension of the teacher model.
        - teacher_weight: :math:`(V, H')`, where hidden size H and H' can be different.
        - shift_labels: :math:`(BT,)`
        - Output: a scalar.

    Examples:
    ```python
    >>> (B, T, H_s, H_t, V) = (2, 2, 3, 5, 10)
    >>> fused_jsd = LigerFusedLinearJSD(jsd_beta=0.1, temperature=2.0)
    >>> # generate inputs and weights
    >>> student_input = torch.rand(B * T, H_s, device="cuda", requires_grad=True)
    >>> student_lin = torch.nn.Linear(H_s, V, bias=False, device="cuda")
    >>> # teacher input doesn't require grad, hidden_dim can be different from student's
    >>> teacher_input = torch.rand(B * T, H_t, device="cuda")
    >>> teacher_lin = torch.nn.Linear(H_t, V, bias=False, device="cuda")
    >>> output = fused_jsd(student_input, student_lin.weight, teacher_input, teacher_lin.weight)
    >>> output.backward()
    >>>
    >>> # Example with labels for supervised fine-tuning (SFT) context:
    >>>
    >>> # Assume hidden_states, lm_heads and corresponding labels are given
    >>> student_lm_head = torch.nn.Linear(H_s, V, bias=False)
    >>> student_hidden_states = torch.randn(B * T, H_s, requires_grad=True).log_softmax(dim=-1)
    >>> teacher_lm_head = torch.nn.Linear(H_t, V, bias=False)
    >>> teacher_hidden_states = torch.randn(B * T, H_t).log_softmax(dim=-1)
    >>> labels = torch.randint(0, V, (B * T,), torch.long)
    >>>
    >>> # Shift so that tokens < n predict n
    >>> shift_student_hidden_states = student_hidden_states[..., :-1, :].contiguous()
    >>> shift_teacher_hidden_states = teacher_hidden_states[..., :-1, :].contiguous()
    >>> shift_labels = labels[..., 1:].contiguous()
    >>>
    >>> # Flatten tokens
    >>> shift_student_hidden_states = shift_student_hidden_states.view(-1, V)
    >>> shift_teacher_hidden_states = shift_teacher_hidden_states.view(-1, V)
    >>> shift_labels = shift_labels.view(-1)
    >>>
    >>> # Calculate loss
    >>> loss_fct = LigerJSD(beta=0.1)
    >>> loss = loss_fct(
    >>>     shift_studetn_hidden_states,
    >>>     student_lm_head.weight,
    >>>     shift_teacher_hidden_states,
    >>>     teacher_lm_head.weight,
    >>>     shift_labels
    >>> )
    ```
    """

    def __init__(self, jsd_beta=0.5, ignore_index=-100, temperature=1.0):
        super().__init__()
        assert temperature != 0, "temperature cannot be 0."
        self.jsd_beta = jsd_beta
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        shift_labels: Optional[torch.LongTensor],
    ):
        return LigerFusedLinearJSDFunction.apply(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            shift_labels,
            self.jsd_beta,
            self.ignore_index,
            self.temperature,
        )


class Model(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.fused_jsd = LigerFusedLinearJSD(jsd_beta=beta, ignore_index=ignore_index, temperature=temperature)

    def forward(self, student_input, teacher_input, label=None):
        return self.fused_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            label,
        )