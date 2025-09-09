import torch
import torch
import torch.nn.functional as F

from typing import Optional

class CrossEntropyWithZLoss(torch.nn.Module):
    def __init__(
        self,
        weight=None,
        lse_square_scale=0.0,
        reduction="mean",
        ignore_index=-100,
        label_smoothing=0.0,
        return_z_loss=False,
        dtype=torch.float32,
        accum_dtype=None,
    ):
        super().__init__()
        self.weight = weight
        self.lse_square_scale = lse_square_scale
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.return_z_loss = return_z_loss
        self.label_smoothing = label_smoothing
        self.dtype = dtype

    def forward(self, logits, targets):
        # Loss calculations are all in float32
        logits = logits.to(torch.float32)

        target_mask = targets != self.ignore_index

        # Standard cross entropy loss
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

        # Compute log-sum-exp term
        lse = torch.logsumexp(logits, dim=-1)

        # Z-loss term
        z_loss = torch.where(targets != self.ignore_index, self.lse_square_scale * (lse**2), 0.0)

        if self.reduction == "mean":
            z_loss = z_loss.sum() / target_mask.sum()
        elif self.reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss
        ce_loss = ce_loss.to(self.dtype)
        z_loss = z_loss.to(self.dtype)

        # Final loss: cross-entropy loss + Z-loss
        total_loss = ce_loss + z_loss
        if self.return_z_loss:
            return total_loss, z_loss
        else:
            return total_loss


class Model(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    :param label_smoothing: label_smoothing to apply on target
    :param lse_square_scale: scaler of lse ^ 2 to compute z loss

    # TODO: if we bump CI env's `transformers` version to >= 4.46, we should just directly
    # call https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L32
    # to be consistent with Hugging Face model implementation.
    """

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
        accum_dtype = None,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ce_loss = CrossEntropyWithZLoss(
            weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            return_z_loss=return_z_loss,
        )
        self.softcap = softcap

    def forward(self, x, y):
        logits = self.lin(x).to(torch.float32)
        if self.softcap is not None and self.softcap != 0.0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return self.ce_loss(logits, y)


B, T, H, V = (8, 128, 1024, 4096)
reduction, scalar, dtype, atol, rtol = ("mean", 1.0, torch.bfloat16, 5e-3, 5e-2)
bias = True
has_ce_weight, label_smoothing, ignore_index, lse_square_scale, softcap, return_z_loss, accum_dtype = (False, 0, -100, 0, None, False, None)
device = 'cuda'


def get_inputs():
    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    target[indices_to_assign] = ignore_index
    return (_tensor, target)


def get_init_inputs():
    if has_ce_weight:
        ce_weight = torch.rand(V, device=device, dtype=torch.float32)
    else:
        ce_weight = None
    return (H, V, dtype, bias, ce_weight, ignore_index, lse_square_scale, label_smoothing, reduction, softcap, return_z_loss, accum_dtype)