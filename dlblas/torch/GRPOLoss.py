# adapted from https://github.com/DeepLink-org/dlBLAS/blob/main/tests/kernels/test_grpo_loss_logits.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/grpo_loss.py
import torch
import torch.nn as nn

KL = 0
UNBIAS = 1
MSE = 2

def torch_grpo_loss( _logprobs, _old_logprobs, _advantages, _ref_logprobs, kl_type, kl_coef, _loss_factor, clip):
    kl_type = {
        'kl': KL,
        'unbias': UNBIAS,
        'mse': MSE
    }.get(kl_type, None)

    logprobs_diff = _logprobs - _old_logprobs
    ratio = torch.exp(logprobs_diff)
    pg_losses = -_advantages.unsqueeze(1) * ratio
    pg_losses2 = -_advantages.unsqueeze(1) * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    pg_loss_max = torch.max(pg_losses, pg_losses2)
    pg_loss = pg_loss_max.sum()
    _loss = pg_loss * _loss_factor

    # Compute KL penalty loss
    if kl_type == 0:
        kl = _ref_logprobs - _logprobs 
        _kl_penalty_loss = (kl_coef * kl).sum(dim=1) * _loss_factor  
    elif kl_type == 1:
        kl = _ref_logprobs - _logprobs
        nonneg_nobias_kl = torch.exp(kl) - kl - 1
        _kl_penalty_loss = (kl_coef * nonneg_nobias_kl).sum(dim=1) * _loss_factor
    elif kl_type == 2:
        _kl_penalty_loss = (kl_coef * (_ref_logprobs - _logprobs).square() / 2).sum(dim=1) * _loss_factor
    else:
        raise ValueError(f"Unsupported KL type: {kl_type}")
    loss = _loss + _kl_penalty_loss
    return loss


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, log_probs, log_probs1, log_probs2, advantages, kl_type, kl_coef, loss_factor, clip, BLOCK_SIZE_T):
        return torch_grpo_loss(log_probs, log_probs1, advantages, log_probs2, kl_type, kl_coef, loss_factor, clip)

B = 8
T = 32
H = 256
V = 1024
BLOCK_SIZE_T = 8
torch.manual_seed(42)
device_ = 'cuda'
kl_type = 'kl'

advantages = torch.randn((T,), dtype=torch.float32, device=device_, requires_grad=True)
log_probs = torch.randn((T, V), dtype=torch.float32, device=device_, requires_grad=True)
log_probs1 = torch.randn((T, V), dtype=torch.float32, device=device_, requires_grad=True)
log_probs2 = torch.randn((T, V), dtype=torch.float32, device=device_, requires_grad=True)

loss_factor = kl_coef = 1.0
clip = 0.2

def get_inputs():
    return [log_probs, log_probs1, log_probs2, advantages, kl_type, kl_coef, loss_factor, clip, BLOCK_SIZE_T]


def get_init_inputs():
    return []  # No special initialization inputs needed