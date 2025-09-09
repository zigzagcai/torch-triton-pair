# adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/transformers/test_jsd.py
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
from typing import Optional

class Model(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ):
        super(Model, self).__init__()
        self.kl = KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.Tensor,  # input student logits
        log_p: torch.Tensor,  # target
        label: Optional[torch.Tensor] = None,
    ):
        if self.beta == 0.0:  # KL(p||q) -> kl(q, p)
            loss = self.kl(log_q, log_p).sum(dim=-1)
        elif self.beta == 1.0:  # KL(q||p) -> kl(p, q)
            loss = self.kl(log_p, log_q).sum(dim=-1)
        else:
            log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
            log_p, log_q = (
                log_p.view(-1, log_p.size(-1)),
                log_q.view(-1, log_q.size(-1)),
            )
            m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
            loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (1 - self.beta) * self.kl(
                torch.log(m), log_q
            ).sum(dim=-1)

        if label is not None:
            loss = torch.where(label != self.ignore_index, loss, 0.0)
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                loss = torch.tensor(0.0).to(loss.device)
            else:
                loss = (loss / n_non_ignore).sum()
        else:
            loss = (loss / log_q.shape[0]).sum()
        return loss.to(self.dtype)

B = 2
T = 10
V = 32
beta = 0.5
ignore_index = -100
device = 'cuda'
dtype = torch.float32

def get_inputs():
    inp = torch.randn(B * T, V, device=device, dtype=dtype, requires_grad=True).log_softmax(dim=-1)
    x1 = inp.detach().clone().requires_grad_(True)
    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).log_softmax(dim=-1)
    label = torch.full((B * T,), ignore_index, device=device, dtype=torch.long)
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    label[indices_to_assign] = ignore_index
    return [x1, target, label]

def get_init_inputs():
    return [beta, ignore_index]