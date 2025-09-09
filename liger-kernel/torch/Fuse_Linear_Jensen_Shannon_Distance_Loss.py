import torch

from torch.nn import KLDivLoss
from typing import Optional

class TorchJSD(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ):
        super(TorchJSD, self).__init__()
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

class Model(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based jsd loss.

    :param H: hidden size
    :param V: vocab size
    :param temperature: softmax temperature
    :param beta: jsd beta
    """

    def __init__(
        self,
        H,
        V,
        dtype,
        device,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super(Model, self).__init__()
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.jsd = TorchJSD(beta=beta, ignore_index=ignore_index, dtype=dtype)
        self.temperature = temperature

    def forward(self, student_input, teacher_input, label=None):
        student_logits = self.student_lin(student_input).to(torch.float32)
        teacher_logits = self.teacher_lin(teacher_input).to(torch.float32)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)
        return self.jsd(student_prob, teacher_prob, label)


B, T, H, V = (8, 128, 1024, 4096)
beta = 0.5
ignore_index = -100
temperature = 1.0
scalar = 1.0
dtype = torch.float32
device = 'cuda'


def get_inputs():
    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar
    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    label[indices_to_assign] = ignore_index
    return [_input1, teacher_input, label]


def get_init_inputs():
    return [H, V, dtype, device, beta, ignore_index, temperature]