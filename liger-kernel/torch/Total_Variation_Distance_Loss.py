# adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/transformers/test_tvd.py
import torch


class Model(torch.nn.Module):
    def __init__(self, reduction="batchmean", ignore_index: int = -100):
        super(Model, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, p, q, label=None):
        tvd = torch.abs(p - q) / 2.0
        n_non_ignore = p.size(0)
        if label is not None:
            tvd = torch.where(label.unsqueeze(1) != self.ignore_index, tvd, torch.zeros_like(tvd))
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                return torch.tensor(0.0).to(tvd.device)

        if self.reduction == "mean":
            return torch.sum(tvd) / (n_non_ignore * p.size(1))
        elif self.reduction == "sum":
            return torch.sum(tvd)
        elif self.reduction == "none":
            return tvd
        elif self.reduction == "batchmean":
            return torch.sum(tvd) / n_non_ignore
        else:
            raise ValueError("Invalid reduction type.")


B, T, V = 1, 4096, 32000
reduction = 'batchmean'
ignore_index = -100
dtype = torch.float32
device = 'cuda'


def get_inputs():
    input = torch.randn(B * T, V, device=device, dtype=dtype, requires_grad=True)

    x1 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, device=device).softmax(dim=-1)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    label[indices_to_assign] = ignore_index
    return [x1, target, label]


def get_init_inputs():
    return [reduction, ignore_index]