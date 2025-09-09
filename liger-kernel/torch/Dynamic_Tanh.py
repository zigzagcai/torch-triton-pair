# adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/transformers/test_dyt.py
import torch

# @torch.compile
def torch_dyt_with_beta(x, alpha, gamma, beta):
    return gamma * torch.tanh(x * alpha) + beta


# @torch.compile
def torch_dyt_without_beta(x, alpha, gamma):
    return gamma * torch.tanh(x * alpha)


class Model(torch.nn.Module):
    def __init__(self, hidden_size, beta=True, init_alpha=0.5):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
        self.beta = None
        if beta:
            self.beta = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        if self.beta is None:
            return torch_dyt_without_beta(x, self.alpha, self.gamma)
        return torch_dyt_with_beta(x, self.alpha, self.gamma, self.beta)



B, T, hidden_size = (2, 8, 4096)
beta = True
init_alpha = 0.5
dtype = torch.float32
device = 'cuda'


def get_inputs():
    _input = torch.randn(B, T, hidden_size, device=device, dtype=dtype)
    return (_input,)


def get_init_inputs():
    return (hidden_size, beta, init_alpha)