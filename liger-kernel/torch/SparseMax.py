# adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/transformers/test_softmax.py
import torch
import torch.nn as nn

def torch_sparsemax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    input_dims = input_tensor.dim()
    if dim < 0:
        dim = input_dims + dim
    input_sorted, _ = torch.sort(input_tensor, dim=dim, descending=True)
    cumsum_input = torch.cumsum(input_sorted, dim=dim)
    input_size = input_tensor.size(dim)
    range_tensor = torch.arange(1, input_size + 1, device=input_tensor.device, dtype=input_tensor.dtype)
    shape = [1] * input_dims
    shape[dim] = input_size
    range_tensor = range_tensor.view(shape)
    k_bound = 1 + range_tensor * input_sorted
    support = k_bound > cumsum_input
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    support_sum = (input_sorted * support).sum(dim=dim, keepdim=True)
    tau = (support_sum - 1) / k
    return torch.clamp(input_tensor - tau, min=0)

class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim
    
    def forward(self, inp):
        return torch_sparsemax(inp, self.dim)

batch_size = 2
seq_len = 128
features = 512
dim = -1
dtype = torch.float32
device = 'cuda'

def get_inputs():
    shape = (batch_size, seq_len, features)
    x = torch.randn(*shape, dtype=dtype, device=device)
    return (x,)

def get_init_inputs():
    return (dim,)  # No special initialization inputs needed