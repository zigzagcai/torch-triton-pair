import torch
import torch.nn as nn


def unpack_weights(packed: torch.Tensor, bits: int = 2) -> torch.Tensor:
    values_per_item = 8 // bits
    packed_shape = packed.shape

    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * values_per_item
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * values_per_item
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)

    for i in range(values_per_item):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        unpacked[start:end] = (packed & mask) >> (2 * i)

    unpacked = unpacked.to(torch.int32) - 1
    return unpacked


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, ht, u):
        # Unpack weights and compute torch output
        unpacked = unpack_weights(u.T, bits=2).T
        torch_output = torch.matmul(ht.to(torch.float32), unpacked.T.contiguous().to(torch.float32))
        return torch_output.to(torch.int32)


batch_size = 2
seq_len = 2048
out_features = 4096
block_size = 128
size = 2048
device = 'cuda'

def get_inputs():
    # Generate the random tensors
    ht = torch.randint(-127, 127, (batch_size, seq_len, size * 4), device=device, dtype=torch.int8)
    u = torch.randint(0, 255, (out_features, size), device=device, dtype=torch.uint8)
    return [ht, u]

def get_init_inputs():
    return []