import torch
import math
import torch.nn as nn

class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        scale: float = None,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def _create_neighborhood_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        half_kernel = self.kernel_size // 2

        for i in range(seq_len):
            start = max(0, i - half_kernel * self.dilation)
            end = min(seq_len, i + half_kernel * self.dilation + 1)

            for j in range(start, end):
                if self.dilation == 1 or (j - i) % self.dilation == 0:
                    mask[i, j] = True

        return mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        mask = self._create_neighborhood_mask(seq_len, hidden_states.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        output = self.out_proj(attn_output)

        return output


batch_size, seq_len, hidden_size, num_heads, kernel_size = (2, 32, 128, 4, 7)
dtype = torch.float32
device = 'cuda'


def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    return [hidden_states]


def get_init_inputs():
    return [hidden_size, num_heads, kernel_size]