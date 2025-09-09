# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/tests/kernels/test_fill_kv_cache.py
# adapted from https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/kernels/fill_kv_cache.py
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        k_states,
        v_states,
        k_caches,
        v_caches,
        seq_lens,
        history_lens,
        block_offsets,
        block_size,
    ):
        batch_size = len(seq_lens)
        k_caches = k_caches.clone()
        v_caches = v_caches.clone()
        splited_k_states = k_states.split(seq_lens)
        splited_v_states = v_states.split(seq_lens)
        for bidx in range(batch_size):
            k_state = splited_k_states[bidx]
            v_state = splited_v_states[bidx]
            h_len = history_lens[bidx]
            b_offs = block_offsets[bidx]
            block_id = _div_up(h_len + 1, block_size) - 1
            fill_start = h_len % block_size
            fill_size = min(block_size - fill_start, k_state.size(0))
            while True:
                boff = b_offs[block_id]
                tmp_ks = k_state[:fill_size]
                tmp_vs = v_state[:fill_size]
                fill_end = fill_start + fill_size
                k_caches[boff, fill_start:fill_end] = tmp_ks
                v_caches[boff, fill_start:fill_end] = tmp_vs
                k_state = k_state[fill_size:]
                v_state = v_state[fill_size:]
                block_id += 1
                fill_start = 0
                fill_size = min(block_size, k_state.size(0))
                if fill_size == 0:
                    break

        return k_caches, v_caches


num_heads = 8
head_dim = 128
block_size = 64
seq_lens = [1,1,1,1]
history_lens = [1,16,31,24]
device = 'cuda'


def _div_up(a, b):
    return (a + b - 1) // b

def _block_offsets(num_blocks_per_input):
    batch_size = len(num_blocks_per_input)
    max_num_blocks = max(num_blocks_per_input)
    batch_ids = torch.arange(batch_size)
    ret = torch.arange(max_num_blocks)
    ret = batch_ids[:, None] + ret[None, :] * batch_size
    return ret


def get_inputs():
    batch_size = len(seq_lens)
    kv_lens = [s + h for s, h, in zip(seq_lens, history_lens)]
    # max_q_seq_length = max(seq_lens)
    num_tokens = sum(seq_lens)
    num_blocks_per_input = [_div_up(kv_len, block_size) for kv_len in kv_lens]
    max_num_blocks = max(num_blocks_per_input)
    # q_seq_length = torch.tensor(seq_lens).to(device)
    # q_start_loc = q_seq_length.cumsum(0) - q_seq_length
    # kv_seq_length = torch.tensor(kv_lens).to(device)
    k_states = torch.rand(num_tokens, num_heads, head_dim).to(device)
    v_states = torch.rand_like(k_states)
    k_caches = torch.full((batch_size * max_num_blocks, block_size, num_heads, head_dim), 0.0).to(device)
    v_caches = torch.rand_like(k_caches)
    block_offsets = _block_offsets(num_blocks_per_input).to(device)

    return [
        k_states,
        v_states,
        k_caches,
        v_caches,
        seq_lens,
        history_lens,
        block_offsets,
        block_size,
    ]

def get_init_inputs():
    return []  # No special initialization inputs needed