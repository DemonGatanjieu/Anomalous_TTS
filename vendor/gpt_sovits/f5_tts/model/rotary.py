"""Rotary position embedding from x-transformers (MIT, Phil Wang), written out by
tools/sync_upstream.py so that package is not needed. Only what the F5-TTS DiT calls."""
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))

    def forward_from_seq_len(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.stack((freqs, freqs), dim=-1).flatten(-2).unsqueeze(0), 1.0


def rotate_half(x):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim=-1).type(orig_dtype)
