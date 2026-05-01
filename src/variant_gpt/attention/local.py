import math
import torch
from torch import nn
from torch.nn import functional as F

from .base import CausalSelfAttention
from .config import AttentionConfig


class LocalAttention(CausalSelfAttention):
    """
    Causal self-attention with a sliding window.

    Each query position `i` attends only to key positions `j` such that
        max(0, i - window_size + 1) <= j <= i,
    i.e. at most `window_size` tokens (including itself).

    `window_size == block_size` is equivalent to full causal attention
    (up to the extra mask materialization cost), and `window_size == 1`
    is pure diagonal (each token only sees itself).

    Shape/API contract is identical to MultiHeadAttention.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        assert config.window_size is not None and config.window_size > 0, (
            "LocalAttention requires config.window_size to be a positive integer"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.window_size = config.window_size
        self.flash = config.flash

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        if not self.flash:
            self.attn_dropout = nn.Dropout(config.dropout)

        # Precompute the sliding-window causal mask once.
        # Broadcasting i (column vector) against j (row vector) gives a (T, T)
        # grid where mask[i, j] tells us whether query i may attend to key j.
        #   mask[i, j] = True  -> allowed to attend
        #   mask[i, j] = False -> masked out
        T = config.block_size
        i = torch.arange(T).view(T, 1)
        j = torch.arange(T).view(1, T)
        allowed = (j <= i) & (j > i - self.window_size)
        self.register_buffer(
            "window_mask",
            allowed.view(1, 1, T, T),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)              # each (B, T, n_embd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)    # (B, n_head, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)    # (B, n_head, T, D)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)    # (B, n_head, T, D)

        mask = self.window_mask[:, :, :T, :T]                            # (1, 1, T, T)
        if self.flash:
            # Boolean attn_mask already encodes causality, so is_causal=False.
            # (SDPA disallows combining is_causal=True with an explicit mask.)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Re-merge heads: (B, n_head, T, D) → (B, T, n_embd).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
