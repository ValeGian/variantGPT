import math

import torch
from torch import nn
from torch.nn import functional as F

from .configuration import GPT2Config


class GPT2MQAttention(nn.Module):
    def __init__(self, config: GPT2Config, flash=True):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head

        # Query projection for all heads, but single K/V head (MQA)
        self.q_attn = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.kv_attn = nn.Linear(self.n_embd, 2 * self.head_dim, bias=config.bias)  # single K and V head
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = flash
        if not self.flash:
            self.attn_dropout = nn.Dropout(config.dropout)
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Queries: (B, nh, T, hs)
        q = self.q_attn(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Single K and V head: (B, 1, T, hs)
        k, v = self.kv_attn(x).split(self.head_dim, dim=2)
        k = k.view(B, T, 1, self.head_dim).transpose(1, 2)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2)

        # Expand K and V across all query heads: (B, nh, T, hs)
        # expand() avoids materializing copies — memory stays shared until a write occurs
        k = k.expand(B, self.n_head, T, self.head_dim)
        v = v.expand(B, self.n_head, T, self.head_dim)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y