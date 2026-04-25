import math
import torch
from torch import nn
from torch.nn import functional as F

from .base import CausalSelfAttention
from .config import AttentionConfig


class GroupedQueryAttention(CausalSelfAttention):
    """
    Grouped-query attention.

    A single implementation that subsumes:
      * MHA  when  n_kv_head == n_head  (or None — interpreted as n_head)
      * MQA  when  n_kv_head == 1
      * GQA  when  1 < n_kv_head < n_head  (must divide n_head)

    Each group of `n_head / n_kv_head` query heads shares one key/value head.

    Parameter naming preserves backward compatibility with the standalone
    MHA/MQA classes, so existing checkpoints load cleanly:
      * MHA mode:   fused `c_attn`  (n_embd → 3·n_embd)              ←  matches old MHA
      * MQA / GQA:  separate `q_attn` and `kv_attn`                  ←  matches old MQA
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0

        # None ⇒ MHA: one KV head per query head.
        n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        assert 1 <= n_kv_head <= config.n_head, (
            f"n_kv_head ({n_kv_head}) must satisfy 1 <= n_kv_head <= n_head ({config.n_head})"
        )
        assert config.n_head % n_kv_head == 0, (
            f"n_head ({config.n_head}) must be divisible by n_kv_head ({n_kv_head}) "
            f"so query heads split into equal-sized groups"
        )

        self.n_head = config.n_head
        self.n_kv_head = n_kv_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.flash = config.flash
        self.kv_dim = n_kv_head * self.head_dim
        self.is_mha = n_kv_head == config.n_head

        if self.is_mha:
            # Fused QKV projection — matches the old standalone MHA.
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        else:
            # Separate Q and KV projections — matches the old standalone MQA, generalized.
            self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.kv_attn = nn.Linear(config.n_embd, 2 * self.kv_dim, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        if not self.flash:
            self.attn_dropout = nn.Dropout(config.dropout)
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()

        if self.is_mha:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        else:
            q = self.q_attn(x)
            k, v = self.kv_attn(x).split(self.kv_dim, dim=2)

        q = q.view(B, T, self.n_head,    self.head_dim).transpose(1, 2)  # (B, n_head,    T, D)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, D)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.flash:
            # PyTorch ≥ 2.5: SDPA broadcasts KV over query groups internally,
            # avoiding any expand/copy of K and V.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
                enable_gqa=not self.is_mha,
            )
        else:
            # Manual path: broadcast K/V over query groups via repeat_interleave.
            if not self.is_mha:
                group_size = self.n_head // self.n_kv_head
                k = k.repeat_interleave(group_size, dim=1)
                v = v.repeat_interleave(group_size, dim=1)
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
