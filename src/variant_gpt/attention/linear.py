import torch
from torch import nn
from torch.nn import functional as F

from .base import CausalSelfAttention
from .config import AttentionConfig


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Feature map φ(x) = elu(x) + 1. Non-negative, smooth, GPU-friendly."""
    return F.elu(x) + 1.0


class LinearAttention(CausalSelfAttention):
    """
    Causal linear attention (Katharopoulos et al., 2020 — "Transformers are RNNs").

    Replaces softmax(QK^T / √d) · V with a kernel feature map φ:

        out_i = (φ(q_i) · S_i) / (φ(q_i) · z_i)

    where  S_i = Σ_{j≤i} φ(k_j) v_j^T   and   z_i = Σ_{j≤i} φ(k_j).

    Complexity is O(T · D²) instead of O(T² · D), which wins at long context.

    Implementation: chunked parallel form. The sequence is split into chunks
    of size C; intra-chunk attention runs as a small C×C causal matmul, and
    inter-chunk contributions come from a cumulative state computed once per
    chunk and broadcast back. Peak intermediate memory is O(B·H·(T/C)·D²)
    rather than O(B·H·T·D²), with no Python loop over chunks.

    Notes:
      * `flash` is ignored — linear attention has its own algorithm; SDPA
        does not apply.
      * Causality is enforced by the cumulative state plus a per-chunk
        causal mask; there is no global T×T mask.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        assert config.chunk_size > 0, "LinearAttention requires chunk_size > 0"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = config.chunk_size

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Per-chunk causal mask, registered once.
        intra_mask = torch.tril(
            torch.ones(self.chunk_size, self.chunk_size, dtype=torch.bool)
        )
        self.register_buffer("intra_mask", intra_mask, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim
        chunk = self.chunk_size

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Feature map on Q and K only; V stays linear.
        q = _phi(q)
        k = _phi(k)

        # Pad T up to a multiple of chunk so we can reshape cleanly.
        # Padding adds zeros → padded q/k contribute nothing to the inter-chunk
        # state (k_pad=0 ⇒ k_pad ⊗ v_pad = 0, k_pad sum = 0), and the intra-chunk
        # causal mask keeps real positions from attending to padded ones.
        pad = (chunk - T % chunk) % chunk
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        Tp = T + pad
        n_chunks = Tp // chunk

        # Reshape to chunks: (B, H, n_chunks, chunk, D)
        q = q.view(B, H, n_chunks, chunk, D)
        k = k.view(B, H, n_chunks, chunk, D)
        v = v.view(B, H, n_chunks, chunk, D)

        # ── Inter-chunk path ────────────────────────────────────────────
        # Per-chunk kv outer product, summed over the chunk.
        # chunk_kv[b,h,c,d_in,d_out] = Σ_{i in chunk c} φ(k)_{i,d_in} · v_{i,d_out}
        chunk_kv = torch.einsum("bhcsd,bhcse->bhcde", k, v)  # (B, H, n_chunks, D, D)
        chunk_k = k.sum(dim=3)                                # (B, H, n_chunks, D)

        # Cumulative state up to and including each chunk.
        cum_kv = chunk_kv.cumsum(dim=2)
        cum_k = chunk_k.cumsum(dim=2)

        # State *before* each chunk = cumulative state shifted right by one.
        # Pad layout for F.pad: (last_dim_l, last_dim_r, 2nd_last_l, 2nd_last_r, ...).
        prev_kv = F.pad(cum_kv[:, :, :-1], (0, 0, 0, 0, 1, 0))  # (B, H, n_chunks, D, D)
        prev_k  = F.pad(cum_k [:, :, :-1], (0, 0, 1, 0))        # (B, H, n_chunks, D)

        inter_num = torch.einsum("bhcsd,bhcde->bhcse", q, prev_kv)  # (B, H, n_chunks, chunk, D)
        inter_den = torch.einsum("bhcsd,bhcd->bhcs",   q, prev_k)   # (B, H, n_chunks, chunk)

        # ── Intra-chunk path ────────────────────────────────────────────
        # Linear attention restricted to each chunk, with a per-chunk causal mask.
        attn = torch.einsum("bhcsd,bhctd->bhcst", q, k)            # (B, H, n_chunks, chunk, chunk)
        attn = attn.masked_fill(~self.intra_mask, 0.0)
        intra_num = torch.einsum("bhcst,bhctd->bhcsd", attn, v)    # (B, H, n_chunks, chunk, D)
        intra_den = attn.sum(dim=-1)                                # (B, H, n_chunks, chunk)

        # ── Combine, normalize, reshape ─────────────────────────────────
        num = inter_num + intra_num
        den = (inter_den + intra_den).unsqueeze(-1).clamp(min=1e-6)
        y = num / den                                              # (B, H, n_chunks, chunk, D)

        y = y.reshape(B, H, Tp, D)
        if pad:
            y = y[:, :, :T]

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
