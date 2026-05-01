import math
import torch
from torch import nn
from torch.nn import functional as F

from .base import CausalSelfAttention
from .config import AttentionConfig


class _RMSNorm(nn.Module):
    """
    Root-mean-square LayerNorm. DeepSeek-V2 normalizes the compressed Q and KV
    latents with RMSNorm; we reproduce that here rather than substituting
    LayerNorm so the module is faithful to the paper. Cheap enough that it
    isn't worth gating behind a config flag.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm in float32 for numerical stability under bf16/fp16
        # autocast, then cast back to the input dtype before scaling.
        x_f32 = x.float()
        norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


def _precompute_rope_cache(seq_len: int, dim: int, base: float):
    """
    Precompute the (cos, sin) tables of shape (seq_len, dim) used by the
    rotate-half RoPE formulation. Stored in float32 and cast at apply time.
    """
    assert dim % 2 == 0, "RoPE requires an even head dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # (seq_len, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in halves and rotate them: [a, b] → [-b, a]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to ``x`` using broadcastable ``cos`` and ``sin`` tables.
    Shapes: x is (..., T, D); cos/sin must broadcast against x.
    """
    return (x * cos) + (_rotate_half(x) * sin)


class MultiHeadLatentAttention(CausalSelfAttention):
    """
    Multi-head Latent Attention (MLA), introduced in DeepSeek-V2
    (Liu et al., 2024 — https://arxiv.org/abs/2405.04434).

    Idea
    ----
    Rather than caching full K and V tensors per head, project the input into
    a low-rank latent ``c_KV ∈ R^{kv_lora_rank}`` (with kv_lora_rank ≪
    n_head·head_dim) and reconstruct K and V from that latent at attention
    time. At inference this collapses the KV cache size from
        2 · n_head · head_dim   →   kv_lora_rank   per token,
    a 4–10× reduction in practice, without the quality regression of MQA/GQA.

    Decoupled RoPE
    --------------
    Combining low-rank KV compression with RoPE naively breaks the
    inference-time matrix-absorption trick that makes MLA cheap, because
    RoPE's position-dependent rotation does not commute with the KV
    up-projection. DeepSeek's fix is to split each Q and K head into two
    parts:

      * a *content* (NoPE) part of dim ``qk_nope_head_dim`` that flows through
        the low-rank compression — this is what gets absorbed at inference;
      * a *position* (RoPE) part of dim ``qk_rope_head_dim`` that is computed
        from the input directly and carries RoPE. A single RoPE-K vector is
        shared across all heads (MQA-style), so the extra KV-cache overhead
        from the position part is just ``qk_rope_head_dim`` per token, not
        ``n_head · qk_rope_head_dim``.

    The two parts are concatenated along the head dim before SDPA, so the
    effective QK head dim is ``qk_nope_head_dim + qk_rope_head_dim`` and the
    V head dim is ``v_head_dim`` (which need not match qk_nope_head_dim,
    though by default it does).

    Optional Q compression
    ----------------------
    If ``q_lora_rank`` is set, the query path also goes through a low-rank
    bottleneck. This doesn't help inference latency (queries are recomputed
    every step), but it cuts training-time parameter count meaningfully on
    large models. Set to ``None`` to use a single full-rank Q projection.

    Note on positional encoding
    ---------------------------
    MLA assumes RoPE for positional information. The host ``GPT2Model`` in
    this codebase adds learned ``wpe`` to the input embeddings; when using
    MLA you should either drop ``wpe`` (so RoPE is the sole positional
    scheme, matching the paper) or accept the redundant double encoding.
    Both work — the module is correct either way.

    Implementation notes
    --------------------
    * This is a training-oriented implementation: it does not exploit the
      inference-time KV-cache compression or the matrix-absorption trick.
      Adding those would mean introducing a KV cache abstraction, which is
      out of scope for the current pretraining harness.
    * The output projection is named ``c_proj`` so the host model's
      GPT-2-style scaled init (``std = 0.02 / √(2·n_layer)``) applies to it
      automatically via the ``c_proj.weight`` name match.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        head_dim = config.n_embd // config.n_head

        # Resolve dims with sensible defaults: the NoPE and V head dims
        # default to the model's natural head_dim; the RoPE head dim and
        # latent ranks have no good default and must be set explicitly.
        qk_nope = config.qk_nope_head_dim if config.qk_nope_head_dim is not None else head_dim
        qk_rope = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim if config.v_head_dim is not None else head_dim
        kv_lora_rank = config.kv_lora_rank
        q_lora_rank = config.q_lora_rank

        assert qk_rope is not None and qk_rope > 0 and qk_rope % 2 == 0, (
            "MLA requires qk_rope_head_dim to be a positive even integer "
            "(RoPE rotates pairs of channels)"
        )
        assert kv_lora_rank is not None and kv_lora_rank > 0, (
            "MLA requires kv_lora_rank > 0 — set it on AttentionConfig / GPT2Config"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = config.flash
        self.qk_nope_head_dim = qk_nope
        self.qk_rope_head_dim = qk_rope
        self.qk_head_dim = qk_nope + qk_rope        # what SDPA actually sees on QK
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank

        # ── Q path ────────────────────────────────────────────────────────
        if q_lora_rank is None:
            # Single full-rank projection: x → (n_head · qk_head_dim).
            self.q_proj = nn.Linear(
                config.n_embd, config.n_head * self.qk_head_dim, bias=config.bias
            )
        else:
            # Low-rank: x → c_q (q_lora_rank) → q. Matches DeepSeek-V2 layout:
            #   q_a_proj      (down)
            #   q_a_layernorm (RMSNorm on the latent)
            #   q_b_proj      (up to per-head Q)
            self.q_a_proj = nn.Linear(config.n_embd, q_lora_rank, bias=config.bias)
            self.q_a_layernorm = _RMSNorm(q_lora_rank)
            self.q_b_proj = nn.Linear(
                q_lora_rank, config.n_head * self.qk_head_dim, bias=config.bias
            )

        # ── KV path ───────────────────────────────────────────────────────
        # The down-projection emits BOTH the compressed KV latent and the
        # head-shared RoPE-K component in one matmul, then we split. The
        # name ``kv_a_proj_with_mqa`` mirrors the HF DeepSeek-V2 reference.
        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embd, kv_lora_rank + qk_rope, bias=config.bias
        )
        self.kv_a_layernorm = _RMSNorm(kv_lora_rank)
        # Up-projection from latent to per-head NoPE-K and V (concatenated).
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, config.n_head * (qk_nope + v_head_dim), bias=config.bias
        )

        # ── Output projection ─────────────────────────────────────────────
        # V output is (n_head · v_head_dim); project back to n_embd.
        # Named c_proj so GPT2Model.__init__ applies the scaled residual init.
        self.c_proj = nn.Linear(
            config.n_head * v_head_dim, config.n_embd, bias=config.bias
        )
        self.resid_dropout = nn.Dropout(config.dropout)

        if not self.flash:
            self.attn_dropout = nn.Dropout(config.dropout)
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )

        # ── RoPE tables ───────────────────────────────────────────────────
        # Precompute cos/sin once for the RoPE head dim, up to block_size.
        # Kept in float32 for precision; cast at apply time to match input.
        rope_base = config.rope_base if config.rope_base is not None else 10000.0
        cos, sin = _precompute_rope_cache(config.block_size, qk_rope, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        H = self.n_head

        # ── Q ─────────────────────────────────────────────────────────────
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, T, H, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ── KV down + split ───────────────────────────────────────────────
        kv_down = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_rope = kv_down.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # k_rope is shared across heads (MQA-style); add a singleton head dim.
        k_rope = k_rope.unsqueeze(2)  # (B, T, 1, qk_rope)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(B, T, H, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ── RoPE on the position components only ──────────────────────────
        # cos/sin shape (T, D) → (1, T, 1, D) so they broadcast against
        # (B, T, H, D) without any per-head expansion.
        cos = self.rope_cos[:T].to(q_rope.dtype).view(1, T, 1, self.qk_rope_head_dim)
        sin = self.rope_sin[:T].to(q_rope.dtype).view(1, T, 1, self.qk_rope_head_dim)
        q_rope = _apply_rope(q_rope, cos, sin)
        k_rope = _apply_rope(k_rope, cos, sin)

        # ── Concatenate NoPE + RoPE on the head_dim axis ──────────────────
        q = torch.cat([q_nope, q_rope], dim=-1)                 # (B, T, H, qk_head_dim)
        # Broadcast the head-shared k_rope to all heads via expand (zero-copy
        # view); torch.cat then materializes a contiguous full K.
        k_rope_full = k_rope.expand(B, T, H, self.qk_rope_head_dim)
        k = torch.cat([k_nope, k_rope_full], dim=-1)            # (B, T, H, qk_head_dim)

        # ── Move heads to dim 1 for SDPA ──────────────────────────────────
        q = q.transpose(1, 2)   # (B, H, T, qk_head_dim)
        k = k.transpose(1, 2)   # (B, H, T, qk_head_dim)
        v = v.transpose(1, 2)   # (B, H, T, v_head_dim)

        # ── Attention ─────────────────────────────────────────────────────
        # Note: SDPA scales by 1/√d_k where d_k = qk_head_dim, which is what
        # we want — q_rope and q_nope share the same Q tensor, so a single
        # scale applies to the full dot-product as in the paper.
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.qk_head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        # y: (B, H, T, v_head_dim)

        # ── Merge heads and project back to n_embd ────────────────────────
        y = y.transpose(1, 2).contiguous().view(B, T, H * self.v_head_dim)
        return self.resid_dropout(self.c_proj(y))
