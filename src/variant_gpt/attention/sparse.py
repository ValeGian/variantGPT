import math
import torch
from torch import nn
from torch.nn import functional as F

from .base import CausalSelfAttention
from .config import AttentionConfig


class SparseAttention(CausalSelfAttention):
    """
    BigBird-style causal sparse attention (Zaheer et al., 2020 — "Big Bird:
    Transformers for Longer Sequences").

    Combines three complementary sparsity patterns into a single attention
    mask, all restricted to the causal lower-triangle:

      1. **Global** — the first ``n_global_tokens`` positions are "global":
         every query attends to them (subject to causality), and they
         themselves attend to every past position. They act as
         information hubs that compress and broadcast context across the
         whole sequence.

      2. **Window** — each query attends to the ``window_size`` immediately
         preceding tokens (including itself). Captures short-range
         dependencies, exactly the same pattern as ``LocalAttention``.

      3. **Random** — each query also attends to ``n_random_tokens``
         additional random past positions. The theoretical insight from
         the BigBird paper is that random edges turn the attention graph
         into an expander, so information can still propagate across the
         whole sequence in a constant number of layers despite the
         sparsity. (This is what gives BigBird its universal-approximator
         property, matching full attention.)

    The number of nonzero entries per query is
    ``n_global + window_size + n_random`` (with overlap), so total work
    drops from O(T²) to O(T · k) for k ≪ T.

    Implementation note
    -------------------
    For didactic clarity we materialize the (T, T) boolean mask once at
    init and pass it to PyTorch's SDPA. This is **not** a real
    block-sparse kernel — the flash path still allocates the full T×T
    attention matrix internally. The mask just zeroes out most of it.
    A production block-sparse implementation (e.g. the one in the
    original BigBird repo, or recent Triton flash-sparse kernels) would
    pack the active blocks into a dense ``(T, n_global + window + n_random)``
    tensor and only do FLOPs on those; that machinery is deliberately out
    of scope here. As such, this implementation will not give you
    BigBird's *memory* savings, but it does illustrate the pattern
    faithfully and trains correctly.

    The random pattern is sampled once at construction time using the
    active PyTorch RNG, so calling ``torch.manual_seed`` before model
    creation makes the masks reproducible. Different layers get
    different random patterns because they are constructed sequentially
    and consume RNG state in turn — desirable, since it gives the model
    diverse random connections across depth.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        assert config.window_size is not None and config.window_size > 0, (
            "SparseAttention requires window_size > 0"
        )
        assert config.n_global_tokens is not None and config.n_global_tokens >= 0, (
            "SparseAttention requires n_global_tokens >= 0"
        )
        assert config.n_random_tokens is not None and config.n_random_tokens >= 0, (
            "SparseAttention requires n_random_tokens >= 0"
        )
        assert config.n_global_tokens <= config.block_size, (
            f"n_global_tokens ({config.n_global_tokens}) cannot exceed "
            f"block_size ({config.block_size})"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.window_size = config.window_size
        self.n_global = config.n_global_tokens
        self.n_random = config.n_random_tokens
        self.flash = config.flash

        # Same fused QKV projection layout as MHA / Local — keeps checkpoints
        # interchangeable across these dense-projection variants.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        if not self.flash:
            self.attn_dropout = nn.Dropout(config.dropout)

        # Precompute the full (block_size, block_size) mask once. We slice it
        # at forward time to whatever T the batch actually uses.
        self.register_buffer(
            "sparse_mask",
            self._build_mask(config.block_size).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def _build_mask(self, T: int) -> torch.Tensor:
        """
        Build the (T, T) boolean attention mask combining global / window /
        random patterns under a causal constraint.

        ``mask[i, j] == True`` iff query position ``i`` is allowed to attend
        to key position ``j``.
        """
        # Index grids for vectorized predicates.
        i_idx = torch.arange(T).view(T, 1)  # (T, 1) — query index
        j_idx = torch.arange(T).view(1, T)  # (1, T) — key index

        # Causal mask — strictly enforced; nothing below combines this away.
        causal = j_idx <= i_idx                                          # (T, T)

        # Global rows / columns:
        #   - column is global  (j < n_global) → attended by every query
        #   - row    is global  (i < n_global) → attends to every past key
        global_part = (j_idx < self.n_global) | (i_idx < self.n_global)  # (T, T)

        # Sliding window of size ``window_size`` ending at i (inclusive).
        # i.e. j ∈ (i - window_size, i].
        window_part = (j_idx > i_idx - self.window_size) & (j_idx <= i_idx)  # (T, T)

        allowed = causal & (global_part | window_part)                   # (T, T)

        # Random connections. For each query i, sample ``n_random`` random
        # positions in [0, i) and mark them as allowed. Overlap with the
        # global/window patterns is fine — duplicates are no-ops on a
        # boolean OR. We only sample from strict past (j < i), since
        # self-attention is already covered by the window.
        if self.n_random > 0:
            for i in range(1, T):
                num = min(self.n_random, i)
                if num == 0:
                    continue
                # randperm(i) gives a permutation of [0, i); take the first ``num``.
                random_keys = torch.randperm(i)[:num]
                allowed[i, random_keys] = True

        return allowed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        B, T, C = x.size()

        # Fused projection then split into Q, K, V. Each is (B, T, n_embd).
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Split embedding dim into heads, then move heads to dim 1 for batched matmul.
        # (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Slice the precomputed mask down to the current sequence length.
        mask = self.sparse_mask[:, :, :T, :T]  # (1, 1, T, T) — broadcasts over B and n_head

        if self.flash:
            # Boolean attn_mask already encodes causality + sparsity, so
            # ``is_causal=False`` (SDPA forbids combining is_causal with an
            # explicit attn_mask).
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale          # (B, n_head, T, T)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v                                       # (B, n_head, T, head_dim)

        # Re-merge heads: (B, n_head, T, head_dim) → (B, T, n_embd).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))