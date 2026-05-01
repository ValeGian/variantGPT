from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionConfig:
    n_embd: int
    n_head: int
    block_size: int
    dropout: float = 0.0
    bias: bool = False
    flash: bool = True

    n_kv_head: Optional[int] = None        # Used by GQA-family variants. None => MHA, 1 => MQA, 1<k<n_head => GQA.
    window_size: Optional[int] = None      # Used by "local" and "sparse" variants. None => full attention (where applicable).
    chunk_size: Optional[int] = None       # Used by "linear" variant for the chunked parallel algorithm.
    n_global_tokens: Optional[int] = None  # Used by "sparse" (BigBird) variant — number of leading global tokens.
    n_random_tokens: Optional[int] = None  # Used by "sparse" (BigBird) variant — random connections per query.

    # ── MLA (DeepSeek-V2) ─────────────────────────────────────────────────
    # Used only by the "mla" variant. See attention/mla.py for the full picture.
    q_lora_rank: Optional[int] = None      # Q-side compression rank. None => single full-rank Q projection.
    kv_lora_rank: Optional[int] = None     # KV latent dim (the thing the KV cache compresses to). REQUIRED for MLA.
    qk_nope_head_dim: Optional[int] = None # Per-head NoPE dim on Q/K. None => head_dim (n_embd // n_head).
    qk_rope_head_dim: Optional[int] = None # Per-head RoPE dim on Q/K (must be even). REQUIRED for MLA.
    v_head_dim: Optional[int] = None       # Per-head V dim. None => head_dim.
    rope_base: Optional[float] = None      # RoPE base θ. None => 10000.
