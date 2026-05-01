from dataclasses import dataclass
from typing import Optional


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    flash: bool = False  # use flash attention
    device: str = "cuda"  # device to run on

    attention_type: str = "mha"
    n_kv_head: Optional[int] = None        # only used by GQA-family variants ("gqa", "mqa")
    window_size: Optional[int] = None      # only used by "local" and "sparse" variants
    chunk_size: Optional[int] = None       # only used by "linear" variant
    n_global_tokens: Optional[int] = None  # only used by "sparse" (BigBird) variant
    n_random_tokens: Optional[int] = None  # only used by "sparse" (BigBird) variant

    # ── MLA (DeepSeek-V2) ─────────────────────────────────────────────────
    # Only used by attention_type="mla". Defaults are deliberately None so an
    # accidentally-built MLA without explicit dims fails fast with a clear
    # message instead of silently picking arbitrary numbers.
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    rope_base: Optional[float] = None
