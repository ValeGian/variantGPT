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