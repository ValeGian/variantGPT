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

    n_kv_head: Optional[int] = None    # Used by GQA-family variants. None => MHA, 1 => MQA, 1<k<n_head => GQA.
    window_size: Optional[int] = None  # Used by local (sliding-window) attention. None => full attention.
    chunk_size: int = 64               # Used by linear attention for the chunked parallel algorithm.
