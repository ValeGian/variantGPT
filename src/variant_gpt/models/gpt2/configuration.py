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
    n_kv_head: Optional[int] = None  # only read by GQA-family variants
    window_size: Optional[int] = None  # only read by local (sliding-window) variant
    chunk_size: Optional[int] = None  # only by linear variant
