from dataclasses import replace

from .config import AttentionConfig
from .gqa import GroupedQueryAttention


class MultiHeadAttention(GroupedQueryAttention):
    """
    Multi-head attention.

    Special case of grouped-query attention where each query head has its own
    key/value head (n_kv_head = n_head). Whatever the user sets for `n_kv_head`
    in the config is overridden here so that `attention_type="mha"` is always
    unambiguously MHA.

    The underlying `GroupedQueryAttention` uses fused `c_attn` (n_embd → 3·n_embd)
    in this regime, matching the parameter layout of the original standalone
    MHA implementation.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(replace(config, n_kv_head=config.n_head))
