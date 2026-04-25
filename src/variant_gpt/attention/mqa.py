from dataclasses import replace

from .config import AttentionConfig
from .gqa import GroupedQueryAttention


class MultiQueryAttention(GroupedQueryAttention):
    """
    Multi-query attention.

    Special case of grouped-query attention where all query heads share a
    single key/value head (n_kv_head = 1). Whatever the user sets for
    `n_kv_head` in the config is overridden here so that `attention_type="mqa"`
    is always unambiguously MQA.

    The underlying `GroupedQueryAttention` uses separate `q_attn` and `kv_attn`
    projections in this regime, matching the parameter layout of the original
    standalone MQA implementation.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(replace(config, n_kv_head=1))
