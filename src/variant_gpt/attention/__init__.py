from typing import Type

from .base import CausalSelfAttention
from .config import AttentionConfig
from .gqa import GroupedQueryAttention
from .mha import MultiHeadAttention
from .mqa import MultiQueryAttention
from .local import LocalAttention
from .linear import LinearAttention
from .sparse import SparseAttention
from .mla import MultiHeadLatentAttention

_REGISTRY: dict[str, Type[CausalSelfAttention]] = {
    "mha": MultiHeadAttention,        # GQA with n_kv_head = n_head
    "mqa": MultiQueryAttention,       # GQA with n_kv_head = 1
    "gqa": GroupedQueryAttention,     # n_kv_head taken from config (1 ≤ k ≤ n_head)
    "local": LocalAttention,          # sliding-window causal attention
    "linear": LinearAttention,        # kernel-feature-map attention, O(T) compute
    "sparse": SparseAttention,        # BigBird-style: global + window + random
    "mla": MultiHeadLatentAttention,  # DeepSeek-V2 low-rank KV + decoupled RoPE
}


def register_attention(name: str):
    """Decorator to register a new attention variant under `name`."""
    def _decorator(cls: Type[CausalSelfAttention]):
        if name in _REGISTRY:
            raise ValueError(f"Attention '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return _decorator


def build_attention(name: str, config: AttentionConfig) -> CausalSelfAttention:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown attention type '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](config)


def available_attentions() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "AttentionConfig",
    "CausalSelfAttention",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    "LocalAttention",
    "LinearAttention",
    "SparseAttention",
    "MultiHeadLatentAttention",
    "build_attention",
    "register_attention",
    "available_attentions",
]
