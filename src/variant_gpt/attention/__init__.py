from typing import Type

from .base import CausalSelfAttention
from .config import AttentionConfig
from .mha import MultiHeadAttention
from .mqa import MultiQueryAttention

_REGISTRY: dict[str, Type[CausalSelfAttention]] = {
    "mha": MultiHeadAttention,
    "mqa": MultiQueryAttention,
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
    "MultiHeadAttention",
    "MultiQueryAttention",
    "build_attention",
    "register_attention",
    "available_attentions",
]
