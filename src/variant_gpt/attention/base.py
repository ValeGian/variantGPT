import torch
from torch import nn

from .config import AttentionConfig


class CausalSelfAttention(nn.Module):
    """
    Base class for causal self-attention variants.

    Contract:
      - __init__ takes an AttentionConfig.
      - forward(x: (B, T, n_embd)) -> (B, T, n_embd).
      - The output projection is exposed as `self.c_proj` (nn.Linear),
        so the host model can apply GPT-style scaled init by name.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
