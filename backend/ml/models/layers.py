"""LayerNorm, feed-forward block and the pre-LN transformer block.

Pre-LN (normalise before each sublayer, add residual after) is used because it
trains stably without learning-rate warmup tricks; post-LN (the original 2017
arrangement) is more sensitive at small batch sizes.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """Layer normalisation implemented directly (mean/variance over last dim).

    Implemented here rather than imported so the normalisation semantics -
    unbiased=False, epsilon inside the square root, per-token scaling - are
    explicit and testable.
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        out = normalized * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class FeedForward(nn.Module):
    """Position-wise MLP with the standard 4x expansion and GELU activation."""

    def __init__(self, d_model: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.fc_in = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.activation(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """One decoder block: pre-LN attention + pre-LN MLP, both residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        ln_eps: float = 1e-5,
        fused_attention: bool = False,
    ):
        super().__init__()
        self.ln_attn = LayerNorm(d_model, eps=ln_eps, bias=bias)
        self.attn = MultiHeadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            bias=bias,
            fused_attention=fused_attention,
        )
        self.ln_mlp = LayerNorm(d_model, eps=ln_eps, bias=bias)
        self.mlp = FeedForward(d_model, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x


__all__ = ["FeedForward", "LayerNorm", "TransformerBlock"]
