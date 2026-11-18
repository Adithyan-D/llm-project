"""Causal scaled dot-product attention and multi-head attention, from scratch.

Shapes used throughout::

    q, k, v        (batch, n_heads, seq, head_dim)
    attention      (batch, n_heads, seq, seq)     # rows = queries, cols = keys
    output         (batch, n_heads, seq, head_dim)

The causal mask is an additive-free boolean mask: positions ``j > i`` are set to
``-inf`` before softmax, so a token can only attend to itself and the past.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

_MASK_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a ``(seq_len, seq_len)`` boolean mask where True means "may attend".

    Row ``i`` allows columns ``0..i`` (inclusive). Keeping it boolean (not
    additive) makes intent explicit and lets tests assert on it directly.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


def cached_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Same mask, memoised per (length, device).

    Training calls attention once per layer per step; rebuilding the mask each
    time is pure overhead. The mask is constant, boolean and gradient-free, so
    caching is safe. Cache keys include the device to avoid cross-device reuse.
    """
    key = (seq_len, str(device))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        mask = build_causal_mask(seq_len, device=device)
        _MASK_CACHE[key] = mask
    return mask


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, C) -> (B, H, T, C/H)."""
    batch, seq, channels = x.shape
    if channels % n_heads != 0:
        raise ValueError(f"channels={channels} is not divisible by n_heads={n_heads}")
    head_dim = channels // n_heads
    return x.view(batch, seq, n_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, D) -> (B, T, H*D)."""
    batch, heads, seq, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq, heads * head_dim)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention written out step by step.

    ``softmax(q k^T / sqrt(d_k) + mask) v``

    Why write it manually instead of ``F.scaled_dot_product_attention``: this is
    the core mechanism the project is meant to demonstrate. ``MultiHeadAttention``
    can opt into the fused kernel via ``fused_attention=True``; this function
    stays the correctness reference and the default.

    Complexity: O(T^2 * d) time and O(T^2) memory per head, which is why context
    length drives CPU cost quadratically.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shape mismatch: {q.shape}, {k.shape}, {v.shape}")

    head_dim = q.size(-1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    seq_len = q.size(-2)
    if is_causal:
        causal = cached_causal_mask(seq_len, device=q.device)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        # attn_mask: broadcastable boolean, True = keep.
        scores = scores.masked_fill(~attn_mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return torch.matmul(weights, v)


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection.

    One ``Linear(d, 3d)`` produces Q, K and V for all heads at once; splitting
    heads is a view/transpose, which keeps memory traffic low compared with
    three separate projections.

    ``fused_attention=True`` dispatches to ``F.scaled_dot_product_attention``
    (a kernel specialised for the same maths). It is off by default so the
    reference implementation is what the model uses unless a caller opts in;
    parity between the two paths is asserted in the test suite.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        fused_attention: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.fused_attention = fused_attention
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, channels = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        q = split_heads(q, self.n_heads)
        k = split_heads(k, self.n_heads)
        v = split_heads(v, self.n_heads)
        if self.fused_attention:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.attn_dropout,
                training=self.training,
            )
        return self.resid_dropout(self.proj(merge_heads(out)))


__all__ = [
    "MultiHeadAttention",
    "build_causal_mask",
    "cached_causal_mask",
    "merge_heads",
    "scaled_dot_product_attention",
    "split_heads",
]
