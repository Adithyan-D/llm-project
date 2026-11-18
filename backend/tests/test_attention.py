"""Attention mechanics: masking, head reshaping, numerical equivalence."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from ml.models import (
    MultiHeadAttention,
    build_causal_mask,
    cached_causal_mask,
    merge_heads,
    scaled_dot_product_attention,
    split_heads,
)


def test_causal_mask_is_lower_triangular() -> None:
    mask = build_causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)
    with pytest.raises(ValueError):
        build_causal_mask(0)


def test_split_merge_heads_roundtrip() -> None:
    x = torch.randn(2, 5, 12)
    split = split_heads(x, 3)
    assert split.shape == (2, 3, 5, 4)
    assert torch.equal(merge_heads(split), x)
    with pytest.raises(ValueError):
        split_heads(x, 5)


def test_attention_matches_naive_reference() -> None:
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 3, 5, 4
    q = torch.randn(batch, heads, seq, head_dim)
    k = torch.randn(batch, heads, seq, head_dim)
    v = torch.randn(batch, heads, seq, head_dim)

    actual = scaled_dot_product_attention(q, k, v, is_causal=True)

    expected = torch.zeros_like(v)
    for i in range(seq):
        scores = torch.einsum("bhd,bhjd->bhj", q[:, :, i], k[:, :, : i + 1]) / math.sqrt(head_dim)
        weights = F.softmax(scores, dim=-1)
        expected[:, :, i] = torch.einsum("bhj,bhjd->bhd", weights, v[:, :, : i + 1])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_causal_attention_ignores_future_tokens() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)

    baseline = scaled_dot_product_attention(q, k, v)
    v_modified = v.clone()
    v_modified[:, :, 4:] = 100.0  # future positions only
    changed = scaled_dot_product_attention(q, k, v_modified)

    assert torch.allclose(baseline[:, :, :4], changed[:, :, :4], atol=1e-6)
    assert not torch.allclose(baseline[:, :, 4:], changed[:, :, 4:])


def test_non_causal_attention_sees_everything() -> None:
    q = torch.randn(1, 1, 3, 2)
    k = torch.randn(1, 1, 3, 2)
    v = torch.randn(1, 1, 3, 2)
    out1 = scaled_dot_product_attention(q, k, v, is_causal=False)
    v2 = v.clone()
    v2[:, :, 2] = 50.0
    out2 = scaled_dot_product_attention(q, k, v2, is_causal=False)
    # The last value influences every earlier position when not causal.
    assert not torch.allclose(out1[:, :, 0], out2[:, :, 0])


def test_explicit_attention_mask_can_hide_keys() -> None:
    q = torch.randn(1, 1, 3, 2)
    k = torch.randn(1, 1, 3, 2)
    v = torch.tensor([[[[1.0, 0.0], [9.0, 9.0], [8.0, 8.0]]]])
    mask = torch.tensor([[[[True, False, False]]]]).expand(1, 1, 3, 3)
    out = scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=mask)
    # Only the first key is visible, so every query returns v[0].
    assert torch.allclose(out, v[:, :, :1].expand_as(out), atol=1e-6)


def test_attention_shape_validation() -> None:
    q = torch.randn(1, 1, 3, 2)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, torch.randn(1, 1, 4, 2), q)


def test_mha_shapes_and_gradients() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, n_heads=4, dropout=0.0)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = mha(x)
    assert out.shape == (2, 5, 16)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in mha.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_mha_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=10, n_heads=3)


def test_head_split_preserves_per_head_slices() -> None:
    x = torch.arange(24, dtype=torch.float32).view(1, 4, 6)  # T=4, C=6, H=2
    split = split_heads(x, 2)
    # Head 0 must equal the first 3 channels, head 1 the last 3 channels.
    assert torch.equal(split[0, 0], x[0, :, :3])
    assert torch.equal(split[0, 1], x[0, :, 3:])


def test_cached_mask_is_reused_and_matches_fresh_mask() -> None:
    first = cached_causal_mask(6, torch.device("cpu"))
    second = cached_causal_mask(6, torch.device("cpu"))
    assert first is second  # memoised, not rebuilt
    assert torch.equal(first, build_causal_mask(6))


def test_fused_attention_matches_reference_path() -> None:
    torch.manual_seed(0)
    reference = MultiHeadAttention(d_model=16, n_heads=4, dropout=0.0)
    fused = MultiHeadAttention(d_model=16, n_heads=4, dropout=0.0, fused_attention=True)
    fused.load_state_dict(reference.state_dict())
    reference.eval()
    fused.eval()

    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        expected = reference(x)
        actual = fused(x)
    assert torch.allclose(actual, expected, atol=1e-5), (actual - expected).abs().max()


def test_fused_attention_gradients_match_reference() -> None:
    torch.manual_seed(0)
    reference = MultiHeadAttention(d_model=8, n_heads=2, dropout=0.0)
    fused = MultiHeadAttention(d_model=8, n_heads=2, dropout=0.0, fused_attention=True)
    fused.load_state_dict(reference.state_dict())
    reference.train()
    fused.train()

    x = torch.randn(1, 5, 8)
    reference(x).sum().backward()
    fused(x).sum().backward()
    for (name_a, param_a), (name_b, param_b) in zip(
        reference.named_parameters(), fused.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.allclose(param_a.grad, param_b.grad, atol=1e-5), name_a
