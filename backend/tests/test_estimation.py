"""Parameter counting and memory estimation tests."""

from __future__ import annotations

import pytest
import torch

from ml.models import GPT, GPTConfig, count_parameters, estimate_memory, parameter_breakdown
from ml.models.estimation import human_bytes

CONFIG = GPTConfig(vocab_size=128, context_length=32, d_model=64, n_heads=4, n_layers=2)
MODEL = GPT(CONFIG)


def test_breakdown_sums_to_reported_total() -> None:
    rows = parameter_breakdown(MODEL)
    counted = sum(int(row["numel"]) for row in rows)
    assert counted == count_parameters(MODEL).total
    assert any(row["tied_to_earlier"] for row in rows)


def test_trainable_matches_requires_grad() -> None:
    for parameter in MODEL.parameters():
        parameter.requires_grad_(False)
    counts = count_parameters(MODEL)
    assert counts.trainable == 0
    assert counts.frozen == counts.total
    for parameter in MODEL.parameters():
        parameter.requires_grad_(True)


def test_memory_estimate_includes_all_components() -> None:
    estimate = estimate_memory(CONFIG, batch_size=4, seq_len=16)
    assert estimate["total_bytes"] == (
        estimate["weights_bytes"]
        + estimate["gradients_bytes"]
        + estimate["optimizer_bytes"]
        + estimate["activations_bytes"]
    )
    assert estimate["parameters"] == count_parameters(MODEL).total


def test_memory_scales_with_batch_and_sequence() -> None:
    small = estimate_memory(CONFIG, batch_size=1, seq_len=8)
    larger_batch = estimate_memory(CONFIG, batch_size=4, seq_len=8)
    larger_seq = estimate_memory(CONFIG, batch_size=1, seq_len=32)
    assert larger_batch["activations_bytes"] > small["activations_bytes"]
    assert larger_seq["activations_bytes"] > larger_batch["activations_bytes"]


def test_activation_checkpointing_reduces_estimate() -> None:
    plain = estimate_memory(CONFIG, batch_size=2, seq_len=16)
    checkpointed = estimate_memory(CONFIG, batch_size=2, seq_len=16, activation_checkpointing=True)
    assert checkpointed["activations_bytes"] < plain["activations_bytes"]


def test_sgd_has_no_optimizer_state() -> None:
    estimate = estimate_memory(CONFIG, batch_size=1, seq_len=8, optimizer="sgd")
    assert estimate["optimizer_bytes"] == 0
    with pytest.raises(ValueError):
        estimate_memory(CONFIG, batch_size=1, seq_len=8, optimizer="lamb")


def test_sequence_longer_than_context_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_memory(CONFIG, batch_size=1, seq_len=CONFIG.context_length + 1)


def test_invalid_shapes_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_memory(CONFIG, batch_size=0, seq_len=8)
    with pytest.raises(ValueError):
        estimate_memory(CONFIG, batch_size=1, seq_len=0)


def test_human_bytes_formatting() -> None:
    assert human_bytes(512) == "512.0 B"
    assert human_bytes(1536) == "1.5 KiB"
    assert human_bytes(1024**2 * 3) == "3.0 MiB"
    assert human_bytes(4 * 1024**3) == "4.0 GiB"


def test_estimate_reports_measured_vs_analytic_note() -> None:
    estimate = estimate_memory(CONFIG, batch_size=2, seq_len=8)
    assert "analytic" in str(estimate["note"])


def test_gradient_memory_scales_with_trainable_fraction() -> None:
    model = GPT(CONFIG)
    for parameter in model.token_embedding.parameters():
        parameter.requires_grad_(False)
    counts = count_parameters(model)
    assert counts.trainable < counts.total
    assert torch.isfinite(torch.tensor(float(counts.total)))
