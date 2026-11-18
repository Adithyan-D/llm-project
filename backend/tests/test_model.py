"""GPT model tests: config validation, shapes, causality, gradients, init."""

from __future__ import annotations

import pytest
import torch

from ml.models import GPT, GPTConfig, count_parameters
from ml.models.estimation import _analytic_parameter_count

SMALL = GPTConfig(
    vocab_size=64,
    context_length=16,
    d_model=32,
    n_heads=4,
    n_layers=2,
    dropout=0.0,
)


def make_model(config: GPTConfig = SMALL) -> GPT:
    torch.manual_seed(0)
    return GPT(config)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, d_model=10, n_heads=3)
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, dropout=1.0)
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=1)
    with pytest.raises(ValueError):
        GPTConfig.from_dict({"vocab_size": 10, "unknown": 1})
    assert SMALL.head_dim == 8


def test_forward_shapes_and_loss() -> None:
    model = make_model()
    idx = torch.randint(0, SMALL.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, SMALL.vocab_size)
    assert loss is None

    targets = torch.randint(0, SMALL.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_forward_rejects_overlong_sequences() -> None:
    model = make_model()
    with pytest.raises(ValueError):
        model(torch.randint(0, SMALL.vocab_size, (1, SMALL.context_length + 1)))


def test_targets_shape_mismatch_rejected() -> None:
    model = make_model()
    idx = torch.randint(0, SMALL.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        model(idx, torch.randint(0, SMALL.vocab_size, (1, 5)))


def test_logits_cannot_see_future_tokens() -> None:
    model = make_model()
    model.eval()
    idx = torch.randint(0, SMALL.vocab_size, (1, 8))
    modified = idx.clone()
    modified[0, -1] = (modified[0, -1] + 1) % SMALL.vocab_size

    with torch.no_grad():
        base, _ = model(idx)
        changed, _ = model(modified)

    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-6)
    assert not torch.allclose(base[:, -1], changed[:, -1])


def test_gradients_flow_to_all_trainable_parameters() -> None:
    model = make_model()
    idx = torch.randint(0, SMALL.vocab_size, (2, 8))
    targets = torch.randint(0, SMALL.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None and p.requires_grad]
    assert missing == []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_weight_tying_shares_storage() -> None:
    model = make_model()
    assert model.lm_head.weight is model.token_embedding.weight
    counts = count_parameters(model)
    assert counts.total == _analytic_parameter_count(SMALL)
    assert counts.tied_parameters == SMALL.vocab_size * SMALL.d_model


def test_untied_head_adds_vocab_projection() -> None:
    config = GPTConfig(**{**SMALL.to_dict(), "tie_weights": False})
    model = make_model(config)
    assert model.lm_head.weight is not model.token_embedding.weight
    assert count_parameters(model).total == _analytic_parameter_count(config)
    assert _analytic_parameter_count(config) - _analytic_parameter_count(SMALL) == (
        SMALL.vocab_size * SMALL.d_model
    )


def test_no_bias_variant_counts_match() -> None:
    config = GPTConfig(**{**SMALL.to_dict(), "bias": False})
    model = make_model(config)
    assert count_parameters(model).total == _analytic_parameter_count(config)


def test_initialisation_is_seed_deterministic() -> None:
    a = make_model()
    b = make_model()
    for (name_a, param_a), (name_b, param_b) in zip(
        a.named_parameters(), b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


def test_eval_mode_is_deterministic_with_dropout() -> None:
    config = GPTConfig(**{**SMALL.to_dict(), "dropout": 0.5})
    model = make_model(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        first, _ = model(idx)
        second, _ = model(idx)
    assert torch.equal(first, second)


def test_training_mode_dropout_is_stochastic() -> None:
    config = GPTConfig(**{**SMALL.to_dict(), "dropout": 0.5})
    model = make_model(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        first, _ = model(idx)
        second, _ = model(idx)
    assert not torch.equal(first, second)


def test_estimate_loss_restores_mode() -> None:
    model = make_model()
    model.train()
    idx = torch.randint(0, SMALL.vocab_size, (1, 4))
    targets = torch.randint(0, SMALL.vocab_size, (1, 4))
    result = model.estimate_loss([(idx, targets), (idx, targets)])
    assert result["batches"] == 2
    assert 0 < result["loss"] < 20
    assert model.training is True


def test_preset_parameter_counts_are_in_cpu_budget() -> None:
    from ml.models import preset

    for name in ("nano-2m", "tiny-5m"):
        config = preset(name)
        counts = count_parameters(make_model(config))
        assert 1_000_000 < counts.total < 8_000_000, (name, counts.total)
        assert counts.total == _analytic_parameter_count(config)


def test_fused_attention_flag_builds_equivalent_parameter_count() -> None:
    config = GPTConfig(**{**SMALL.to_dict(), "fused_attention": True})
    model = make_model(config)
    assert all(block.attn.fused_attention for block in model.blocks)
    assert count_parameters(model).total == _analytic_parameter_count(SMALL)

    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_config_json_roundtrip_preserves_fused_flag() -> None:
    config = GPTConfig(**{**SMALL.to_dict(), "fused_attention": True})
    assert GPTConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        GPTConfig(**{**SMALL.to_dict(), "fused_attention": "yes"})
