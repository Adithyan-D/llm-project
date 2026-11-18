"""Generation policy tests: sampling controls, determinism, stopping criteria."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ml.models import GPT, GenerationConfig, GPTConfig
from ml.models.generation import (
    apply_repetition_penalty,
    filter_top_k,
    filter_top_p,
    generate,
)

CONFIG = GPTConfig(vocab_size=32, context_length=16, d_model=32, n_heads=4, n_layers=1)


def model() -> GPT:
    torch.manual_seed(0)
    return GPT(CONFIG)


class FixedLogitsModel(nn.Module):
    """Deterministic stub: always prefers ``winner`` (optionally after some tokens)."""

    def __init__(self, vocab: int = 8, winner: int = 3, context_length: int = 8):
        super().__init__()
        self.config = GPTConfig(
            vocab_size=vocab, context_length=context_length, d_model=8, n_heads=2
        )
        self.winner = winner
        self.linear = nn.Linear(1, 1)

    def forward(self, idx: torch.Tensor):
        vocab = self.config.vocab_size
        logits = torch.full((idx.size(0), idx.size(1), vocab), -10.0)
        logits[:, :, self.winner] = 10.0
        return logits, None


def test_generation_config_validation() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=-1)
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-0.1)
    with pytest.raises(ValueError):
        GenerationConfig(top_k=0)
    with pytest.raises(ValueError):
        GenerationConfig(top_p=0.0)
    with pytest.raises(ValueError):
        GenerationConfig(repetition_penalty=0.0)


def test_greedy_generation_length_and_determinism() -> None:
    gpt = model()
    prompt = torch.randint(0, CONFIG.vocab_size, (1, 4))
    cfg = GenerationConfig(max_new_tokens=6, greedy=True)
    first = generate(gpt, prompt, cfg)
    second = generate(gpt, prompt, cfg)
    assert first.shape == (1, 10)
    assert torch.equal(first, second)
    assert torch.equal(first[:, :4], prompt)


def test_temperature_zero_matches_greedy() -> None:
    gpt = model()
    prompt = torch.randint(0, CONFIG.vocab_size, (1, 3))
    greedy = generate(gpt, prompt, GenerationConfig(max_new_tokens=5, greedy=True))
    zero = generate(gpt, prompt, GenerationConfig(max_new_tokens=5, temperature=0.0))
    assert torch.equal(greedy, zero)


def test_top_k_one_is_greedy() -> None:
    gpt = model()
    prompt = torch.randint(0, CONFIG.vocab_size, (1, 3))
    greedy = generate(gpt, prompt, GenerationConfig(max_new_tokens=5, greedy=True))
    top1 = generate(gpt, prompt, GenerationConfig(max_new_tokens=5, top_k=1, seed=1))
    assert torch.equal(greedy, top1)


def test_sampling_is_seed_reproducible() -> None:
    gpt = model()
    prompt = torch.randint(0, CONFIG.vocab_size, (1, 3))
    cfg = GenerationConfig(max_new_tokens=8, temperature=1.2, seed=1234)
    assert torch.equal(generate(gpt, prompt, cfg), generate(gpt, prompt, cfg))
    other = GenerationConfig(max_new_tokens=8, temperature=1.2, seed=99)
    assert not torch.equal(generate(gpt, prompt, cfg), generate(gpt, prompt, other))


def test_top_k_limits_support_to_k_tokens() -> None:
    logits = torch.randn(2, 10)
    filtered = filter_top_k(logits, 3)
    finite = torch.isfinite(filtered).sum(dim=-1)
    assert torch.equal(finite, torch.tensor([3, 3]))
    # Kept logits are exactly the top-3 values.
    assert torch.allclose(
        filtered.sort(dim=-1, descending=True).values[:, :3],
        logits.sort(dim=-1, descending=True).values[:, :3],
    )


def test_top_k_larger_than_vocab_is_safe() -> None:
    logits = torch.randn(1, 4)
    assert torch.equal(filter_top_k(logits, 99), logits)


def test_top_p_keeps_mass_and_at_least_one_token() -> None:
    logits = torch.randn(2, 12)
    filtered = filter_top_p(logits, 0.5)
    finite = torch.isfinite(filtered)
    assert finite.sum(dim=-1).min() >= 1
    for row in range(2):
        kept = torch.softmax(filtered[row, finite[row]], dim=-1).sum()
        assert kept >= 0.5 - 1e-5


def test_top_p_one_keeps_everything() -> None:
    logits = torch.randn(1, 6)
    filtered = filter_top_p(logits, 1.0)
    assert torch.allclose(filtered, logits)


def test_repetition_penalty_matches_manual_math() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0, 0.0]])
    tokens = torch.tensor([[0, 2]])
    adjusted = apply_repetition_penalty(logits, tokens, 2.0)
    assert torch.allclose(adjusted, torch.tensor([[1.0, 1.0, -2.0, 0.0]]))
    assert torch.allclose(apply_repetition_penalty(logits, tokens, 1.0), logits)


def test_repetition_penalty_reduces_repeat_probability() -> None:
    torch.manual_seed(0)
    logits = torch.zeros(1, 8)
    logits[0, 4] = 2.0
    tokens = torch.tensor([[4, 4, 4]])
    adjusted = apply_repetition_penalty(logits, tokens, 2.0)
    before = F.softmax(logits, dim=-1)[0, 4]
    after = F.softmax(adjusted, dim=-1)[0, 4]
    assert after < before


def test_eos_stops_generation() -> None:
    stub = FixedLogitsModel(winner=3)
    prompt = torch.tensor([[1, 2]])
    out = generate(stub, prompt, GenerationConfig(max_new_tokens=10, greedy=True, eos_token_id=3))
    assert out.shape == (1, 3)
    assert out[0, -1].item() == 3


def test_prompt_is_cropped_to_context_window() -> None:
    stub = FixedLogitsModel(context_length=4)
    prompt = torch.arange(1, 7).unsqueeze(0)  # 6 tokens > context 4
    out = generate(stub, prompt, GenerationConfig(max_new_tokens=2, greedy=True))
    assert out.shape == (1, 8)


def test_generate_rejects_bad_inputs() -> None:
    gpt = model()
    with pytest.raises(ValueError):
        generate(gpt, torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        generate(gpt, torch.zeros(1, 0, dtype=torch.long))


def test_generate_restores_training_mode() -> None:
    gpt = model()
    gpt.train()
    generate(gpt, torch.randint(0, CONFIG.vocab_size, (1, 2)), GenerationConfig(max_new_tokens=1))
    assert gpt.training is True


def test_generate_batch_rows_are_independent() -> None:
    gpt = model()
    prompt = torch.randint(0, CONFIG.vocab_size, (3, 3))
    out = generate(gpt, prompt, GenerationConfig(max_new_tokens=2, greedy=True))
    assert out.shape == (3, 5)
    single = generate(gpt, prompt[1:2], GenerationConfig(max_new_tokens=2, greedy=True))
    assert torch.equal(out[1:2], single)
