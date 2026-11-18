"""Autoregressive sampling: temperature, top-k, top-p and repetition penalty.

The model exposes logits; this module owns the *decision* of which token comes
next, which keeps generation policies testable without a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters. Defaults are deterministic (greedy)."""

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    greedy: bool = False
    seed: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 or None")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1] or None")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")


def apply_repetition_penalty(
    logits: torch.Tensor, tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style penalty: shrink positive logits and grow negative ones.

    Applied once per step over the *unique* tokens already generated, so a word
    repeated five times is not penalised five times.
    """
    if penalty <= 0:
        raise ValueError("penalty must be > 0")
    if penalty == 1.0:
        return logits
    scores = logits.clone()
    for row in range(scores.size(0)):
        unique = torch.unique(tokens[row])
        values = scores[row, unique]
        scores[row, unique] = torch.where(values > 0, values / penalty, values * penalty)
    return scores


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest logits per row, set the rest to -inf."""
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, logits.size(-1))
    threshold = logits.topk(k, dim=-1).values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose mass reaches p.

    The highest-probability token is always kept, even if its own probability
    exceeds p, otherwise the distribution could become empty.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = (cumulative - sorted_probs) > p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(dim=-1, index=sorted_indices, src=sorted_logits)


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    config: GenerationConfig | None = None,
) -> torch.Tensor:
    """Generate tokens greedily or by sampling, returning prompt + continuation.

    The model is evaluated in eval mode (dropout off) and the input is cropped to
    the last ``context_length`` tokens each step. No KV cache yet: every step
    recomputes the full window, so cost per token grows with context length.
    """
    cfg = config or GenerationConfig()
    if idx.dim() != 2:
        raise ValueError(f"expected (batch, seq) input, got shape {tuple(idx.shape)}")
    if idx.size(1) == 0:
        raise ValueError("prompt must contain at least one token")

    context_length = getattr(getattr(model, "config", None), "context_length", None)
    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.seed)

    was_training = model.training
    model.eval()
    tokens = idx
    try:
        for _ in range(cfg.max_new_tokens):
            window = tokens[:, -context_length:] if context_length else tokens
            logits, _ = model(window)
            next_logits = logits[:, -1, :].float()

            if cfg.repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, tokens, cfg.repetition_penalty)

            if cfg.greedy or cfg.temperature == 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / cfg.temperature
                if cfg.top_k is not None:
                    next_logits = filter_top_k(next_logits, cfg.top_k)
                if cfg.top_p is not None:
                    next_logits = filter_top_p(next_logits, cfg.top_p)
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities.cpu(), num_samples=1, generator=generator
                ).to(tokens.device)

            tokens = torch.cat((tokens, next_token), dim=1)
            if cfg.eos_token_id is not None and bool((next_token == cfg.eos_token_id).all()):
                break
    finally:
        model.train(was_training)
    return tokens


__all__ = [
    "GenerationConfig",
    "apply_repetition_penalty",
    "filter_top_k",
    "filter_top_p",
    "generate",
]
