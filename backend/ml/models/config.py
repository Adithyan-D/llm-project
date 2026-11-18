"""GPT model configuration with validation and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters of the decoder-only transformer.

    All fields affect parameter count or compute, so the config is the single
    source of truth written into checkpoints and experiment metadata.
    """

    vocab_size: int
    context_length: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True
    ln_eps: float = 1e-5
    init_std: float = 0.02
    fused_attention: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("d_model, n_heads and n_layers must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.ln_eps < 1.0:
            raise ValueError("ln_eps must be in (0, 1)")
        if not isinstance(self.fused_attention, bool):
            raise ValueError("fused_attention must be a bool")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GPTConfig:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown GPTConfig fields: {sorted(unknown)}")
        return cls(**payload)  # type: ignore[arg-type]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> GPTConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Hand-tuned presets for this CPU. Parameter counts are verified against the
# model in tests, not assumed: see `scripts/inspect_model.py`.
PRESETS: dict[str, dict[str, object]] = {
    "nano-2m": {
        "vocab_size": 1025,
        "context_length": 128,
        "d_model": 192,
        "n_heads": 4,
        "n_layers": 4,
        "dropout": 0.0,
    },
    "tiny-5m": {
        "vocab_size": 1025,
        "context_length": 128,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 6,
        "dropout": 0.0,
    },
}


def preset(name: str) -> GPTConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return GPTConfig(**PRESETS[name])  # type: ignore[arg-type]


__all__ = ["GPTConfig", "PRESETS", "preset"]
