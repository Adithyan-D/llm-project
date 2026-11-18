"""Training configuration for the from-scratch GPT pretraining loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..models.config import GPTConfig

SCHEDULES = ("cosine", "linear", "constant")


@dataclass(frozen=True)
class TrainingConfig:
    """Everything needed to reproduce a pretraining run.

    The model architecture is embedded as a ``GPTConfig`` so a checkpoint can
    be self-describing: resuming checks that the architecture matches instead of
    silently loading incompatible tensors.
    """

    model: GPTConfig
    dataset_dir: str
    tokenizer_path: str
    out_dir: str = "checkpoints/run"
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 400
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 20
    schedule: str = "cosine"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 10
    checkpoint_interval: int = 200
    log_interval: int = 20
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.model, GPTConfig):
            raise ValueError("model must be a GPTConfig")
        if not self.dataset_dir:
            raise ValueError("dataset_dir must not be empty")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not 0.0 < self.lr <= 1.0:
            raise ValueError("lr must be in (0, 1]")
        if not 0.0 <= self.min_lr_ratio < 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1)")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be < max_steps")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be >= 0")
        if (
            min(self.eval_interval, self.eval_batches, self.checkpoint_interval, self.log_interval)
            < 1
        ):
            raise ValueError("intervals and eval_batches must be >= 1")
        if self.device != "cpu":
            raise ValueError(
                f"device={self.device!r} is not supported: this torch build is CPU-only"
            )
        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed by one optimizer step (all micro-batches)."""
        return self.batch_size * self.grad_accum_steps * self.model.context_length

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TrainingConfig:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown TrainingConfig fields: {sorted(unknown)}")
        data = dict(payload)
        model_payload = data.get("model")
        if isinstance(model_payload, dict):
            data["model"] = GPTConfig.from_dict(model_payload)
        return cls(**data)  # type: ignore[arg-type]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["SCHEDULES", "TrainingConfig"]
