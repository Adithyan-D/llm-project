"""Learning-rate schedules: linear warmup followed by decay.

The schedule is a pure function of the step number. Nothing stateful is stored,
so a resumed run computes exactly the same learning rate as an uninterrupted one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SCHEDULES


@dataclass(frozen=True)
class LRSchedule:
    """Warmup + cosine/linear/constant decay.

    Step numbering is zero-based: ``lr_at(0)`` is the first update's rate. The
    warmup ramps from ``base_lr / warmup_steps`` (never exactly zero, which would
    make the first step a no-op) to ``base_lr`` at ``step == warmup_steps``.
    """

    max_steps: int
    base_lr: float
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    kind: str = "cosine"

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not 0.0 < self.base_lr <= 1.0:
            raise ValueError("base_lr must be in (0, 1]")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps)")
        if not 0.0 <= self.min_lr_ratio < 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1)")
        if self.kind not in SCHEDULES:
            raise ValueError(f"kind must be one of {SCHEDULES}")

    @property
    def min_lr(self) -> float:
        return self.base_lr * self.min_lr_ratio

    def lr_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be >= 0")
        if step >= self.max_steps:
            return self.min_lr
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        span = self.max_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / span if span > 0 else 1.0
        if self.kind == "constant":
            return self.base_lr
        if self.kind == "linear":
            return self.base_lr + (self.min_lr - self.base_lr) * progress
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )


__all__ = ["LRSchedule"]
