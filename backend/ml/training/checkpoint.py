"""Atomic checkpoints and resume.

A checkpoint contains everything needed to continue a run exactly where it
stopped: model weights, AdamW moments, the step counter, the data cursor, RNG
state and both configs. Files are written to a temporary name and moved into
place with ``os.replace``, so a crash mid-write cannot corrupt the last good
checkpoint.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .data import BatchCursor

CHECKPOINT_VERSION = 1
LATEST_FILENAME = "latest.json"


def save_checkpoint(
    run_dir: str | Path,
    *,
    step: int,
    tokens_seen: int,
    best_val_loss: float | None,
    cursor: BatchCursor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    history: list[dict[str, object]],
    stream_info: dict[str, object] | None = None,
) -> Path:
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / f"step-{step:06d}.pt"
    tmp_path = target_dir / f".step-{step:06d}.pt.tmp"

    payload = {
        "version": CHECKPOINT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "cursor": asdict(cursor),
        "model_config": config.model.to_dict(),
        "training_config": config.to_dict(),
        "stream_info": stream_info or {},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
        "history": history[-200:],
    }
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    latest = target_dir / LATEST_FILENAME
    latest_tmp = target_dir / f".{LATEST_FILENAME}.tmp"
    latest_tmp.write_text(
        json.dumps(
            {
                "checkpoint": str(final_path),
                "step": int(step),
                "created_at": payload["created_at"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(latest_tmp, latest)
    return final_path


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a checkpoint file, a run directory, or its ``latest.json``."""
    target = Path(path)
    if target.is_dir():
        target = target / LATEST_FILENAME
    if target.name == LATEST_FILENAME:
        if not target.is_file():
            raise FileNotFoundError(f"no checkpoint pointer at {target}")
        pointer_dir = target.parent
        pointer = json.loads(target.read_text(encoding="utf-8"))
        resolved = Path(pointer["checkpoint"])
        if not resolved.is_absolute():
            # The pointer stores the path as the trainer saw it, which is
            # relative to the working directory, not to the run directory.
            # Fall back to the run directory for moved/copied checkpoint dirs.
            candidates = [resolved]
            if (pointer_dir / resolved) != resolved:
                candidates.append(pointer_dir / resolved)
            # A run directory that was moved keeps its filenames, so the last
            # resort is the same filename next to the pointer.
            if (pointer_dir / resolved.name) != resolved:
                candidates.append(pointer_dir / resolved.name)
            resolved = next((c for c in candidates if c.is_file()), resolved)
        target = resolved
    if not target.is_file():
        raise FileNotFoundError(f"checkpoint not found: {target}")
    return target


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
    require_matching_model: bool = True,
) -> dict[str, object]:
    """Restore a run and return its metadata.

    Checkpoints are pickles, so only load files produced by this project.
    """
    checkpoint_path = resolve_checkpoint(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{checkpoint_path}: unsupported checkpoint format")
    for key in ("step", "model", "optimizer", "cursor", "model_config"):
        if key not in payload:
            raise ValueError(f"{checkpoint_path}: checkpoint is missing {key!r}")

    expected = model.config.to_dict()
    if require_matching_model and payload["model_config"] != expected:
        raise ValueError(
            f"{checkpoint_path}: model config does not match the current model; "
            f"checkpoint={payload['model_config']} current={expected}"
        )

    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if restore_rng and isinstance(payload.get("rng"), dict):
        torch.set_rng_state(payload["rng"]["torch"])
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])

    cursor = payload["cursor"]
    if not isinstance(cursor, dict):
        raise ValueError(f"{checkpoint_path}: bad cursor {cursor!r}")

    return {
        "path": str(checkpoint_path),
        "step": int(payload["step"]),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "best_val_loss": payload.get("best_val_loss"),
        "cursor": BatchCursor(
            epoch=int(cursor.get("epoch", 0)), batch_index=int(cursor.get("batch_index", 0))
        ),
        "history": list(payload.get("history", [])),
        "training_config": payload.get("training_config", {}),
        "stream_info": payload.get("stream_info", {}),
    }


__all__ = [
    "CHECKPOINT_VERSION",
    "LATEST_FILENAME",
    "load_checkpoint",
    "resolve_checkpoint",
    "save_checkpoint",
]
