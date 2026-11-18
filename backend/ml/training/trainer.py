"""Pretraining loop: batched CE loss, AdamW, schedule, clipping, eval, resume."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..models import GPT, count_parameters, estimate_memory
from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .data import (
    Batch,
    BatchCursor,
    PackedSampler,
    TokenStreams,
    build_token_streams,
    evaluation_batches,
)
from .safety import (
    TrainingOOMError,
    available_memory_bytes,
    ensure_memory_budget,
    is_oom_error,
)
from .scheduler import LRSchedule


class Trainer:
    """Owns the model, optimiser, schedule, data cursor and logs for one run."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        streams: TokenStreams | None = None,
        model: nn.Module | None = None,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.verbose = verbose
        torch.set_num_threads(config.num_threads)
        self._seed_everything(config.seed)

        self.streams = streams or build_token_streams(config.dataset_dir, config.tokenizer_path)
        if self.streams.vocab_size != config.model.vocab_size:
            raise ValueError(
                f"tokenizer vocabulary ({self.streams.vocab_size}) does not match "
                f"model vocab_size ({config.model.vocab_size})"
            )

        self.model = model or GPT(config.model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.schedule = LRSchedule(
            max_steps=config.max_steps,
            base_lr=config.lr,
            warmup_steps=config.warmup_steps,
            min_lr_ratio=config.min_lr_ratio,
            kind=config.schedule,
        )

        train_tokens = self.streams.load("train")
        val_tokens = self.streams.load("val")
        self.train_sampler = PackedSampler(
            train_tokens,
            context_length=config.model.context_length,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        self.val_batches = evaluation_batches(
            val_tokens,
            context_length=config.model.context_length,
            batch_size=config.batch_size,
            max_batches=config.eval_batches,
        )

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss: float | None = None
        self.cursor = BatchCursor()
        self.history: list[dict[str, object]] = []
        self.last_checkpoint: Path | None = None

        self.run_dir = Path(config.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train_log.jsonl"

        self._loss_window: list[float] = []
        self._last_loss: float | None = None
        self._first_step_loss: float | None = None
        self._tokens_at_start = 0
        self._start_time: float | None = None
        self._active_max_steps = config.max_steps

    # ------------------------------------------------------------------ setup
    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def tokens_per_epoch(self) -> int:
        return (
            self.train_sampler.steps_per_epoch
            * self.config.batch_size
            * self.config.model.context_length
        )

    def memory_estimate(self) -> dict[str, object]:
        counts = count_parameters(self.model)
        return estimate_memory(
            self.config.model,
            batch_size=self.config.batch_size,
            seq_len=self.config.model.context_length,
            total_parameters=counts.total,
        )

    def check_memory(self, *, max_fraction: float = 0.7, force: bool = False) -> dict[str, object]:
        """Refuse to start when the analytic footprint exceeds the RAM budget."""
        estimate = self.memory_estimate()
        available = available_memory_bytes()
        if not force and available is not None:
            ensure_memory_budget(
                int(estimate["total_bytes"]),
                available,
                max_fraction=max_fraction,
                context=f"{self.config.model.n_layers} layers, batch {self.config.batch_size}",
            )
        return estimate

    def stream_info(self) -> dict[str, object]:
        manifest = self.streams.manifest
        return {
            "dataset_id": manifest.get("dataset_id"),
            "dataset_hash": manifest.get("dataset_hash"),
            "tokenizer_sha256": manifest.get("tokenizer_sha256"),
            "eos_token_id": manifest.get("eos_token_id"),
            "splits": {name: stream.tokens for name, stream in self.streams.splits.items()},
        }

    # ------------------------------------------------------------------- steps
    def train_step(self, batches: list[Batch]) -> tuple[float, float, float]:
        """One optimizer step over micro-batches; returns (loss, grad_norm, lr).

        Losses are divided by the number of micro-batches so the accumulated
        gradient equals the gradient of the mean loss over the full batch.
        """
        if not batches:
            raise ValueError("train_step needs at least one batch")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for batch in batches:
            try:
                _, loss = self.model(batch.inputs, batch.targets)
            except RuntimeError as exc:
                if is_oom_error(exc):
                    raise TrainingOOMError(self._oom_help()) from exc
                raise
            if loss is None:
                raise RuntimeError("model returned no loss")
            (loss / len(batches)).backward()
            total += float(loss.detach())

        lr = self.schedule.lr_at(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        max_norm = self.config.grad_clip if self.config.grad_clip > 0 else math.inf
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm))
        self.optimizer.step()

        mean_loss = total / len(batches)
        if self._first_step_loss is None:
            self._first_step_loss = mean_loss
        self.step += 1
        self.tokens_seen += sum(len(batch) * batch.inputs.size(1) for batch in batches)
        return mean_loss, grad_norm, lr

    def evaluate(self) -> float:
        """Mean loss over the fixed validation batches (eval mode, no grad)."""
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        try:
            with torch.no_grad():
                for batch in self.val_batches:
                    try:
                        _, loss = self.model(batch.inputs, batch.targets)
                    except RuntimeError as exc:
                        if is_oom_error(exc):
                            raise TrainingOOMError(self._oom_help()) from exc
                        raise
                    if loss is None:
                        raise RuntimeError("model returned no loss")
                    losses.append(float(loss.detach()))
        finally:
            self.model.train(was_training)
        return sum(losses) / len(losses)

    # -------------------------------------------------------------------- run
    def fit(
        self,
        *,
        max_steps: int | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, object]:
        if resume_from is not None:
            self.resume(resume_from)
        target = self.config.max_steps if max_steps is None else max_steps
        if target <= self.step:
            raise ValueError(
                f"max_steps={target} must be greater than the current step ({self.step})"
            )
        self._active_max_steps = target
        self._tokens_at_start = self.tokens_seen
        self._start_time = time.perf_counter()

        while self.step < target:
            batches: list[Batch] = []
            for _ in range(self.config.grad_accum_steps):
                batch, self.cursor = self.train_sampler.advance(self.cursor)
                batches.append(batch)
            loss, grad_norm, lr = self.train_step(batches)
            self._loss_window.append(loss)
            self._last_loss = loss

            is_eval = self.step % self.config.eval_interval == 0
            is_log = self.step % self.config.log_interval == 0
            if is_eval:
                val_loss = self.evaluate()
                record = self._record(lr, grad_norm, val_loss)
                self.history.append(record)
                self._append_log(record)
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                self._print_eval(record)
            elif is_log:
                record = self._record(lr, grad_norm, None)
                self.history.append(record)
                self._append_log(record)
                self._print_train(record)

            if self.step % self.config.checkpoint_interval == 0 or self.step >= target:
                self._save()

        return self.report()

    def resume(self, path: str | Path) -> None:
        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            restore_rng=True,
        )
        recorded = state.get("stream_info") or {}
        current = self.stream_info()
        for key in ("dataset_hash", "tokenizer_sha256", "eos_token_id"):
            if recorded.get(key) is not None and recorded.get(key) != current.get(key):
                raise ValueError(
                    f"checkpoint was trained on a different {key}: "
                    f"{recorded.get(key)!r} != {current.get(key)!r}"
                )
        self.step = int(state["step"])
        self.tokens_seen = int(state["tokens_seen"])
        best = state.get("best_val_loss")
        self.best_val_loss = None if best is None else float(best)
        self.cursor = state["cursor"]  # type: ignore[assignment]
        self.history = list(state["history"])

        recorded_config = state.get("training_config")
        if isinstance(recorded_config, dict):
            recorded_max = recorded_config.get("max_steps")
            if isinstance(recorded_max, int) and recorded_max != self.config.max_steps:
                # Keep the original LR horizon so an extended run does not
                # silently jump to a different point on the curve.
                self.schedule = LRSchedule(
                    max_steps=recorded_max,
                    base_lr=self.config.lr,
                    warmup_steps=max(0, min(self.config.warmup_steps, recorded_max - 1)),
                    min_lr_ratio=self.config.min_lr_ratio,
                    kind=self.config.schedule,
                )
                if self.verbose:
                    print(f"schedule      : keeping checkpoint horizon of {recorded_max} steps")
        if self.verbose:
            print(f"resumed      : step {self.step}, {self.tokens_seen:,} tokens seen")

    # ------------------------------------------------------------------ logs
    def _record(self, lr: float, grad_norm: float, val_loss: float | None) -> dict[str, object]:
        if self._loss_window:
            train_loss = sum(self._loss_window) / len(self._loss_window)
            self._loss_window.clear()
        else:
            train_loss = float(self._last_loss or 0.0)
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        run_tokens = self.tokens_seen - self._tokens_at_start
        record = {
            "step": self.step,
            "epoch": self.cursor.epoch,
            "tokens_seen": self.tokens_seen,
            "lr": round(lr, 9),
            "train_loss": round(train_loss, 6),
            "val_loss": None if val_loss is None else round(val_loss, 6),
            "grad_norm": round(grad_norm, 6),
            "elapsed_s": round(elapsed, 3),
            "tokens_per_s": round(run_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        }
        return record

    def _append_log(self, record: dict[str, object]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _print_train(self, record: dict[str, object]) -> None:
        if not self.verbose:
            return
        print(
            f"step {record['step']:>4}/{self._active_max_steps} | "
            f"train {record['train_loss']:.4f} | "
            f"lr {record['lr']:.2e} | "
            f"{record['tokens_per_s']:,.0f} tok/s"
        )

    def _print_eval(self, record: dict[str, object]) -> None:
        if not self.verbose:
            return
        print(
            f"step {record['step']:>4}/{self._active_max_steps} | "
            f"train {record['train_loss']:.4f} | val {record['val_loss']:.4f} | "
            f"lr {record['lr']:.2e} | tok {record['tokens_seen']:,} | "
            f"{record['tokens_per_s']:,.0f} tok/s"
        )

    def _save(self) -> Path:
        path = save_checkpoint(
            self.run_dir,
            step=self.step,
            tokens_seen=self.tokens_seen,
            best_val_loss=self.best_val_loss,
            cursor=self.cursor,
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            history=self.history,
            stream_info=self.stream_info(),
        )
        self.last_checkpoint = path
        if self.verbose:
            print(f"checkpoint    : {path}")
        return path

    def _oom_help(self) -> str:
        estimate = self.memory_estimate()
        return (
            "CPU allocator ran out of memory. "
            f"Analytic estimate for this config: {estimate['total_human']}. "
            "Reduce batch_size, context_length or grad_accum_steps, close other "
            "processes, or rerun with a smaller model preset."
        )

    # ----------------------------------------------------------------- report
    def report(self) -> dict[str, object]:
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
        run_tokens = self.tokens_seen - self._tokens_at_start
        val_losses = [r["val_loss"] for r in self.history if r.get("val_loss") is not None]
        return {
            "run_dir": str(self.run_dir),
            "checkpoint": None if self.last_checkpoint is None else str(self.last_checkpoint),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "tokens_this_run": run_tokens,
            "wall_seconds": round(elapsed, 3),
            "tokens_per_second": round(run_tokens / elapsed, 1) if elapsed > 0 else 0.0,
            "tokens_per_epoch": self.tokens_per_epoch,
            "steps_per_epoch": self.train_sampler.steps_per_epoch,
            "first_train_loss": self.history[0]["train_loss"] if self.history else None,
            "first_step_loss": (
                None if self._first_step_loss is None else round(self._first_step_loss, 6)
            ),
            "final_train_loss": None if self._last_loss is None else round(self._last_loss, 6),
            "final_val_loss": None if not val_losses else val_losses[-1],
            "best_val_loss": self.best_val_loss,
            "parameters": count_parameters(self.model).as_dict(),
            "config": self.config.to_dict(),
            "environment": {
                "torch": torch.__version__,
                "threads": self.config.num_threads,
                "device": self.config.device,
            },
            "history": self.history,
        }


__all__ = ["Trainer"]
