"""Training engine tests: token streams, packing, schedule, resume, OOM guards."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.data import (
    DatasetConfig,
    NormalizationConfig,
    iter_split,
    process_dataset,
)
from ml.models import GPT, GPTConfig
from ml.tokenization import BPETokenizer
from ml.training import (
    Batch,
    BatchCursor,
    InsufficientMemoryError,
    LRSchedule,
    PackedSampler,
    SplitStream,
    TokenStreams,
    Trainer,
    TrainingConfig,
    TrainingOOMError,
    available_memory_bytes,
    build_token_streams,
    ensure_memory_budget,
    evaluation_batches,
    is_oom_error,
    load_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)

SMALL_MODEL = GPTConfig(
    vocab_size=64,
    context_length=8,
    d_model=32,
    n_heads=4,
    n_layers=1,
    dropout=0.0,
)


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ------------------------------------------------------------------ fixtures
def make_dataset(root: Path) -> Path:
    lines = [f"line {i}: the quick brown fox jumps over the lazy dog {i % 7}" for i in range(48)]
    source = write(root / "corpus.txt", "\n".join(lines) + "\n")
    result = process_dataset(
        DatasetConfig(
            name="training-unit",
            inputs=[str(source)],
            output_dir=str(root / "processed"),
            val_fraction=0.25,
            test_fraction=0.0,
            seed=3,
            deduplicate=False,
            normalization=NormalizationConfig(min_chars=1),
        )
    )
    return Path(result.output_dir)


def make_tokenizer(
    dataset_dir: Path, root: Path, vocab_size: int = 300
) -> tuple[Path, BPETokenizer]:
    texts = list(iter_split(dataset_dir, "train"))
    tokenizer = BPETokenizer.train(
        texts,
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<|endoftext|>"],
    )
    path = root / "tokenizer"
    tokenizer.save(path)
    return path, tokenizer


def fake_streams(
    root: Path, *, vocab_size: int = 64, eos_token_id: int | None = None
) -> TokenStreams:
    """Self-contained token streams so trainer tests do not need a real corpus."""
    root.mkdir(parents=True, exist_ok=True)
    splits: dict[str, SplitStream] = {}
    recorded: dict[str, dict[str, object]] = {}
    for name, repeats in (("train", 600), ("val", 100)):
        array = np.tile(np.arange(vocab_size, dtype=np.int32), repeats)
        path = root / f"{name}.npy"
        np.save(path, array)
        splits[name] = SplitStream(
            name=name,
            path=path,
            tokens=int(array.size),
            documents=repeats,
            chars=int(array.size),
            sha256="0" * 64,
        )
        recorded[name] = {
            "file": path.name,
            "tokens": int(array.size),
            "documents": repeats,
            "chars": int(array.size),
            "sha256": "0" * 64,
        }
    manifest = {
        "version": 1,
        "dataset_id": "fake",
        "dataset_hash": "fake-hash",
        "tokenizer_path": "fake",
        "tokenizer_sha256": "1" * 64,
        "vocab_size": vocab_size,
        "eos_token": "<|endoftext|>" if eos_token_id is not None else None,
        "eos_token_id": eos_token_id,
        "splits": recorded,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return TokenStreams(
        cache_dir=root,
        manifest_path=manifest_path,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        splits=splits,
        manifest=manifest,
    )


def base_config(root: Path, streams: TokenStreams, **overrides) -> TrainingConfig:
    values: dict[str, object] = {
        "model": replace(SMALL_MODEL, vocab_size=streams.vocab_size),
        "dataset_dir": "unused",
        "tokenizer_path": "unused",
        "out_dir": str(root / "run"),
        "batch_size": 4,
        "grad_accum_steps": 1,
        "max_steps": 6,
        "lr": 0.01,
        "min_lr_ratio": 0.5,
        "warmup_steps": 2,
        "schedule": "cosine",
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "eval_interval": 3,
        "eval_batches": 2,
        "checkpoint_interval": 3,
        "log_interval": 3,
        "seed": 0,
        "num_threads": 2,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------- token streams
def test_token_stream_appends_eos_and_matches_tokenizer(tmp_path: Path) -> None:
    dataset_dir = make_dataset(tmp_path)
    tokenizer_path, tokenizer = make_tokenizer(dataset_dir, tmp_path)
    streams = build_token_streams(dataset_dir, tokenizer_path)

    eos = tokenizer.special_tokens["<|endoftext|>"]
    expected_train: list[int] = []
    for text in iter_split(dataset_dir, "train"):
        expected_train.extend(tokenizer.encode(text))
        expected_train.append(eos)

    train = streams.load("train")
    assert train.dtype == np.int32
    assert np.array_equal(train, np.asarray(expected_train, dtype=np.int32))
    assert streams.eos_token_id == eos
    assert streams.vocab_size == len(tokenizer)
    assert streams.splits["train"].documents == len(list(iter_split(dataset_dir, "train")))
    assert streams.splits["val"].tokens > 0
    for name in ("train", "val"):
        assert streams.splits[name].path.is_file()


def test_token_stream_cache_is_reused_without_reencoding(tmp_path: Path) -> None:
    dataset_dir = make_dataset(tmp_path)
    tokenizer_path, _ = make_tokenizer(dataset_dir, tmp_path)
    first = build_token_streams(dataset_dir, tokenizer_path)
    mtimes = {name: stream.path.stat().st_mtime_ns for name, stream in first.splits.items()}

    second = build_token_streams(dataset_dir, tokenizer_path)
    assert second.cache_dir == first.cache_dir
    assert second.manifest["created_at"] == first.manifest["created_at"]
    assert {name: s.path.stat().st_mtime_ns for name, s in second.splits.items()} == mtimes


def test_changing_the_tokenizer_builds_a_separate_cache(tmp_path: Path) -> None:
    dataset_dir = make_dataset(tmp_path)
    tokenizer_path, _ = make_tokenizer(dataset_dir, tmp_path, vocab_size=300)
    first = build_token_streams(dataset_dir, tokenizer_path)

    loaded = BPETokenizer.load(tokenizer_path)
    loaded.special_tokens["<|pad|>"] = len(loaded.vocab) + len(loaded.special_tokens)
    loaded.save(tokenizer_path)

    second = build_token_streams(dataset_dir, tokenizer_path)
    assert second.cache_dir != first.cache_dir
    assert second.manifest["tokenizer_sha256"] != first.manifest["tokenizer_sha256"]
    assert second.vocab_size == first.vocab_size + 1


def test_missing_separator_is_rejected(tmp_path: Path) -> None:
    dataset_dir = make_dataset(tmp_path)
    tokenizer_path, _ = make_tokenizer(dataset_dir, tmp_path)
    with pytest.raises(ValueError, match="no separator"):
        build_token_streams(dataset_dir, tokenizer_path, document_separator="<|missing|>")


# -------------------------------------------------------------------- sampling
def test_sampler_shapes_and_shifted_targets() -> None:
    tokens = np.arange(1000, dtype=np.int32)
    sampler = PackedSampler(tokens, context_length=8, batch_size=3, seed=5)
    batch = sampler.batch(0, 0)
    assert batch.inputs.shape == (3, 8)
    assert batch.targets.shape == (3, 8)
    assert torch.equal(batch.targets, batch.inputs + 1)
    assert int(batch.inputs.min()) % 8 == 0  # non-overlapping windows


def test_sampler_epoch_is_deterministic_and_shuffled() -> None:
    tokens = np.arange(1000, dtype=np.int32)
    a = PackedSampler(tokens, context_length=8, batch_size=3, seed=11)
    b = PackedSampler(tokens, context_length=8, batch_size=3, seed=11)
    assert torch.equal(a.batch(0, 2).inputs, b.batch(0, 2).inputs)
    epoch_zero = torch.cat([a.batch(0, i).inputs for i in range(a.steps_per_epoch)])
    epoch_one = torch.cat([a.batch(1, i).inputs for i in range(a.steps_per_epoch)])
    assert not torch.equal(epoch_zero, epoch_one)


def test_sampler_covers_each_window_once_per_epoch() -> None:
    tokens = np.arange(1000, dtype=np.int32)
    sampler = PackedSampler(tokens, context_length=8, batch_size=3, seed=0)
    starts = torch.cat([sampler.batch(0, i).inputs[:, 0] for i in range(sampler.steps_per_epoch)])
    assert len(torch.unique(starts)) == starts.numel()
    assert starts.numel() == sampler.steps_per_epoch * 3
    assert int(starts.max()) + 8 < tokens.size


def test_sampler_rejects_streams_that_cannot_fill_a_batch() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        PackedSampler(np.arange(10, dtype=np.int32), context_length=8, batch_size=2)
    sampler = PackedSampler(np.arange(200, dtype=np.int32), context_length=8, batch_size=2)
    with pytest.raises(IndexError):
        sampler.batch(0, sampler.steps_per_epoch)
    with pytest.raises(ValueError):
        sampler.batch(-1, 0)


def test_cursor_wraps_and_advances() -> None:
    tokens = np.arange(400, dtype=np.int32)
    sampler = PackedSampler(tokens, context_length=8, batch_size=2, seed=1)
    cursor = BatchCursor()
    seen: list[int] = []
    for _ in range(sampler.steps_per_epoch + 1):
        batch, cursor = sampler.advance(cursor)
        seen.append(batch.inputs.size(0))
    assert cursor.epoch == 1
    assert cursor.batch_index == 1
    assert all(size == 2 for size in seen)


def test_validation_batches_are_fixed_and_spread() -> None:
    tokens = np.arange(4000, dtype=np.int32)
    batches = evaluation_batches(tokens, context_length=8, batch_size=3, max_batches=4)
    assert len(batches) == 4
    again = evaluation_batches(tokens, context_length=8, batch_size=3, max_batches=4)
    for first, second in zip(batches, again, strict=True):
        assert torch.equal(first.inputs, second.inputs)
    starts = torch.cat([batch.inputs[:, 0] for batch in batches])
    assert int(starts.max()) > int(tokens.size * 0.8)  # not just the opening windows
    with pytest.raises(ValueError):
        evaluation_batches(tokens, context_length=8, batch_size=3, max_batches=0)


# ------------------------------------------------------------------- schedule
def test_schedule_warmup_peak_and_decay() -> None:
    schedule = LRSchedule(max_steps=100, base_lr=1.0, warmup_steps=10, min_lr_ratio=0.1)
    assert schedule.lr_at(0) == pytest.approx(0.1)
    assert schedule.lr_at(9) == pytest.approx(1.0)
    assert schedule.lr_at(10) == pytest.approx(1.0)
    assert schedule.lr_at(99) == pytest.approx(0.1, abs=5e-4)
    after = [schedule.lr_at(step) for step in range(10, 100)]
    assert all(a >= b for a, b in zip(after, after[1:], strict=False))
    assert schedule.lr_at(100) == pytest.approx(0.1)


def test_schedule_kinds_and_validation() -> None:
    linear = LRSchedule(max_steps=50, base_lr=2e-3, warmup_steps=5, min_lr_ratio=0.2, kind="linear")
    assert linear.lr_at(5) == pytest.approx(2e-3)
    assert linear.lr_at(49) < 2e-3
    assert linear.lr_at(50) == pytest.approx(2e-3 * 0.2)

    constant = LRSchedule(max_steps=50, base_lr=3e-3, warmup_steps=0, kind="constant")
    assert constant.lr_at(0) == pytest.approx(3e-3)
    assert constant.lr_at(49) == pytest.approx(3e-3)

    with pytest.raises(ValueError):
        LRSchedule(max_steps=10, base_lr=1.0, warmup_steps=10)
    with pytest.raises(ValueError):
        LRSchedule(max_steps=10, base_lr=1.0, kind="triangle")
    with pytest.raises(ValueError):
        constant.lr_at(-1)


# ------------------------------------------------------------------ checkpoints
def _trained_tiny_gpt(seed: int = 7) -> tuple[GPT, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    model = GPT(SMALL_MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    idx = torch.randint(0, SMALL_MODEL.vocab_size, (2, SMALL_MODEL.context_length))
    _, loss = model(idx, idx)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_checkpoint_roundtrip_restores_model_optimizer_and_rng(tmp_path: Path) -> None:
    model, optimizer = _trained_tiny_gpt()
    config = base_config(tmp_path, fake_streams(tmp_path / "s"))

    torch.manual_seed(123)
    path = save_checkpoint(
        tmp_path / "run",
        step=4,
        tokens_seen=512,
        best_val_loss=1.5,
        cursor=BatchCursor(epoch=2, batch_index=3),
        model=model,
        optimizer=optimizer,
        config=config,
        history=[{"step": 4}],
    )
    assert path.is_file()
    assert (tmp_path / "run" / "latest.json").is_file()
    assert resolve_checkpoint(tmp_path / "run") == path

    expected_next = torch.randn(2)  # first draw after the saved RNG state
    restored = GPT(SMALL_MODEL)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-2)
    state = load_checkpoint(path, model=restored, optimizer=restored_opt)

    assert state["step"] == 4
    assert state["tokens_seen"] == 512
    assert state["best_val_loss"] == 1.5
    assert state["cursor"] == BatchCursor(epoch=2, batch_index=3)
    for (name_a, param_a), (name_b, param_b) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)
    actual_next = torch.randn(2)
    assert torch.equal(expected_next, actual_next)


def test_latest_pointer_resolves_paths_relative_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run dir under the CWD must not resolve to ``run/run/step-...pt``."""
    model, optimizer = _trained_tiny_gpt()
    config = base_config(tmp_path, fake_streams(tmp_path / "s"))
    monkeypatch.chdir(tmp_path)
    path = save_checkpoint(
        "run",
        step=2,
        tokens_seen=16,
        best_val_loss=None,
        cursor=BatchCursor(),
        model=model,
        optimizer=optimizer,
        config=config,
        history=[],
    )
    assert resolve_checkpoint("run") == path
    monkeypatch.chdir(tmp_path.parent)
    assert resolve_checkpoint(tmp_path / "run").name == "step-000002.pt"


def test_latest_pointer_falls_back_to_the_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "moved"
    run_dir.mkdir()
    target = run_dir / "step-000007.pt"
    target.write_bytes(b"placeholder")
    (run_dir / "latest.json").write_text(
        json.dumps({"checkpoint": "step-000007.pt", "step": 7}), encoding="utf-8"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert resolve_checkpoint(run_dir) == target

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(elsewhere)


def test_checkpoint_rejects_mismatched_model_config(tmp_path: Path) -> None:
    model, optimizer = _trained_tiny_gpt()
    config = base_config(tmp_path, fake_streams(tmp_path / "s"))
    path = save_checkpoint(
        tmp_path / "run",
        step=1,
        tokens_seen=32,
        best_val_loss=None,
        cursor=BatchCursor(),
        model=model,
        optimizer=optimizer,
        config=config,
        history=[],
    )
    other = GPT(replace(SMALL_MODEL, vocab_size=32))
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, model=other)


def test_checkpoint_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"version": 999, "step": 0}, path)
    with pytest.raises(ValueError, match="unsupported"):
        load_checkpoint(path, model=GPT(SMALL_MODEL))


# --------------------------------------------------------------------- safety
def test_memory_budget_guard() -> None:
    ensure_memory_budget(500, 1000, max_fraction=0.7)
    with pytest.raises(InsufficientMemoryError, match="budget"):
        ensure_memory_budget(900, 1000, max_fraction=0.7)
    with pytest.raises(ValueError):
        ensure_memory_budget(10, 100, max_fraction=0.0)


def test_available_memory_is_a_positive_number_on_this_machine() -> None:
    available = available_memory_bytes()
    assert available is None or available > 0


def test_oom_detection() -> None:
    assert is_oom_error(RuntimeError("DefaultCPUAllocator: can't allocate memory"))
    assert is_oom_error(RuntimeError("CUDA out of memory"))
    assert not is_oom_error(RuntimeError("size mismatch"))


# -------------------------------------------------------------------- training
def test_config_validation_and_json_roundtrip(tmp_path: Path) -> None:
    streams = fake_streams(tmp_path / "s")
    config = base_config(tmp_path, streams)
    assert config.tokens_per_step == 4 * 1 * 8
    assert TrainingConfig.from_dict(config.to_dict()) == config
    path = config.save(tmp_path / "cfg.json")
    assert TrainingConfig.load(path) == config

    with pytest.raises(ValueError):
        replace(config, lr=0.0)
    with pytest.raises(ValueError):
        replace(config, warmup_steps=config.max_steps)
    with pytest.raises(ValueError):
        replace(config, schedule="triangle")
    with pytest.raises(ValueError):
        replace(config, device="cuda")
    with pytest.raises(ValueError, match="unknown"):
        TrainingConfig.from_dict({**config.to_dict(), "unknown": 1})


def test_trainer_reduces_loss_on_learnable_data(tmp_path: Path) -> None:
    streams = fake_streams(tmp_path / "s")
    config = base_config(
        tmp_path,
        streams,
        max_steps=60,
        lr=3e-3,
        eval_interval=20,
        checkpoint_interval=30,
        log_interval=20,
        seed=1,
    )
    trainer = Trainer(config, streams=streams, verbose=False)
    assert trainer.streams.vocab_size == config.model.vocab_size
    first = trainer.evaluate()
    report = trainer.fit()

    assert report["step"] == 60
    assert report["tokens_seen"] == 60 * config.tokens_per_step
    assert report["final_val_loss"] is not None
    assert report["final_val_loss"] < first
    assert report["final_train_loss"] < 3.0
    assert report["first_step_loss"] is not None
    assert report["first_step_loss"] > report["final_train_loss"]
    assert trainer.log_path.is_file()

    records = [
        json.loads(line) for line in trainer.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records and {"step", "train_loss", "val_loss", "lr"} <= set(records[0])
    assert any(record["val_loss"] is not None for record in records)
    assert (tmp_path / "run" / "latest.json").is_file()
    assert "parameters" in report and report["parameters"]["total"] > 0


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    streams = fake_streams(tmp_path / "s")
    config = base_config(tmp_path, streams, max_steps=6, checkpoint_interval=3)

    uninterrupted = Trainer(
        replace(config, out_dir=str(tmp_path / "a")), streams=streams, verbose=False
    )
    uninterrupted.fit()

    interrupted = Trainer(
        replace(config, out_dir=str(tmp_path / "b")), streams=streams, verbose=False
    )
    interrupted.fit(max_steps=3)
    assert interrupted.step == 3
    resumed = Trainer(replace(config, out_dir=str(tmp_path / "b")), streams=streams, verbose=False)
    report = resumed.fit(max_steps=6, resume_from=tmp_path / "b")

    assert report["step"] == 6
    assert report["tokens_seen"] == uninterrupted.tokens_seen
    assert report["final_train_loss"] == pytest.approx(uninterrupted.report()["final_train_loss"])
    for (name_a, param_a), (name_b, param_b) in zip(
        uninterrupted.model.state_dict().items(),
        resumed.model.state_dict().items(),
        strict=True,
    ):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, atol=1e-7)


def test_resume_keeps_checkpoint_schedule_horizon(tmp_path: Path) -> None:
    """Extending a run must not move the learning rate to a later point on the curve."""
    streams = fake_streams(tmp_path / "s")
    original = base_config(tmp_path, streams, max_steps=6, checkpoint_interval=3)
    trainer = Trainer(
        replace(original, out_dir=str(tmp_path / "a")), streams=streams, verbose=False
    )
    trainer.fit(max_steps=3)

    extended = base_config(tmp_path, streams, max_steps=10, checkpoint_interval=3)
    resumed = Trainer(
        replace(extended, out_dir=str(tmp_path / "a")), streams=streams, verbose=False
    )
    resumed.resume(tmp_path / "a")

    assert resumed.schedule.max_steps == original.max_steps
    assert resumed.schedule.lr_at(3) == pytest.approx(
        LRSchedule(
            max_steps=original.max_steps,
            base_lr=original.lr,
            warmup_steps=original.warmup_steps,
            min_lr_ratio=original.min_lr_ratio,
            kind=original.schedule,
        ).lr_at(3)
    )


def test_gradient_accumulation_matches_full_batch(tmp_path: Path) -> None:
    streams = fake_streams(tmp_path / "s")
    config = base_config(tmp_path, streams, batch_size=4)
    single = Trainer(replace(config, out_dir=str(tmp_path / "a")), streams=streams, verbose=False)
    accum = Trainer(replace(config, out_dir=str(tmp_path / "b")), streams=streams, verbose=False)

    batch = single.train_sampler.batch(0, 0)
    half = batch.inputs.size(0) // 2
    losses = Batch(inputs=batch.inputs[:half], targets=batch.targets[:half])
    other = Batch(inputs=batch.inputs[half:], targets=batch.targets[half:])

    loss_single, _, _ = single.train_step([batch])
    loss_accum, _, _ = accum.train_step([losses, other])
    assert loss_single == pytest.approx(loss_accum, rel=1e-5, abs=1e-6)
    for (name_a, param_a), (name_b, param_b) in zip(
        single.model.state_dict().items(), accum.model.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, atol=1e-6, rtol=1e-5)


def test_oom_during_step_is_translated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    streams = fake_streams(tmp_path / "s")
    trainer = Trainer(base_config(tmp_path, streams), streams=streams, verbose=False)
    batch = trainer.train_sampler.batch(0, 0)

    def boom(*args, **kwargs):
        raise RuntimeError("DefaultCPUAllocator: can't allocate memory")

    monkeypatch.setattr(type(trainer.model), "forward", boom)
    with pytest.raises(TrainingOOMError, match="out of memory"):
        trainer.train_step([batch])


def test_memory_estimate_uses_real_parameter_count(tmp_path: Path) -> None:
    streams = fake_streams(tmp_path / "s")
    trainer = Trainer(base_config(tmp_path, streams), streams=streams, verbose=False)
    estimate = trainer.memory_estimate()
    assert estimate["parameters"] == sum(p.numel() for p in trainer.model.parameters())
    assert estimate["batch_size"] == trainer.config.batch_size


def test_seed_everything_makes_runs_identical(tmp_path: Path) -> None:
    streams = fake_streams(tmp_path / "s")
    config = base_config(tmp_path, streams, max_steps=4)
    a = Trainer(replace(config, out_dir=str(tmp_path / "a")), streams=streams, verbose=False)
    b = Trainer(replace(config, out_dir=str(tmp_path / "b")), streams=streams, verbose=False)
    a.fit()
    b.fit()
    assert a.report()["final_train_loss"] == b.report()["final_train_loss"]
    for (name_a, param_a), (name_b, param_b) in zip(
        a.model.state_dict().items(), b.model.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


def test_random_module_state_is_restored_by_checkpoint(tmp_path: Path) -> None:
    streams = fake_streams(tmp_path / "s")
    config = base_config(tmp_path, streams)
    model, optimizer = _trained_tiny_gpt()
    random.seed(5)
    torch.manual_seed(5)
    np.random.seed(5)
    path = save_checkpoint(
        tmp_path / "run",
        step=0,
        tokens_seen=0,
        best_val_loss=None,
        cursor=BatchCursor(),
        model=model,
        optimizer=optimizer,
        config=config,
        history=[],
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    random.seed(999)
    np.random.seed(999)
    load_checkpoint(path, model=model, optimizer=optimizer)
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
