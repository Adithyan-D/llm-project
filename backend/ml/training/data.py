"""Token streams and packed batching for pretraining.

Pipeline::

    processed dataset (JSONL)  +  BPE tokenizer
        -> per-split int32 token stream (documents separated by EOS)
        -> non-overlapping windows of context_length + 1 tokens
        -> shuffled batches of (inputs, targets) where targets are inputs + 1

Packing (concatenating documents into one stream and filling every window
completely) means no padding tokens are ever trained on. The cost is that a
window may cross a document boundary; the EOS separator makes that boundary
visible to the model, which is standard GPT-style pretraining.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ..data import iter_split, read_meta
from ..tokenization import BPETokenizer

TOKEN_STREAM_VERSION = 1
TOKEN_DTYPE = np.int32
DEFAULT_SEPARATOR = "<|endoftext|>"


# ------------------------------------------------------------------ fingerprints
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------- streams
@dataclass(frozen=True)
class SplitStream:
    name: str
    path: Path
    tokens: int
    documents: int
    chars: int
    sha256: str


@dataclass(frozen=True)
class TokenStreams:
    cache_dir: Path
    manifest_path: Path
    vocab_size: int
    eos_token_id: int | None
    splits: dict[str, SplitStream]
    manifest: dict[str, object]

    def load(self, split: str) -> np.ndarray:
        """Load a split as a 1D int32 array (read-only)."""
        try:
            stream = self.splits[split]
        except KeyError:
            available = sorted(self.splits)
            raise KeyError(f"split {split!r} not in stream; available: {available}") from None
        array = np.load(stream.path)
        if array.ndim != 1 or array.dtype != TOKEN_DTYPE:
            raise ValueError(
                f"{stream.path}: expected 1D {TOKEN_DTYPE}, got {array.dtype}/{array.shape}"
            )
        return array


def _cache_dir(dataset_dir: Path, tokenizer_sha: str, eos_token_id: int | None) -> Path:
    suffix = f"eos{eos_token_id}" if eos_token_id is not None else "noeos"
    return dataset_dir / f"tokens-{tokenizer_sha[:12]}-{suffix}"


def _manifest_matches(
    manifest: dict[str, object],
    *,
    dataset_hash: str,
    tokenizer_sha: str,
    eos_token_id: int | None,
    splits: tuple[str, ...],
    cache_dir: Path,
) -> bool:
    if manifest.get("version") != TOKEN_STREAM_VERSION:
        return False
    if manifest.get("dataset_hash") != dataset_hash:
        return False
    if manifest.get("tokenizer_sha256") != tokenizer_sha:
        return False
    if manifest.get("eos_token_id") != eos_token_id:
        return False
    recorded = manifest.get("splits")
    if not isinstance(recorded, dict) or set(recorded) != set(splits):
        return False
    for name in splits:
        entry = recorded.get(name)
        if not isinstance(entry, dict):
            return False
        path = cache_dir / str(entry.get("file"))
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if entry.get("sha256") != sha256_file(path):
            return False
    return True


def build_token_streams(
    dataset_dir: str | Path,
    tokenizer_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "val"),
    document_separator: str | None = DEFAULT_SEPARATOR,
    force: bool = False,
) -> TokenStreams:
    """Encode every split once and cache the token ids next to the dataset.

    The cache key is the dataset content hash plus the tokenizer file hash, so
    re-running the same preparation is a pure file check. Passing a different
    tokenizer rebuilds the cache instead of silently mixing vocabularies.
    """
    dataset_path = Path(dataset_dir)
    tokenizer_file = Path(tokenizer_path)
    meta = read_meta(dataset_path)
    dataset_hash = str(meta["dataset_hash"])
    tokenizer_sha = sha256_file(
        tokenizer_file / "tokenizer.json" if tokenizer_file.is_dir() else tokenizer_file
    )
    tokenizer = BPETokenizer.load(tokenizer_file)

    if document_separator is None:
        eos_token_id = None
    else:
        eos_token_id = tokenizer.special_tokens.get(document_separator)
        if eos_token_id is None:
            available = sorted(tokenizer.special_tokens)
            raise ValueError(
                f"tokenizer has no separator {document_separator!r}; available specials: {available}"
            )

    target_dir = (
        Path(cache_dir)
        if cache_dir is not None
        else _cache_dir(dataset_path, tokenizer_sha, eos_token_id)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.json"

    if not force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_matches(
            manifest,
            dataset_hash=dataset_hash,
            tokenizer_sha=tokenizer_sha,
            eos_token_id=eos_token_id,
            splits=splits,
            cache_dir=target_dir,
        ):
            return TokenStreams(
                cache_dir=target_dir,
                manifest_path=manifest_path,
                vocab_size=int(manifest["vocab_size"]),
                eos_token_id=eos_token_id,
                splits=_splits_from_manifest(manifest, target_dir),
                manifest=manifest,
            )

    recorded: dict[str, dict[str, object]] = {}
    for split in splits:
        ids: list[int] = []
        documents = 0
        chars = 0
        for text in iter_split(dataset_path, split):
            ids.extend(tokenizer.encode(text))
            if eos_token_id is not None:
                ids.append(eos_token_id)
            documents += 1
            chars += len(text)
        if len(ids) < 2:
            raise ValueError(f"split {split!r} produced {len(ids)} tokens; need at least 2")
        array = np.asarray(ids, dtype=TOKEN_DTYPE)
        filename = f"{split}.npy"
        final_path = target_dir / filename
        tmp_path = target_dir / f".{split}.tmp.npy"
        np.save(tmp_path, array)
        os.replace(tmp_path, final_path)
        recorded[split] = {
            "file": filename,
            "tokens": int(array.size),
            "documents": documents,
            "chars": chars,
            "sha256": sha256_file(final_path),
        }

    manifest = {
        "version": TOKEN_STREAM_VERSION,
        "dataset_id": meta["dataset_id"],
        "dataset_hash": dataset_hash,
        "tokenizer_path": str(tokenizer_file),
        "tokenizer_sha256": tokenizer_sha,
        "vocab_size": len(tokenizer),
        "eos_token": document_separator,
        "eos_token_id": eos_token_id,
        "splits": recorded,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    tmp_manifest = target_dir / "manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)

    return TokenStreams(
        cache_dir=target_dir,
        manifest_path=manifest_path,
        vocab_size=len(tokenizer),
        eos_token_id=eos_token_id,
        splits=_splits_from_manifest(manifest, target_dir),
        manifest=manifest,
    )


def _splits_from_manifest(manifest: dict[str, object], cache_dir: Path) -> dict[str, SplitStream]:
    entries = manifest.get("splits")
    if not isinstance(entries, dict):
        raise ValueError(f"{cache_dir}: manifest has no splits table")
    splits: dict[str, SplitStream] = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{cache_dir}: bad split entry for {name!r}")
        splits[name] = SplitStream(
            name=name,
            path=cache_dir / str(entry["file"]),
            tokens=int(entry["tokens"]),
            documents=int(entry["documents"]),
            chars=int(entry["chars"]),
            sha256=str(entry["sha256"]),
        )
    return splits


# ---------------------------------------------------------------------- batches
@dataclass(frozen=True)
class Batch:
    """One (inputs, targets) pair; targets are inputs shifted by one token."""

    inputs: torch.Tensor
    targets: torch.Tensor

    def __len__(self) -> int:
        return int(self.inputs.size(0))


@dataclass(frozen=True)
class BatchCursor:
    """Position in the token stream, serialised into checkpoints.

    Two integers are enough because the shuffled window order is regenerated
    deterministically from ``(seed, epoch)``; no permutation needs to be stored.
    """

    epoch: int = 0
    batch_index: int = 0


def _batch_from_starts(tokens: np.ndarray, starts: np.ndarray, context_length: int) -> Batch:
    inputs = np.stack([tokens[s : s + context_length] for s in starts])
    targets = np.stack([tokens[s + 1 : s + 1 + context_length] for s in starts])
    return Batch(
        inputs=torch.from_numpy(inputs.astype(np.int64, copy=False)),
        targets=torch.from_numpy(targets.astype(np.int64, copy=False)),
    )


class PackedSampler:
    """Deterministic epoch sampler over non-overlapping windows.

    ``steps_per_epoch`` counts *micro-batches*, not optimizer steps: with
    gradient accumulation the caller consumes several of them per update.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        *,
        context_length: int,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if seed < 0:
            raise ValueError("seed must be >= 0")

        self.tokens = tokens
        self.context_length = context_length
        self.batch_size = batch_size
        self.seed = seed
        self.window_starts = np.arange(
            0, max(0, tokens.size - context_length), context_length, dtype=np.int64
        )
        self.steps_per_epoch = int(self.window_starts.size // batch_size)
        if self.steps_per_epoch < 1:
            raise ValueError(
                f"stream of {tokens.size} tokens cannot fill one batch of "
                f"{batch_size} x {context_length}; need at least "
                f"{batch_size * (context_length + 1)} tokens"
            )

    def _shuffled_windows(self, epoch: int) -> np.ndarray:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(self.window_starts.size)
        return self.window_starts[order]

    def batch(self, epoch: int, batch_index: int) -> Batch:
        if not 0 <= batch_index < self.steps_per_epoch:
            raise IndexError(f"batch_index={batch_index} outside [0, {self.steps_per_epoch})")
        windows = self._shuffled_windows(epoch)
        start = batch_index * self.batch_size
        starts = windows[start : start + self.batch_size]
        return _batch_from_starts(self.tokens, starts, self.context_length)

    def epoch_batches(self, epoch: int) -> Iterator[Batch]:
        for index in range(self.steps_per_epoch):
            yield self.batch(epoch, index)

    def advance(self, cursor: BatchCursor) -> tuple[Batch, BatchCursor]:
        """Return the next batch and the cursor after it (epoch wraps automatically)."""
        if cursor.batch_index >= self.steps_per_epoch:
            cursor = BatchCursor(epoch=cursor.epoch + 1, batch_index=0)
        batch = self.batch(cursor.epoch, cursor.batch_index)
        return batch, BatchCursor(epoch=cursor.epoch, batch_index=cursor.batch_index + 1)


def evaluation_batches(
    tokens: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    max_batches: int,
) -> list[Batch]:
    """Fixed, evenly spaced validation batches.

    Using the same windows at every evaluation keeps val loss comparable across
    steps. Windows are spread across the split instead of taking the first N, so
    the number reflects the whole validation set rather than its opening lines.
    """
    if max_batches < 1:
        raise ValueError("max_batches must be >= 1")
    windows = np.arange(0, max(0, tokens.size - context_length), context_length, dtype=np.int64)
    if windows.size < batch_size:
        raise ValueError(
            f"validation stream of {tokens.size} tokens has {windows.size} windows; "
            f"need at least {batch_size}"
        )
    wanted = min(max_batches * batch_size, int(windows.size))
    positions = np.unique(np.linspace(0, windows.size - 1, wanted, dtype=np.int64))
    selected = windows[positions]
    usable = (selected.size // batch_size) * batch_size
    if usable < batch_size:
        raise ValueError("not enough validation windows for a single batch")
    return [
        _batch_from_starts(tokens, selected[i : i + batch_size], context_length)
        for i in range(0, usable, batch_size)
    ]


__all__ = [
    "DEFAULT_SEPARATOR",
    "TOKEN_DTYPE",
    "TOKEN_STREAM_VERSION",
    "Batch",
    "BatchCursor",
    "PackedSampler",
    "SplitStream",
    "TokenStreams",
    "build_token_streams",
    "evaluation_batches",
    "sha256_file",
]
