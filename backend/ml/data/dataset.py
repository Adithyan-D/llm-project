"""Dataset processing pipeline: load -> validate -> normalise -> dedup -> split.

Every run produces a self-describing directory:

    datasets/processed/<dataset_id>/
        train.jsonl  val.jsonl  test.jsonl   # {"text": "..."} per line
        quarantine.jsonl                     # rejected records + reason (capped)
        meta.json                            # config, sources, stats, hashes

Split assignment is a pure function of ``(seed, text)``, so it is reproducible
without shuffling and works in streaming mode. Exact proportions are therefore
approximate (within sampling noise), which is documented in meta.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .loaders import LoadIssue, LoadStats, iter_sources
from .preprocess import (
    Deduplicator,
    NormalizationConfig,
    normalize_text,
    text_fingerprint,
    validate_text,
)
from .stats import SplitStats, merge_stats

SPLITS = ("train", "val", "test")


@dataclass
class DatasetConfig:
    name: str
    inputs: list[str]
    output_dir: str = "datasets/processed"
    text_field: str = "text"
    txt_mode: str = "line"
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    deduplicate: bool = True
    max_records: int | None = None
    tokenizer_path: str | None = None
    quarantine_limit: int = 100

    def __post_init__(self) -> None:
        if not 0 <= self.test_fraction < 1:
            raise ValueError("test_fraction must be in [0, 1)")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")
        if not self.inputs:
            raise ValueError("at least one input file is required")


@dataclass
class ProcessingResult:
    dataset_id: str
    output_dir: str
    dataset_hash: str
    stats: dict[str, object]
    sources: list[dict[str, object]]
    load_stats: dict[str, object]
    rejected: dict[str, int]
    duplicate_count: int
    tokenizer: str | None
    duration_seconds: float
    created_at: str
    meta_path: str


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assign_split(digest: bytes, seed: int, val_fraction: float, test_fraction: float) -> str:
    seeded = hashlib.blake2b(digest + seed.to_bytes(8, "little"), digest_size=16).digest()
    position = int.from_bytes(seeded[:8], "big") / 2**64
    if position < test_fraction:
        return "test"
    if position < test_fraction + val_fraction:
        return "val"
    return "train"


def _load_tokenizer(path: str | None):
    if path is None:
        return None
    from ml.tokenization.tokenizer import BPETokenizer

    return BPETokenizer.load(path)


def process_dataset(cfg: DatasetConfig) -> ProcessingResult:
    """Run the full pipeline and write the processed dataset plus metadata."""
    started = time.perf_counter()
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

    inputs = [Path(p) for p in cfg.inputs]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    sources = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in inputs
    ]

    tokenizer = _load_tokenizer(cfg.tokenizer_path)
    load_stats = LoadStats()
    issues: list[LoadIssue] = []
    rejected: dict[str, int] = {}
    dedup = Deduplicator()
    stats = {split: SplitStats() for split in SPLITS}
    fingerprints = hashlib.blake2b(digest_size=32)

    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f"_staging_{cfg.name}_{int(started)}"
    staging.mkdir(parents=True, exist_ok=True)

    handles = {split: (staging / f"{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    quarantine_handle = (staging / "quarantine.jsonl").open("w", encoding="utf-8")
    records_kept = 0
    try:
        for source, raw_text, location in iter_sources(
            inputs,
            text_field=cfg.text_field,
            txt_mode=cfg.txt_mode,
            issues=issues,
            stats=load_stats,
        ):
            if cfg.max_records is not None and records_kept >= cfg.max_records:
                break

            text = normalize_text(raw_text, cfg.normalization)
            reason = validate_text(text, cfg.normalization)
            if reason is None and cfg.deduplicate and dedup.is_duplicate(text):
                reason = "duplicate"

            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                if rejected[reason] <= cfg.quarantine_limit:
                    quarantine_handle.write(
                        json.dumps(
                            {
                                "source": source,
                                "location": location,
                                "reason": reason,
                                "text": raw_text[:2000],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            digest = text_fingerprint(text)
            fingerprints.update(digest)
            split = _assign_split(digest, cfg.seed, cfg.val_fraction, cfg.test_fraction)
            handles[split].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            stats[split].add(text)
            if tokenizer is not None:
                stats[split].add_tokens(len(tokenizer.encode(text)))
            records_kept += 1
    finally:
        for handle in handles.values():
            handle.close()
        quarantine_handle.close()

    if issues:
        rejected["loader_errors"] = len(issues)
        with (staging / "quarantine.jsonl").open("a", encoding="utf-8") as handle:
            for issue in issues[: cfg.quarantine_limit]:
                handle.write(
                    json.dumps(
                        {
                            "source": issue.source,
                            "location": issue.line,
                            "reason": f"loader:{issue.reason}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if records_kept == 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(
            f"no usable records produced (rejected={rejected}, loader_issues={len(issues)})"
        )

    dataset_hash = fingerprints.hexdigest()
    dataset_id = f"{cfg.name}-{dataset_hash[:12]}"
    final_dir = root / dataset_id
    if final_dir.exists():
        # Re-running the same corpus with different config (e.g. adding a
        # tokenizer for exact token counts) is idempotent: identical content
        # hash means the split files are reproducible, so replace them.
        existing_meta = final_dir / "meta.json"
        existing_hash = None
        if existing_meta.is_file():
            try:
                existing_hash = json.loads(existing_meta.read_text(encoding="utf-8"))[
                    "dataset_hash"
                ]
            except (json.JSONDecodeError, KeyError):
                existing_hash = None
        if existing_hash == dataset_hash:
            shutil.rmtree(final_dir)
        else:
            final_dir = root / f"{dataset_id}-{int(started)}"
    staging.rename(final_dir)

    for split in SPLITS:
        stats[split].bytes_on_disk = (final_dir / f"{split}.jsonl").stat().st_size

    summary = merge_stats(stats)
    duration = time.perf_counter() - started
    meta = {
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "name": cfg.name,
        "created_at": created_at,
        "duration_seconds": round(duration, 3),
        "config": asdict(cfg),
        "sources": sources,
        "load": load_stats.as_dict(),
        "rejected": rejected,
        "duplicate_count": dedup.duplicates,
        "stats": summary,
        "tokenizer": cfg.tokenizer_path,
        "token_count_method": "exact_bpe" if tokenizer is not None else "not_computed",
        "files": {split: f"{split}.jsonl" for split in SPLITS},
    }
    meta_path = final_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return ProcessingResult(
        dataset_id=dataset_id,
        output_dir=str(final_dir),
        dataset_hash=dataset_hash,
        stats=summary,
        sources=sources,
        load_stats=load_stats.as_dict(),
        rejected=rejected,
        duplicate_count=dedup.duplicates,
        tokenizer=cfg.tokenizer_path,
        duration_seconds=round(duration, 3),
        created_at=created_at,
        meta_path=str(meta_path),
    )


def read_meta(dataset_dir: str | Path) -> dict[str, object]:
    return json.loads((Path(dataset_dir) / "meta.json").read_text(encoding="utf-8"))


def iter_split(dataset_dir: str | Path, split: str) -> Iterator[str]:
    """Yield texts for one split without loading the file into memory."""
    path = Path(dataset_dir) / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]
