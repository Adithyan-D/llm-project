"""Streaming dataset loaders for TXT, JSON and JSONL sources.

Design notes
------------
* Loaders are generators: a 1 GB JSONL file is never materialised in RAM.
* JSON arrays must be parsed whole (stdlib limitation); JSONL is the recommended
  format for large corpora and is documented as such.
* Loaders only reject *structural* problems (bad JSON, wrong types, missing
  fields). Text quality (length, control characters, duplicates) belongs to
  ``preprocess.py`` so the two concerns stay testable in isolation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = (".txt", ".json", ".jsonl", ".ndjson")
TEXT_FIELDS_FALLBACK = ("text", "content", "document", "body")


class DatasetFormatError(ValueError):
    """Raised when a source file cannot be interpreted at all."""


@dataclass(frozen=True)
class LoadIssue:
    source: str
    line: int | None
    reason: str


@dataclass
class LoadStats:
    files: int = 0
    records_seen: int = 0
    records_loaded: int = 0
    issues: list[LoadIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, int | list[dict[str, str | int | None]]]:
        return {
            "files": self.files,
            "records_seen": self.records_seen,
            "records_loaded": self.records_loaded,
            "issue_count": len(self.issues),
            "issues": [
                {"source": i.source, "line": i.line, "reason": i.reason} for i in self.issues
            ],
        }


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "txt"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".json":
        return "json"
    raise DatasetFormatError(
        f"unsupported file type {path.suffix!r}; expected one of {SUPPORTED_SUFFIXES}"
    )


def _extract_text(record: object, source: str, line: int | None, text_field: str) -> str | None:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for candidate in (text_field, *TEXT_FIELDS_FALLBACK):
            value = record.get(candidate)
            if isinstance(value, str):
                return value
        return None
    return None


def _iter_txt(
    path: Path, txt_mode: str, issues: list[LoadIssue] | None
) -> Iterator[tuple[str, int | None]]:
    if txt_mode == "document":
        text = path.read_text(encoding="utf-8", errors="replace")
        yield text, None
        return
    if txt_mode != "line":
        raise ValueError(f"txt_mode must be 'line' or 'document', got {txt_mode!r}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            yield stripped, line_no


def _iter_jsonl(
    path: Path, text_field: str, issues: list[LoadIssue] | None
) -> Iterator[tuple[str, int | None]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if issues is not None:
                    issues.append(LoadIssue(str(path), line_no, f"invalid_json: {exc.msg}"))
                continue
            text = _extract_text(record, str(path), line_no, text_field)
            if text is None:
                if issues is not None:
                    issues.append(
                        LoadIssue(str(path), line_no, f"missing_text_field: {text_field!r}")
                    )
                continue
            yield text, line_no


def _iter_json(
    path: Path, text_field: str, issues: list[LoadIssue] | None
) -> Iterator[tuple[str, int | None]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc

    if isinstance(payload, list):
        records: Iterable[object] = payload
    elif isinstance(payload, dict):
        for key in ("data", "records", "documents"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            records = [payload]
    else:
        raise DatasetFormatError(f"{path}: expected a JSON array or object, got {type(payload)}")

    for index, record in enumerate(records):
        text = _extract_text(record, str(path), index, text_field)
        if text is None:
            if issues is not None:
                issues.append(LoadIssue(str(path), index, f"missing_text_field: {text_field!r}"))
            continue
        yield text, index


def iter_text_records(
    path: str | Path,
    *,
    text_field: str = "text",
    txt_mode: str = "line",
    issues: list[LoadIssue] | None = None,
) -> Iterator[tuple[str, int | None]]:
    """Yield ``(text, location)`` for one source file, streaming where possible."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    fmt = detect_format(path)
    if fmt == "txt":
        yield from _iter_txt(path, txt_mode, issues)
    elif fmt == "jsonl":
        yield from _iter_jsonl(path, text_field, issues)
    else:
        yield from _iter_json(path, text_field, issues)


def iter_sources(
    paths: Iterable[str | Path],
    *,
    text_field: str = "text",
    txt_mode: str = "line",
    issues: list[LoadIssue] | None = None,
    stats: LoadStats | None = None,
) -> Iterator[tuple[str, str, int | None]]:
    """Yield ``(source, text, location)`` across multiple files."""
    for raw_path in paths:
        path = Path(raw_path)
        if stats is not None:
            stats.files += 1
        for text, location in iter_text_records(
            path, text_field=text_field, txt_mode=txt_mode, issues=issues
        ):
            if stats is not None:
                stats.records_seen += 1
                stats.records_loaded += 1
            yield str(path), text, location


def load_texts(
    paths: Iterable[str | Path],
    *,
    text_field: str = "text",
    txt_mode: str = "line",
) -> list[str]:
    """Convenience for small files/tests; prefer ``iter_sources`` for real data."""
    return [text for _, text, _ in iter_sources(paths, text_field=text_field, txt_mode=txt_mode)]
