"""Incremental dataset statistics (no full-corpus buffering)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SplitStats:
    samples: int = 0
    chars: int = 0
    words: int = 0
    min_chars: int | None = None
    max_chars: int | None = None
    tokens: int | None = None
    bytes_on_disk: int = 0

    def add(self, text: str) -> None:
        length = len(text)
        self.samples += 1
        self.chars += length
        self.words += len(text.split())
        self.min_chars = length if self.min_chars is None else min(self.min_chars, length)
        self.max_chars = length if self.max_chars is None else max(self.max_chars, length)

    def add_tokens(self, count: int) -> None:
        self.tokens = (self.tokens or 0) + count

    def as_dict(self) -> dict[str, float | int | None]:
        avg_chars = self.chars / self.samples if self.samples else 0.0
        avg_words = self.words / self.samples if self.samples else 0.0
        tokens = self.tokens
        return {
            "samples": self.samples,
            "chars": self.chars,
            "words": self.words,
            "avg_chars": round(avg_chars, 2),
            "avg_words": round(avg_words, 2),
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "tokens": tokens,
            "bytes_on_disk": self.bytes_on_disk,
        }


def merge_stats(stats: dict[str, SplitStats]) -> dict[str, object]:
    """Combine per-split stats into a total block plus split shares."""
    total = SplitStats()
    total_tokens: int | None = 0
    for split in stats.values():
        total.samples += split.samples
        total.chars += split.chars
        total.words += split.words
        total.bytes_on_disk += split.bytes_on_disk
        if split.min_chars is not None:
            total.min_chars = (
                split.min_chars
                if total.min_chars is None
                else min(total.min_chars, split.min_chars)
            )
        if split.max_chars is not None:
            total.max_chars = (
                split.max_chars
                if total.max_chars is None
                else max(total.max_chars, split.max_chars)
            )
        if split.tokens is None:
            total_tokens = None
        elif total_tokens is not None:
            total_tokens += split.tokens
    total.tokens = total_tokens

    result = total.as_dict()
    result["splits"] = {
        name: {
            **split.as_dict(),
            "sample_share": round(split.samples / total.samples, 4) if total.samples else 0.0,
        }
        for name, split in stats.items()
    }
    return result
