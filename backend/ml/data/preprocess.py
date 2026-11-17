"""Text normalisation, quality validation and near-exact deduplication."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_SPACE_AROUND_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")


@dataclass(frozen=True)
class NormalizationConfig:
    """Conservative by default: normalisation must not rewrite content silently."""

    unicode_form: str | None = "NFKC"
    strip: bool = True
    collapse_spaces: bool = False
    remove_control_chars: bool = True
    min_chars: int = 1
    max_chars: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "unicode_form": self.unicode_form,
            "strip": self.strip,
            "collapse_spaces": self.collapse_spaces,
            "remove_control_chars": self.remove_control_chars,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
        }


def normalize_text(text: str, cfg: NormalizationConfig) -> str:
    """Apply the configured normalisation steps in a deterministic order."""
    if cfg.unicode_form:
        text = unicodedata.normalize(cfg.unicode_form, text)
    if cfg.remove_control_chars:
        text = _CONTROL_RE.sub("", text)
    if cfg.collapse_spaces:
        text = _SPACE_RUN_RE.sub(" ", text)
        text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    if cfg.strip:
        text = text.strip()
    return text


def validate_text(text: str, cfg: NormalizationConfig) -> str | None:
    """Return a rejection reason, or ``None`` if the text is acceptable."""
    if len(text) < cfg.min_chars:
        return "too_short"
    if cfg.max_chars is not None and len(text) > cfg.max_chars:
        return "too_long"
    if not text:
        return "empty"
    return None


def text_fingerprint(text: str) -> bytes:
    """Stable 16-byte content hash (blake2b) used for dedup and split assignment."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


class Deduplicator:
    """Hash-based exact dedup after normalisation.

    Memory: ~50 bytes per unique record (16-byte digest + set overhead), so
    ~5M unique samples stay in the low hundreds of MB. Near-duplicates are not
    detected; that would need MinHash/SimHash and is out of scope here.
    """

    def __init__(self) -> None:
        self._seen: set[bytes] = set()
        self.duplicates = 0

    def is_duplicate(self, text: str) -> bool:
        fingerprint = text_fingerprint(text)
        if fingerprint in self._seen:
            self.duplicates += 1
            return True
        self._seen.add(fingerprint)
        return False

    def __len__(self) -> int:
        return len(self._seen)
