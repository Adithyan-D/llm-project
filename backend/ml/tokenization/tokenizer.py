"""Byte-level BPE tokenizer: encoding, decoding, persistence, inspection.

Encoding applies the learned merges greedily by rank (the standard BPE encode
rule): for every pretoken, merge the pair with the lowest merge index until no
pair remains. Per-pretoken results are memoised, which is the main CPU win for
repeated vocabulary items.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .bpe import (
    BYTE_VOCAB_SIZE,
    DEFAULT_PRETOKEN_PATTERN,
    BPETrainer,
    pretokenize,
)

TOKENIZER_FILENAME = "tokenizer.json"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 512
    min_frequency: int = 2
    pretoken_pattern: str = DEFAULT_PRETOKEN_PATTERN


@dataclass
class VisualToken:
    id: int
    text: str
    byte_length: int
    is_special: bool = False


@dataclass
class BPETokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[int, int]]
    special_tokens: dict[str, int] = field(default_factory=dict)
    config: TokenizerConfig = field(default_factory=TokenizerConfig)
    metadata: dict[str, object] = field(default_factory=dict)
    _merge_ranks: dict[tuple[int, int], int] = field(init=False, repr=False)
    _merge_ids: dict[tuple[int, int], int] = field(init=False, repr=False)
    _special_ids: dict[int, str] = field(init=False, repr=False)
    _special_pattern: re.Pattern[str] | None = field(init=False, repr=False, default=None)
    _cache: dict[bytes, tuple[int, ...]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.merges = [tuple(pair) for pair in self.merges]  # type: ignore[misc]
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._merge_ids = {pair: BYTE_VOCAB_SIZE + rank for rank, pair in enumerate(self.merges)}
        self._special_ids = {index: token for token, index in self.special_tokens.items()}
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile("|".join(re.escape(t) for t in ordered))

    # ---------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        documents: Iterable[str],
        *,
        vocab_size: int = 512,
        min_frequency: int = 2,
        special_tokens: Iterable[str] | None = None,
        pretoken_pattern: str = DEFAULT_PRETOKEN_PATTERN,
        metadata: dict[str, object] | None = None,
    ) -> BPETokenizer:
        trainer = BPETrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            pretoken_pattern=pretoken_pattern,
        )
        result = trainer.train(documents)

        specials: dict[str, int] = {}
        next_id = BYTE_VOCAB_SIZE + len(result.merges)
        for token in special_tokens or ():
            if token in specials:
                continue
            specials[token] = next_id
            next_id += 1

        meta = dict(metadata or {})
        meta.setdefault("created_at", datetime.now(UTC).isoformat(timespec="seconds"))
        meta.update(
            {
                "training_documents": result.stats.documents,
                "training_chars": result.stats.chars,
                "training_pretokens": result.stats.pretokens,
                "unique_pretokens": result.stats.unique_pretokens,
                "merge_steps": result.stats.merge_steps,
                "training_seconds": result.stats.duration_seconds,
            }
        )
        return cls(
            vocab=result.vocab,
            merges=result.merges,
            special_tokens=specials,
            config=TokenizerConfig(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                pretoken_pattern=pretoken_pattern,
            ),
            metadata=meta,
        )

    # ---------------------------------------------------------------- encoding
    @property
    def n_merges(self) -> int:
        return len(self.merges)

    @property
    def n_base_tokens(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.vocab) + len(self.special_tokens)

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        raw = pretoken.encode("utf-8")
        cached = self._cache.get(raw)
        if cached is not None:
            return cached
        symbols = list(raw)
        while len(symbols) >= 2:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            for left, right in zip(symbols, symbols[1:], strict=False):
                rank = self._merge_ranks.get((left, right))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, (left, right)
            if best_pair is None or best_rank is None:
                break
            new_id = self._merge_ids[best_pair]
            merged: list[int] = []
            index = 0
            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and symbols[index] == best_pair[0]
                    and symbols[index + 1] == best_pair[1]
                ):
                    merged.append(new_id)
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        result = tuple(symbols)
        self._cache[raw] = result
        return result

    def iter_ids(self, text: str) -> Iterator[int]:
        """Yield token ids, handling special tokens that appear literally."""
        if self._special_pattern is None:
            for pretoken in pretokenize(text, self.config.pretoken_pattern):
                yield from self._encode_pretoken(pretoken)
            return

        position = 0
        for match in self._special_pattern.finditer(text):
            for pretoken in pretokenize(
                text[position : match.start()], self.config.pretoken_pattern
            ):
                yield from self._encode_pretoken(pretoken)
            yield self.special_tokens[match.group()]
            position = match.end()
        for pretoken in pretokenize(text[position:], self.config.pretoken_pattern):
            yield from self._encode_pretoken(pretoken)

    def encode(self, text: str) -> list[int]:
        return list(self.iter_ids(text))

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    # ---------------------------------------------------------------- decoding
    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        buffer = bytearray()
        pieces: list[str] = []

        def flush() -> None:
            if buffer:
                pieces.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()

        for token_id in ids:
            special = self._special_ids.get(token_id)
            if special is not None:
                flush()
                if not skip_special_tokens:
                    pieces.append(special)
                continue
            token_bytes = self.vocab.get(token_id)
            if token_bytes is None:
                raise ValueError(f"unknown token id {token_id}")
            buffer.extend(token_bytes)
        flush()
        return "".join(pieces)

    def token_bytes(self, token_id: int) -> bytes | None:
        return self.vocab.get(token_id)

    def token_text(self, token_id: int) -> str:
        special = self._special_ids.get(token_id)
        if special is not None:
            return special
        token_bytes = self.vocab.get(token_id)
        if token_bytes is None:
            raise ValueError(f"unknown token id {token_id}")
        return token_bytes.decode("utf-8", errors="replace")

    # -------------------------------------------------------------- inspection
    def visualize(self, text: str) -> list[VisualToken]:
        """Split encoded id stream into displayable pieces (per pretoken).

        A merged token can contain several source characters, so pieces are
        decoded token-by-token; invalid UTF-8 fragments (multi-byte characters
        split across merges are impossible, but byte-level tokens are arbitrary
        before decoding) are shown with replacement characters.
        """
        tokens: list[VisualToken] = []
        position = 0
        if self._special_pattern is not None:
            segments: list[tuple[str, bool]] = []
            for match in self._special_pattern.finditer(text):
                segments.append((text[position : match.start()], False))
                segments.append((match.group(), True))
                position = match.end()
            segments.append((text[position:], False))
        else:
            segments = [(text, False)]

        for segment, is_special in segments:
            if is_special:
                tokens.append(
                    VisualToken(
                        id=self.special_tokens[segment],
                        text=segment,
                        byte_length=len(segment.encode("utf-8")),
                        is_special=True,
                    )
                )
                continue
            for pretoken in pretokenize(segment, self.config.pretoken_pattern):
                for token_id in self._encode_pretoken(pretoken):
                    raw = self.vocab[token_id]
                    tokens.append(
                        VisualToken(
                            id=token_id,
                            text=raw.decode("utf-8", errors="replace"),
                            byte_length=len(raw),
                        )
                    )
        return tokens

    def stats(self) -> dict[str, object]:
        return {
            "vocab_size": len(self),
            "base_tokens": self.n_base_tokens,
            "merges": self.n_merges,
            "special_tokens": dict(self.special_tokens),
            "config": asdict(self.config),
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".json":
            target.mkdir(parents=True, exist_ok=True)
            target = target / TOKENIZER_FILENAME
        else:
            target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": FORMAT_VERSION,
            "type": "byte_bpe",
            "config": asdict(self.config),
            "merges": [[left, right] for left, right in self.merges],
            "special_tokens": self.special_tokens,
            "metadata": {**self.metadata, **(metadata or {})},
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        target = Path(path)
        if target.is_dir():
            target = target / TOKENIZER_FILENAME
        if not target.is_file():
            raise FileNotFoundError(target)
        payload = json.loads(target.read_text(encoding="utf-8"))

        if payload.get("type") != "byte_bpe":
            raise ValueError(f"{target}: unsupported tokenizer type {payload.get('type')!r}")
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError(f"{target}: unsupported tokenizer format version")

        config = TokenizerConfig(**payload["config"])
        merges = [tuple(pair) for pair in payload["merges"]]

        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}
        for offset, (left, right) in enumerate(merges):
            if left not in vocab or right not in vocab:
                raise ValueError(f"{target}: merge {offset} references an undefined token")
            vocab[BYTE_VOCAB_SIZE + offset] = vocab[left] + vocab[right]
        if len(vocab) != BYTE_VOCAB_SIZE + len(merges):
            raise ValueError(f"{target}: corrupt merge table")

        special_tokens = dict(payload.get("special_tokens", {}))
        for token, token_id in special_tokens.items():
            if token_id < len(vocab):
                raise ValueError(
                    f"{target}: special token {token!r} collides with token id {token_id}"
                )

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
            config=config,
            metadata=dict(payload.get("metadata", {})),
        )
