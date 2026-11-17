"""Byte-Pair Encoding trainer implemented from scratch (no ``tokenizers`` dependency).

Why byte-level BPE?
-------------------
* Lossless: every UTF-8 string is representable, so there is no ``<unk>`` token
  and decoding is exact. This matters for code, emoji and mixed scripts.
* Small alphabet: the base vocabulary is 256 byte tokens, and every merge adds
  exactly one token, so ``vocab_size`` is a hard, predictable knob.
* It is the scheme used by GPT-2/GPT-4 class models, so the architecture in
  Phase 2/3 lines up with well-understood behaviour.

Algorithm (Sennrich et al. 2016, applied to bytes):
  1. Split text into pretokens (words/whitespace/punctuation) with a regex.
  2. Count word frequencies so the corpus is scanned once.
  3. Repeatedly merge the most frequent adjacent symbol pair (ties broken by
     the pair itself for determinism) and add the merged symbol to the vocab.

Complexity: initial pair counting is O(total pretokens). Each merge only
re-touches words that contain the merged pair (incremental counters), instead
of rescanning the corpus, so training is O(total pretokens + sum of affected
words). Naive implementations rescan everything once per merge, which is
O(merges x corpus) - prohibitive on CPU.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

BYTE_VOCAB_SIZE = 256

# GPT-2-style pretokenisation, expressed with stdlib ``re`` (no \p{L} support):
# whitespace | letters (any script) | digit runs | any other single character.
DEFAULT_PRETOKEN_PATTERN = r"\s+|[^\W\d_]+|\d+|[^\w\s]"


def pretokenize(text: str, pattern: str = DEFAULT_PRETOKEN_PATTERN) -> list[str]:
    """Split text into BPE training units.

    Merges never cross pretoken boundaries. This keeps common words compact
    while preventing merges like ``"the cat"`` from swallowing whitespace into
    low-value multi-word tokens.
    """
    return re.findall(pattern, text)


@dataclass
class TrainingStats:
    documents: int = 0
    chars: int = 0
    pretokens: int = 0
    unique_pretokens: int = 0
    merge_steps: int = 0
    vocab_size: int = BYTE_VOCAB_SIZE
    duration_seconds: float = 0.0
    last_merge_count: int = 0
    merge_history: list[tuple[tuple[int, int], int]] = field(default_factory=list)


@dataclass
class TrainingResult:
    vocab: dict[int, bytes]
    merges: list[tuple[int, int]]
    stats: TrainingStats


def _pairs(word: tuple[int, ...]):
    return zip(word, word[1:], strict=False)


def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Replace non-overlapping occurrences of ``pair`` in one word."""
    out: list[int] = []
    index = 0
    length = len(word)
    while index < length:
        if index < length - 1 and word[index] == pair[0] and word[index + 1] == pair[1]:
            out.append(new_id)
            index += 2
        else:
            out.append(word[index])
            index += 1
    return tuple(out)


class BPETrainer:
    """Train a byte-level BPE vocabulary on an iterable of documents."""

    def __init__(
        self,
        vocab_size: int = 512,
        min_frequency: int = 2,
        pretoken_pattern: str = DEFAULT_PRETOKEN_PATTERN,
    ) -> None:
        if vocab_size <= BYTE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be > {BYTE_VOCAB_SIZE}")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pretoken_pattern = pretoken_pattern

    def train(self, documents: Iterable[str]) -> TrainingResult:
        started = time.perf_counter()
        stats = TrainingStats()

        word_counts: Counter[tuple[int, ...]] = Counter()
        for document in documents:
            stats.documents += 1
            stats.chars += len(document)
            for pretoken in pretokenize(document, self.pretoken_pattern):
                stats.pretokens += 1
                word_counts[tuple(pretoken.encode("utf-8"))] += 1
        stats.unique_pretokens = len(word_counts)

        words = list(word_counts.keys())
        frequencies = list(word_counts.values())

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for index, (word, frequency) in enumerate(zip(words, frequencies, strict=True)):
            for pair in _pairs(word):
                pair_counts[pair] += frequency
                pair_words[pair].add(index)

        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}
        merges: list[tuple[int, int]] = []
        target_merges = self.vocab_size - BYTE_VOCAB_SIZE

        while len(merges) < target_merges and pair_counts:
            (pair, count) = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            if count < self.min_frequency:
                break

            new_id = BYTE_VOCAB_SIZE + len(merges)
            merges.append(pair)
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            stats.merge_history.append((pair, count))

            affected = pair_words.pop(pair, set())
            for index in affected:
                old_word = words[index]
                frequency = frequencies[index]
                for old_pair in _pairs(old_word):
                    remaining = pair_counts[old_pair] - frequency
                    if remaining <= 0:
                        pair_counts.pop(old_pair, None)
                        pair_words.get(old_pair, set()).discard(index)
                    else:
                        pair_counts[old_pair] = remaining
                        pair_words[old_pair].discard(index)

                new_word = _merge_word(old_word, pair, new_id)
                words[index] = new_word
                for new_pair in _pairs(new_word):
                    pair_counts[new_pair] += frequency
                    pair_words[new_pair].add(index)

        stats.merge_steps = len(merges)
        stats.vocab_size = BYTE_VOCAB_SIZE + len(merges)
        stats.last_merge_count = int(pair_counts.most_common(1)[0][1]) if pair_counts else 0
        stats.duration_seconds = round(time.perf_counter() - started, 3)
        return TrainingResult(vocab=vocab, merges=merges, stats=stats)
