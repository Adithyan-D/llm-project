"""Tokenization benchmark helpers (comparison only; not used for tokenization)."""

from __future__ import annotations

import time
from collections.abc import Iterable

from .tokenizer import BPETokenizer


def _encode_all(tokenizer, texts: list[str]) -> tuple[list[list[int]], float]:
    start = time.perf_counter()
    encoded = [tokenizer.encode(text) for text in texts]
    elapsed = time.perf_counter() - start
    return encoded, elapsed


def benchmark_local(
    tokenizer: BPETokenizer,
    texts: Iterable[str],
    *,
    repeats: int = 3,
    warmup: int = 1,
) -> dict[str, object]:
    """Measure compression and encoding throughput for the from-scratch tokenizer."""
    corpus = list(texts)
    chars = sum(len(t) for t in corpus)

    for text in corpus[: max(1, len(corpus) // 20)]:
        tokenizer.encode(text)

    encoded: list[list[int]] = []
    total_seconds = 0.0
    for _ in range(warmup):
        _encode_all(tokenizer, corpus)
    for _ in range(repeats):
        encoded, elapsed = _encode_all(tokenizer, corpus)
        total_seconds += elapsed

    tokens = sum(len(ids) for ids in encoded)
    roundtrip = all(
        tokenizer.decode(ids) == text for ids, text in zip(encoded, corpus, strict=True)
    )
    return {
        "tokenizer": "local_byte_bpe",
        "vocab_size": len(tokenizer),
        "merges": tokenizer.n_merges,
        "documents": len(corpus),
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": round(chars / tokens, 4) if tokens else 0.0,
        "encode_chars_per_second": round(chars * repeats / total_seconds, 1)
        if total_seconds
        else 0.0,
        "roundtrip_ok": roundtrip,
    }


def benchmark_hf(
    texts: Iterable[str],
    *,
    tokenizer_id: str = "gpt2",
    repeats: int = 3,
    warmup: int = 1,
    cache_dir: str | None = None,
) -> dict[str, object]:
    """Benchmark a Hugging Face tokenizer (downloaded once into the project cache)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer_path = hf_hub_download(
        repo_id=tokenizer_id, filename="tokenizer.json", cache_dir=cache_dir
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    corpus = list(texts)
    chars = sum(len(t) for t in corpus)

    for text in corpus[: max(1, len(corpus) // 20)]:
        tokenizer.encode(text)

    encoded = []
    total_seconds = 0.0
    for _ in range(warmup):
        tokenizer.encode_batch(corpus)
    for _ in range(repeats):
        start = time.perf_counter()
        encoded = tokenizer.encode_batch(corpus)
        total_seconds += time.perf_counter() - start

    tokens = sum(len(enc.ids) for enc in encoded)
    decoded_ok = all(
        tokenizer.decode(enc.ids, skip_special_tokens=True) == text
        for enc, text in zip(encoded, corpus, strict=True)
    )
    unknown_id = tokenizer.token_to_id("<|endoftext|>")
    return {
        "tokenizer": tokenizer_id,
        "vocab_size": tokenizer.get_vocab_size(),
        "documents": len(corpus),
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": round(chars / tokens, 4) if tokens else 0.0,
        "encode_chars_per_second": round(chars * repeats / total_seconds, 1)
        if total_seconds
        else 0.0,
        "roundtrip_ok": bool(decoded_ok),
        "eos_token_id": unknown_id,
    }
