"""BPE tokenizer tests: training invariants, encode/decode, persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.tokenization import BYTE_VOCAB_SIZE, BPETokenizer, BPETrainer, pretokenize

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick and the dog is lazy",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the the the the the",
] * 20


def train(**kwargs) -> BPETokenizer:
    params = {"vocab_size": 320, "min_frequency": 1}
    params.update(kwargs)
    return BPETokenizer.train(CORPUS, **params)


def test_pretokenize_keeps_boundaries() -> None:
    assert pretokenize("hello world") == ["hello", " ", "world"]
    assert pretokenize("a  b\n") == ["a", "  ", "b", "\n"]
    assert pretokenize("3.14") == ["3", ".", "14"]


def test_training_produces_expected_vocab_size() -> None:
    tokenizer = train()
    # The corpus may run out of distinct pairs before the budget is reached.
    assert len(tokenizer) <= 320
    assert len(tokenizer) == BYTE_VOCAB_SIZE + tokenizer.n_merges
    assert tokenizer.n_merges == len(tokenizer.merges)
    assert tokenizer.vocab[0] == b"\x00"


def test_large_corpus_reaches_requested_vocab_size() -> None:
    corpus = [f"token{i} value{i} pair{i} word{i}" for i in range(600)]
    tokenizer = BPETokenizer.train(corpus, vocab_size=320, min_frequency=1)
    assert len(tokenizer) == 320


def test_training_is_deterministic() -> None:
    a = train()
    b = train()
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_min_frequency_stops_merging() -> None:
    tokenizer = BPETokenizer.train(["a b c d e f g"], vocab_size=400, min_frequency=5)
    assert len(tokenizer) <= BYTE_VOCAB_SIZE + 1


def test_vocab_size_must_exceed_bytes() -> None:
    with pytest.raises(ValueError):
        BPETrainer(vocab_size=256)


def test_invalid_min_frequency_rejected() -> None:
    with pytest.raises(ValueError):
        BPETrainer(min_frequency=0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "héllo wörld",
        "世界 语言模型",
        "emoji 🚀🔥 test",
        "tabs\tand\nnewlines\r\n",
        "   ",
        "a" * 1000,
        "\x1b[31mansi\x1b[0m",
    ],
)
def test_encode_decode_roundtrip(text: str) -> None:
    tokenizer = train()
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_merges_compress_repetitive_text() -> None:
    tokenizer = train()
    text = "the quick brown fox " * 10
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))


def test_encode_is_stable_across_calls() -> None:
    tokenizer = train()
    assert tokenizer.encode("the quick brown fox") == tokenizer.encode("the quick brown fox")


def test_batch_matches_single() -> None:
    tokenizer = train()
    texts = ["the fox", "quick dog", ""]
    assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]


def test_save_and_reload_preserves_behaviour(tmp_path: Path) -> None:
    tokenizer = train(special_tokens=["<|endoftext|>"])
    path = tokenizer.save(tmp_path / "tok")

    reloaded = BPETokenizer.load(path)
    reloaded_dir = BPETokenizer.load(tmp_path / "tok")
    text = "the quick brown fox\n世界"
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert reloaded_dir.encode(text) == tokenizer.encode(text)
    assert reloaded.config == tokenizer.config
    assert reloaded.special_tokens == tokenizer.special_tokens
    assert "created_at" in reloaded.metadata


def test_special_tokens_roundtrip(tmp_path: Path) -> None:
    tokenizer = train(special_tokens=["<|endoftext|>"])
    special_id = tokenizer.special_tokens["<|endoftext|>"]
    ids = tokenizer.encode("a<|endoftext|>b")
    assert special_id in ids
    assert tokenizer.decode(ids) == "a<|endoftext|>b"
    assert tokenizer.decode(ids, skip_special_tokens=True) == "ab"


def test_visualization_matches_encoding() -> None:
    tokenizer = train()
    text = "the quick brown fox"
    tokens = tokenizer.visualize(text)
    assert [t.id for t in tokens] == tokenizer.encode(text)
    assert "".join(t.text for t in tokens) == text


def test_visualization_includes_special_tokens() -> None:
    tokenizer = train(special_tokens=["<|endoftext|>"])
    tokens = tokenizer.visualize("x<|endoftext|>")
    assert tokens[-1].is_special
    assert tokens[-1].text == "<|endoftext|>"


def test_unknown_token_id_raises() -> None:
    tokenizer = train()
    with pytest.raises(ValueError):
        tokenizer.decode([999_999])
    with pytest.raises(ValueError):
        tokenizer.token_text(999_999)


def test_load_rejects_wrong_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"type": "wordpiece", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


def test_load_rejects_corrupt_merges(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "type": "byte_bpe",
        "config": {"vocab_size": 258, "min_frequency": 1, "pretoken_pattern": "."},
        "merges": [[0, 999]],
        "special_tokens": {},
        "metadata": {},
    }
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


def test_load_rejects_special_token_collision(tmp_path: Path) -> None:
    tokenizer = train()
    path = tokenizer.save(tmp_path / "tok")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["special_tokens"] = {"<|eos|>": 0}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


def test_missing_tokenizer_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path / "missing")


def test_trainer_stats_describe_corpus() -> None:
    trainer = BPETrainer(vocab_size=300, min_frequency=1)
    result = trainer.train(["abc def"])
    assert result.stats.documents == 1
    assert result.stats.chars == 7
    assert result.stats.pretokens == 3
    assert result.stats.merge_steps == len(result.merges)
    assert result.stats.duration_seconds >= 0
