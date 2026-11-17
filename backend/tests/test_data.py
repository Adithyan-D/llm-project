"""Dataset pipeline tests: loaders, normalisation, dedup, splitting, stats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.data import (
    DatasetConfig,
    DatasetFormatError,
    Deduplicator,
    NormalizationConfig,
    detect_format,
    iter_sources,
    iter_split,
    normalize_text,
    process_dataset,
    read_meta,
    text_fingerprint,
    validate_text,
)
from ml.tokenization import BPETokenizer


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------- loaders
def test_txt_line_mode_skips_blank_lines(tmp_path: Path) -> None:
    src = write(tmp_path / "a.txt", "first line\n\n   \nsecond line\n")
    issues: list = []
    records = list(iter_sources([src], issues=issues))
    assert [text for _, text, _ in records] == ["first line", "second line"]
    assert [location for _, _, location in records] == [1, 4]
    assert issues == []


def test_txt_document_mode_yields_whole_file(tmp_path: Path) -> None:
    src = write(tmp_path / "a.txt", "line1\nline2\n")
    records = list(iter_sources([src], txt_mode="document"))
    assert len(records) == 1
    assert records[0][1] == "line1\nline2\n"


def test_jsonl_collects_structural_issues_but_keeps_valid_records(tmp_path: Path) -> None:
    src = write(
        tmp_path / "a.jsonl",
        '{"text": "good one"}\n'
        "not json\n"
        '{"missing": "field"}\n'
        '{"text": 123}\n'
        '{"content": "fallback field"}\n'
        '{"text": "good two"}\n',
    )
    issues: list = []
    records = [t for _, t, _ in iter_sources([src], issues=issues)]
    assert records == ["good one", "fallback field", "good two"]
    assert len(issues) == 3
    assert issues[0].line == 2


def test_json_array_of_strings_and_objects(tmp_path: Path) -> None:
    src = write(tmp_path / "a.json", json.dumps(["alpha", {"text": "beta"}, {"x": 1}]))
    issues: list = []
    assert [t for _, t, _ in iter_sources([src], issues=issues)] == ["alpha", "beta"]
    assert len(issues) == 1


def test_json_wrapped_records_key(tmp_path: Path) -> None:
    src = write(tmp_path / "a.json", json.dumps({"records": [{"text": "wrapped"}]}))
    assert [t for _, t, _ in iter_sources([src])] == ["wrapped"]


def test_unsupported_extension_rejected(tmp_path: Path) -> None:
    src = write(tmp_path / "a.csv", "a,b\n")
    with pytest.raises(DatasetFormatError):
        detect_format(src)
    with pytest.raises(DatasetFormatError):
        list(iter_sources([src]))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_sources([tmp_path / "nope.txt"]))


def test_malformed_json_file_raises(tmp_path: Path) -> None:
    src = write(tmp_path / "a.json", "{not valid")
    with pytest.raises(DatasetFormatError):
        list(iter_sources([src]))


# ---------------------------------------------------------------- preprocessing
def test_default_normalisation_is_conservative() -> None:
    cfg = NormalizationConfig()
    assert normalize_text("  Caf\u00e9  \n", cfg) == "Caf\u00e9"
    assert normalize_text("a\x00b", cfg) == "ab"
    assert normalize_text("keep   inner   spaces", cfg) == "keep   inner   spaces"


def test_nfkc_ligature_and_optional_space_collapse() -> None:
    cfg = NormalizationConfig(collapse_spaces=True)
    assert normalize_text("\ufb01ne   text", cfg) == "fine text"
    assert normalize_text("a \n  b", cfg) == "a\nb"


def test_validation_bounds() -> None:
    cfg = NormalizationConfig(min_chars=3, max_chars=5)
    assert validate_text("ab", cfg) == "too_short"
    assert validate_text("abcdef", cfg) == "too_long"
    assert validate_text("abc", cfg) is None


def test_deduplicator_counts_and_fingerprints_are_stable() -> None:
    dedup = Deduplicator()
    assert dedup.is_duplicate("same") is False
    assert dedup.is_duplicate("same") is True
    assert dedup.is_duplicate("other") is False
    assert dedup.duplicates == 1
    assert len(dedup) == 2
    assert text_fingerprint("x") == text_fingerprint("x")
    assert len(text_fingerprint("x")) == 16


# ------------------------------------------------------------ end-to-end run
def make_corpus(tmp_path: Path) -> Path:
    lines = [f"sample number {i} with some words to count" for i in range(60)]
    lines += ["sample number 0 with some words to count"] * 5  # duplicates
    lines += ["ab"] * 3  # too short
    return write(tmp_path / "corpus.txt", "\n".join(lines) + "\n\n")


def run(corpus: Path, out: Path, **overrides) -> object:
    cfg = DatasetConfig(
        name="unit",
        inputs=[str(corpus)],
        output_dir=str(out),
        normalization=NormalizationConfig(min_chars=3),
        **overrides,
    )
    return process_dataset(cfg)


def test_process_dataset_writes_splits_and_meta(tmp_path: Path) -> None:
    result = run(make_corpus(tmp_path), tmp_path / "out")
    out = Path(result.output_dir)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "quarantine.jsonl", "meta.json"):
        assert (out / name).exists()

    total = sum(1 for split in ("train", "val", "test") for _ in iter_split(out, split))
    assert total == 60
    assert result.rejected["duplicate"] == 5
    assert result.rejected["too_short"] == 3
    assert result.stats["samples"] == 60

    meta = read_meta(out)
    assert meta["dataset_id"] == result.dataset_id
    assert meta["sources"][0]["bytes"] > 0
    shares = [meta["stats"]["splits"][s]["sample_share"] for s in ("train", "val", "test")]
    assert abs(sum(shares) - 1.0) < 1e-6


def test_stats_match_split_contents(tmp_path: Path) -> None:
    result = run(make_corpus(tmp_path), tmp_path / "out")
    out = Path(result.output_dir)
    texts = [t for split in ("train", "val", "test") for t in iter_split(out, split)]
    assert result.stats["chars"] == sum(len(t) for t in texts)
    assert result.stats["words"] == sum(len(t.split()) for t in texts)


def test_split_assignment_is_deterministic(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    first = run(corpus, tmp_path / "out1")
    second = run(corpus, tmp_path / "out2")
    assert first.dataset_hash == second.dataset_hash
    for split in ("train", "val", "test"):
        assert list(iter_split(first.output_dir, split)) == list(
            iter_split(second.output_dir, split)
        )


def test_seed_changes_split_assignment(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    a = run(corpus, tmp_path / "a", seed=1)
    b = run(corpus, tmp_path / "b", seed=2)
    assert list(iter_split(a.output_dir, "test")) != list(iter_split(b.output_dir, "test"))


def test_max_records_limit(tmp_path: Path) -> None:
    result = run(make_corpus(tmp_path), tmp_path / "out", max_records=10)
    assert result.stats["samples"] == 10


def test_invalid_fractions_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DatasetConfig(name="x", inputs=["a.txt"], val_fraction=0.6, test_fraction=0.5)
    with pytest.raises(ValueError):
        DatasetConfig(name="x", inputs=[])


def test_exact_token_stats_with_tokenizer(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    docs = [line for line in corpus.read_text(encoding="utf-8").splitlines() if len(line) >= 3]
    tokenizer = BPETokenizer.train(docs[:20], vocab_size=300, min_frequency=1)
    tok_path = tokenizer.save(tmp_path / "tok")

    result = run(corpus, tmp_path / "out", tokenizer_path=str(tok_path))
    out = Path(result.output_dir)
    assert read_meta(out)["token_count_method"] == "exact_bpe"
    recounted = sum(
        len(tokenizer.encode(t)) for s in ("train", "val", "test") for t in iter_split(out, s)
    )
    assert result.stats["tokens"] == recounted
