# Dataset Pipeline & BPE Tokenizer (Phase 1)

Everything here is implemented and tested in this repository:

| Code | Tests |
|---|---|
| `backend/ml/data/{loaders,preprocess,stats,dataset}.py` | `backend/tests/test_data.py` |
| `backend/ml/tokenization/{bpe,tokenizer,benchmark}.py` | `backend/tests/test_tokenizer.py` |
| `scripts/{download_corpus,prepare_dataset,train_tokenizer,tokenize_demo,bench_tokenizer}.py` | run manually, see §8 |

---

## 1. Pipeline

```
raw file(s) → loader (streaming) → validate → normalize → dedup
            → hash-based split (train/val/test) → JSONL + meta.json
```

Each run writes `datasets/processed/<dataset_id>/`:

```
train.jsonl  val.jsonl  test.jsonl   one {"text": ...} object per line
quarantine.jsonl                     rejected samples + reason (capped, default 100 per reason)
meta.json                            config, source sha256, stats, dataset_hash, rejected counts
```

`dataset_id = <name>-<first 12 hex of dataset_hash>`.

**Dataset hash** = blake2b-256 over the 16-byte content fingerprints of every kept
record, in source order. It is a version identifier, not a security hash: rerunning
the same corpus + normalisation + dedup config reproduces it exactly, and a re-run
with an added tokenizer (for exact token counts) keeps the same id and replaces the
directory in place (idempotent, verified in tests).

## 2. Loaders

| Format | Mode | Memory |
|---|---|---|
| `.txt` | `line` (default) or `document` | streaming for `line` |
| `.jsonl` / `.ndjson` | one JSON value per line | streaming |
| `.json` | array, or `{"data"/"records"/"documents": [...]}`, or single object | whole file (stdlib limit) |

Rules:
- Records may be strings or objects. Text is taken from `--text-field` (default
  `text`) with fallbacks `content`, `document`, `body`.
- Structural problems (bad JSON line, missing/non-string text) are recorded as
  issues and the record is quarantined; a malformed *file* raises
  `DatasetFormatError` instead of silently producing an empty dataset.
- Blank lines are skipped without being counted as errors.
- Use JSONL for anything large; JSON arrays are documented as non-streaming.

## 3. Validation & normalisation

Defaults are deliberately conservative — normalisation must not rewrite content:

0. Optional Unicode `NFKC` (default on). Folds compatibility forms such as `ﬁ` → `fi`,
   which is safe for text but would be wrong for byte-exact code corpora — it is a
   config flag (`unicode_form=None`) for that case.
1. Control characters removed (`\x00-\x08`, `\x0b`, `\x0c`, `\x0e-\x1f`, `\x7f`).
   Newlines/tabs are preserved.
2. Optional whitespace collapsing (off by default): multi-space runs → single space,
   spaces around newlines removed.
3. Strip leading/trailing whitespace (default on).
4. Validation: `len >= min_chars`, `len <= max_chars` (optional). No lowercasing,
   no punctuation stripping, no stopword removal — those destroy signal the model
   could learn.

**Dedup** is exact-match on the blake2b-128 digest of the normalised text, keeping
the first occurrence. Memory ≈ 50 bytes/sample, so millions of samples stay feasible.
Near-duplicates (e.g. one-word edits) are *not* detected; that needs MinHash/SimHash
and is explicitly out of scope for Phase 1.

## 4. Splitting

Split is a pure function of `blake2b(content_digest + seed)` mapped to [0,1):

- `u < test_fraction` → test
- `u < test_fraction + val_fraction` → val
- else → train

Why not `random.shuffle` + slicing? Shuffling requires buffering the corpus (or
two passes) and makes splits order-dependent. Hash assignment is streaming,
order-independent for a given seed, reproducible across machines, and changing the
seed genuinely reshuffles (`test_seed_changes_split_assignment`). Actual proportions
deviate from the targets by sampling noise — measured 80.12/10.03/9.85 for a
80/10/10 target, recorded in `meta.json`.

## 5. Statistics

`meta.json → stats` contains per split and total: `samples`, `chars`, `words`
(whitespace split), `avg_chars`, `avg_words`, `min/max_chars`, `bytes_on_disk`, and
`tokens`. `tokens` is `null` unless a trained tokenizer was passed with
`--tokenizer`, in which case it is an **exact** BPE token count
(`token_count_method: exact_bpe`). No char/4 estimates are presented as if they were
measurements.

## 6. BPE tokenizer

### 6.1 Pretokenisation

```python
r"\s+|[^\W\d_]+|\d+|[^\w\s]"
```

whitespace runs | letter runs (any script) | digit runs | single other chars.
Merges never cross pretoken boundaries, so `"the cat"` cannot become one token and
whitespace stays cheap (one token per run) without swallowing words.

### 6.2 Alphabet: bytes, not words

- Base vocabulary is 256 byte values; every merge adds exactly one id.
  `vocab_size` is therefore a hard budget: `256 + n_merges (+ specials)`.
- Any UTF-8 text — emoji, CJK, code, invalid sequences from files — is encodable.
  There is **no `<unk>`** and decoding is lossless (property-tested on unicode,
  emoji, ANSI escapes, empty strings).
- Cost: non-ASCII text needs up to 3–4 tokens per character until merges learn
  common multi-byte sequences. A vocabulary of 1k is a small-vocab baseline, not a
  production tokenizer; Phase 3 can sweep `vocab_size` with the same code.

### 6.3 Training algorithm

```
unique_words = Counter(pretokenize(doc) as byte tuples)
pair_counts, pair_words = initial counts and inverted index
repeat until vocab_size reached or no pair has count >= min_frequency:
    pair = argmax count (ties broken by pair value → deterministic)
    vocab[new_id] = vocab[a] + vocab[b]
    for each word containing pair:
        remove old pair contributions, merge all occurrences, add new contributions
```

Sennrich et al. (2016) BPE with **incremental counters**. A naive implementation
rescans the whole corpus per merge: O(merges × corpus). This one touches only the
words containing the merged pair after the initial O(corpus) count, which is why
800 KB trains in seconds on 4 CPU cores.

### 6.4 Encoding / decoding

- Encode: per pretoken, repeatedly merge the adjacent pair with the **lowest merge
  rank** until none applies (the standard BPE rule), applying the chosen merge to
  all its occurrences in that pretoken. Per-pretoken results are memoised.
- Decode: concatenate token byte strings, then UTF-8 decode with `errors="replace"`
  at the end (never per token — a character may legitimately span tokens).
- Special tokens (e.g. `<|endoftext|>`) are matched literally with a length-ordered
  alternation, emitted as their own ids, and can be skipped on decode.
- Persistence stores only `merges` + config + specials; the vocabulary is rebuilt
  deterministically on load and validated (undefined ids, id collisions, format
  version all raise).

## 7. Corpus used

`tinyshakespeare` (public domain, 1,115,394 bytes, 32,777 lines), pinned by SHA-256
in `scripts/download_corpus.py`:

```powershell
.venv\Scripts\python.exe scripts\download_corpus.py --name tinyshakespeare
```

## 8. Measured results (2026-11-17, this machine)

Preprocessing (`scripts/prepare_dataset.py`, line mode, NFKC, dedup, 80/10/10):

| Metric | Value |
|---|---|
| Kept samples | 25,721 (train 20,607 / val 2,580 / test 2,534) |
| Duplicates removed | 7,056 (21.5% — repeated short Shakespeare lines) |
| Chars / words | 1,003,675 / 192,862 |
| Avg sample | 39.02 chars, 7.5 words |
| Exact tokens (BPE-1024) | 524,233 → 1.915 chars/token |
| Processed size on disk | 1.37 MB |
| Preprocess time | 2.1 s (second pass, with token counting) |

Tokenizer training (`scripts/train_tokenizer.py --vocab-size 1024`):

| Metric | Value |
|---|---|
| Requested / actual vocab | 1024 / **1025** = 256 bytes + 768 merges + 1 special |
| Early stop | `min_frequency=2` exhausted (no pair appears twice) |
| Training time | 2.85 s (800 KB train split, 4-thread CPU) |
| File size | 33.5 KB (`tokenizer.json`) |
| Round-trip (`decode(encode(x)) == x`) | yes, on all test splits and edge cases |

Comparison on 500 held-out test documents (`scripts/bench_tokenizer.py`):

| Tokenizer | Vocab | Chars/token | Encode throughput | Round-trip |
|---|---|---|---|---|
| local byte-BPE (this repo) | 1,025 | 1.91 | 1.37 M chars/s | yes |
| HF `gpt2` (Rust `tokenizers`) | 50,257 | 3.82 | 2.63 M chars/s | yes |

Read this honestly: GPT-2 compresses ~2× better because it was trained on a corpus
several orders of magnitude larger with a 50× bigger vocabulary; its encoder is also
Rust-backed. The comparison validates correctness and gives a baseline — it does not
claim the local tokenizer is better. With `vocab_size=1024` the model in Phase 3
simply gets longer sequences; that trade-off will be measured, not assumed.

## 9. Reproduce Phase 1 end to end

```powershell
$py = ".venv\Scripts\python.exe"
& $py scripts\download_corpus.py --name tinyshakespeare
& $py scripts\prepare_dataset.py --input datasets\raw\tinyshakespeare\input.txt --name tinyshakespeare
& $py scripts\train_tokenizer.py --dataset datasets\processed\tinyshakespeare-971a2b0d5ff5 `
      --vocab-size 1024 --name tinyshakespeare-bpe-1024 --compare-hf gpt2
& $py scripts\prepare_dataset.py --input datasets\raw\tinyshakespeare\input.txt --name tinyshakespeare `
      --tokenizer model_registry\tokenizers\tinyshakespeare-bpe-1024
& $py scripts\tokenize_demo.py --tokenizer model_registry\tokenizers\tinyshakespeare-bpe-1024 `
      --text "To be, or not to be"
& $py -m pytest
& $py -m ruff check .
```

## 10. Known limitations (Phase 1)

- JSON arrays are parsed whole; use JSONL for larger corpora.
- Exact-match dedup only (no near-duplicate detection).
- No document-level packing or EOS insertion yet — that belongs to Phase 3, which
  will consume `train.jsonl` and append `<|endoftext|>` between samples.
- Word count is a whitespace heuristic, not linguistic tokenisation.
- Benchmark throughput is single-threaded Python vs single-threaded Rust; batch
  behaviour of the local tokenizer is deliberately simple (`list` comprehension).
