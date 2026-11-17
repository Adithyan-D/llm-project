"""Train a byte-level BPE tokenizer on a processed dataset split.

Example:
    .venv\\Scripts\\python.exe scripts\\train_tokenizer.py ^
        --dataset datasets\\processed\\tinyshakespeare-<hash> --vocab-size 1024

The tokenizer is saved under model_registry/tokenizers/<name>/ together with a
report describing the corpus, config and the sample encoding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.data import iter_split, read_meta  # noqa: E402
from ml.tokenization import BPETokenizer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="processed dataset directory")
    parser.add_argument("--split", default="train")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--name", default=None, help="tokenizer name (default: <dataset>-bpe-N)")
    parser.add_argument("--out", default="model_registry/tokenizers")
    parser.add_argument("--special-token", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--compare-hf", nargs="?", const="gpt2", default=None)
    parser.add_argument(
        "--demo-text", default="First Citizen:\nBefore we proceed any further, hear me speak."
    )
    return parser


def _sample_compression(tokenizer: BPETokenizer, texts: list[str], limit: int = 1000) -> float:
    """chars/token measured on a bounded sample so cost stays O(1) in corpus size."""
    sample = texts[:limit]
    chars = sum(len(t) for t in sample)
    tokens = sum(len(tokenizer.encode(t)) for t in sample)
    return round(chars / tokens, 4) if tokens else 0.0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset)
    meta = read_meta(dataset_dir)
    name = args.name or f"{meta['dataset_id']}-bpe-{args.vocab_size}"

    start = time.perf_counter()
    texts = list(iter_split(dataset_dir, args.split))
    specials = [] if args.no_special_tokens else args.special_token
    tokenizer = BPETokenizer.train(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=specials,
        metadata={
            "dataset_id": meta["dataset_id"],
            "dataset_hash": meta["dataset_hash"],
            "dataset_split": args.split,
            "corpus_documents": len(texts),
            "corpus_chars": sum(len(t) for t in texts),
        },
    )
    train_seconds = round(time.perf_counter() - start, 3)

    out_dir = Path(args.out) / name
    save_path = tokenizer.save(out_dir)

    ids = tokenizer.encode(args.demo_text)
    visualization = tokenizer.visualize(args.demo_text)
    report = {
        "tokenizer": name,
        "path": str(save_path),
        "dataset": str(dataset_dir),
        "dataset_split": args.split,
        "vocab_size": len(tokenizer),
        "merges": tokenizer.n_merges,
        "special_tokens": tokenizer.special_tokens,
        "training_seconds": train_seconds,
        "chars": meta["stats"]["splits"][args.split]["chars"],
        "chars_per_token_sample": _sample_compression(tokenizer, texts),
        "demo_text": args.demo_text,
        "demo_ids": ids,
        "demo_tokens": [
            {"id": t.id, "text": t.text, "bytes": t.byte_length} for t in visualization
        ],
    }

    if args.compare_hf:
        from ml.tokenization.benchmark import benchmark_hf, benchmark_local

        raw_texts = texts[:200]
        report["comparison"] = {
            "corpus_sample_documents": len(raw_texts),
            "local": benchmark_local(tokenizer, raw_texts, repeats=2),
            "hf": benchmark_hf(
                raw_texts,
                tokenizer_id=args.compare_hf,
                repeats=2,
                cache_dir=str(PROJECT_ROOT / "model_registry" / "hf_cache"),
            ),
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"tokenizer    : {name}")
    print(f"saved        : {save_path}")
    print(f"vocab size   : {len(tokenizer)} ({tokenizer.n_merges} merges)")
    print(f"train time   : {train_seconds}s")
    print(f"demo text    : {args.demo_text!r}")
    print(f"demo tokens  : {[t.text for t in visualization]}")
    print(f"demo ids     : {ids}")
    print(f"round trip   : {tokenizer.decode(ids) == args.demo_text}")
    if "comparison" in report:
        print(json.dumps(report["comparison"], indent=2))
    print(f"report       : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
