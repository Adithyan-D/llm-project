"""Encode a processed dataset with a BPE tokenizer into cached training streams.

Example:
    .venv\\Scripts\\python.exe scripts\\prepare_training_data.py ^
        --dataset datasets\\processed\\tinyshakespeare-971a2b0d5ff5 ^
        --tokenizer model_registry\\tokenizers\\tinyshakespeare-bpe-1024

The cache lives next to the dataset as ``tokens-<tokenizer-hash>-eos<id>/`` with
one int32 ``.npy`` file per split and a manifest pinning the dataset and tokenizer
hashes. Re-running with the same inputs only verifies the files; changing either
input builds a separate cache instead of mixing vocabularies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.training import DEFAULT_SEPARATOR, build_token_streams  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--separator", default=DEFAULT_SEPARATOR)
    parser.add_argument("--no-separator", action="store_true")
    parser.add_argument("--out", default=None, help="cache directory (default: next to dataset)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache matches")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    streams = build_token_streams(
        args.dataset,
        args.tokenizer,
        cache_dir=args.out,
        splits=tuple(args.splits),
        document_separator=None if args.no_separator else args.separator,
        force=args.force,
    )

    print(f"dataset       : {streams.manifest['dataset_id']}")
    print(f"hash          : {streams.manifest['dataset_hash']}")
    print(f"tokenizer     : {args.tokenizer}")
    print(f"vocab size    : {streams.vocab_size}")
    print(f"eos token id  : {streams.eos_token_id}")
    print(f"cache         : {streams.cache_dir}")
    for name in args.splits:
        stream = streams.splits[name]
        size = stream.path.stat().st_size if stream.path.exists() else 0
        print(
            f"{name:<13} : {stream.tokens:>9,} tokens | "
            f"{stream.documents:>7,} docs | {stream.chars:>10,} chars | "
            f"{size / 1024:.1f} KiB"
        )
    print(f"manifest      : {streams.manifest_path}")
    print(json.dumps({name: stream.tokens for name, stream in streams.splits.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
