"""Visualise text -> BPE tokens -> ids for a trained tokenizer.

Example:
    .venv\\Scripts\\python.exe scripts\\tokenize_demo.py ^
        --tokenizer model_registry\\tokenizers\\tinyshakespeare-bpe-1024 ^
        --text "To be, or not to be"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.data import iter_split  # noqa: E402
from ml.tokenization import BPETokenizer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text", default=None, help="text to tokenize")
    parser.add_argument("--dataset", default=None, help="dataset dir to sample text from")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = BPETokenizer.load(args.tokenizer)

    if args.text is not None:
        text = args.text
    elif args.dataset:
        text = list(iter_split(args.dataset, args.split))[args.index]
    else:
        raise SystemExit("provide --text or --dataset")

    tokens = tokenizer.visualize(text)
    ids = tokenizer.encode(text)

    print(f"vocab size : {len(tokenizer)} ({tokenizer.n_merges} merges)")
    print(
        f"chars      : {len(text)}  tokens: {len(ids)}  chars/token: {len(text) / max(1, len(ids)):.2f}"
    )
    print()
    for token in tokens:
        print(f"  id={token.id:5d} bytes={token.byte_length}  {token.text!r}")
    print()
    print("ids:", ids)
    print("decoded equals input:", tokenizer.decode(ids) == text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
