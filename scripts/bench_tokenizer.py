"""Compare the local BPE tokenizer with a Hugging Face tokenizer on real text.

Example:
    .venv\\Scripts\\python.exe scripts\\bench_tokenizer.py ^
        --tokenizer model_registry\\tokenizers\\tinyshakespeare-bpe-1024 ^
        --dataset datasets\\processed\\tinyshakespeare-<hash> --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.data import iter_split  # noqa: E402
from ml.tokenization import BPETokenizer  # noqa: E402
from ml.tokenization.benchmark import benchmark_hf, benchmark_local  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--hf", default="gpt2")
    parser.add_argument("--out", default="experiments")
    args = parser.parse_args(argv)

    local = BPETokenizer.load(args.tokenizer)
    texts = list(iter_split(args.dataset, args.split))[: args.limit] or [""]

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "documents": len(texts),
        "local": benchmark_local(local, texts),
        "hf": benchmark_hf(
            texts,
            tokenizer_id=args.hf,
            cache_dir=str(PROJECT_ROOT / "model_registry" / "hf_cache"),
        ),
    }

    out_path = Path(args.out) / "tokenizer_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
