"""Prepare a processed dataset (validate, normalise, dedup, split, stats).

Example:
    .venv\\Scripts\\python.exe scripts\\prepare_dataset.py ^
        --input datasets\\raw\\tinyshakespeare\\input.txt --name tinyshakespeare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.data import DatasetConfig, NormalizationConfig, process_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="TXT / JSON / JSONL sources")
    parser.add_argument("--name", required=True, help="dataset name used in the dataset id")
    parser.add_argument("--out", default="datasets/processed", help="processed output root")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--txt-mode", choices=("line", "document"), default="line")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="optional trained tokenizer for exact token counts (not for training)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = DatasetConfig(
        name=args.name,
        inputs=[str(Path(p)) for p in args.input],
        output_dir=args.out,
        text_field=args.text_field,
        txt_mode=args.txt_mode,
        val_fraction=args.val_frac,
        test_fraction=args.test_frac,
        seed=args.seed,
        normalization=NormalizationConfig(min_chars=args.min_chars, max_chars=args.max_chars),
        deduplicate=not args.no_dedup,
        max_records=args.max_records,
        tokenizer_path=args.tokenizer,
    )
    result = process_dataset(cfg)

    stats = result.stats
    print(f"dataset id   : {result.dataset_id}")
    print(f"output       : {result.output_dir}")
    print(f"hash         : {result.dataset_hash[:16]}...")
    print(f"duration     : {result.duration_seconds}s")
    print(
        f"samples      : {stats['samples']} "
        f"(train/val/test shares: "
        f"{stats['splits']['train']['sample_share']}/"
        f"{stats['splits']['val']['sample_share']}/"
        f"{stats['splits']['test']['sample_share']})"
    )
    print(f"chars/words  : {stats['chars']} / {stats['words']}")
    print(f"avg length   : {stats['avg_chars']} chars, {stats['avg_words']} words")
    print(f"tokens       : {stats['tokens']} ({result.tokenizer or 'not computed'})")
    print(f"rejected     : {result.rejected} (duplicates: {result.duplicate_count})")
    print(f"meta         : {result.meta_path}")
    print(json.dumps(result.stats["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
