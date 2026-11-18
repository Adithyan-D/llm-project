"""Train the from-scratch GPT on tokenized text, with resume and real logging.

Examples:
    # fresh run from a training config
    .venv\\Scripts\\python.exe scripts\\train_gpt.py ^
        --config configs\\training\\pretrain-gpt5m-shakespeare.json

    # continue the latest checkpoint in that run directory
    .venv\\Scripts\\python.exe scripts\\train_gpt.py ^
        --config configs\\training\\pretrain-gpt5m-shakespeare.json --resume latest

    # only generate from an existing checkpoint (no training)
    .venv\\Scripts\\python.exe scripts\\train_gpt.py ^
        --config configs\\training\\pretrain-gpt5m-shakespeare.json --resume latest --sample-only

The report written to <out_dir>/report.json contains only measured values: loss
curve points, tokens/s, wall time, parameter counts, checkpoint path and the
generated samples verbatim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import torch  # noqa: E402

from ml.models import GenerationConfig, count_parameters  # noqa: E402
from ml.tokenization import BPETokenizer  # noqa: E402
from ml.training import (  # noqa: E402
    Trainer,
    TrainingConfig,
    build_token_streams,
    resolve_checkpoint,
)

DEFAULT_PROMPTS = ("ROMEO:", "To be, or not to be")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training config JSON")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="checkpoint file, run directory, or 'latest' inside --out",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", default=None, help="override the run directory")
    parser.add_argument("--force", action="store_true", help="skip the RAM pre-flight check")
    parser.add_argument(
        "--sample-only", action="store_true", help="skip training; requires --resume"
    )
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    overrides: dict[str, object] = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        overrides["grad_accum_steps"] = args.grad_accum
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.warmup_steps is not None:
        overrides["warmup_steps"] = args.warmup_steps
    if args.eval_interval is not None:
        overrides["eval_interval"] = args.eval_interval
    if args.checkpoint_interval is not None:
        overrides["checkpoint_interval"] = args.checkpoint_interval
    if args.log_interval is not None:
        overrides["log_interval"] = args.log_interval
    if args.threads is not None:
        overrides["num_threads"] = args.threads
    if args.out is not None:
        overrides["out_dir"] = args.out
    if args.seed is not None:
        overrides["seed"] = args.seed
    return replace(config, **overrides)


def resolve_resume_arg(value: str, run_dir: Path) -> Path:
    if value == "latest":
        return run_dir
    return Path(value)


def generate_samples(
    trainer: Trainer,
    tokenizer: BPETokenizer,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    model = trainer.model
    was_training = model.training
    model.eval()
    eos_id = trainer.streams.eos_token_id
    samples: list[dict[str, object]] = []
    try:
        for index, prompt in enumerate(args.prompts):
            prompt_ids = tokenizer.encode(prompt)
            if eos_id is not None:
                prompt_ids = [eos_id, *prompt_ids]
            idx = torch.tensor([prompt_ids], dtype=torch.long)
            started = time.perf_counter()
            output = model.generate(
                idx,
                GenerationConfig(
                    max_new_tokens=args.sample_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    seed=(args.seed or 0) + index,
                    eos_token_id=eos_id,
                ),
            )
            seconds = time.perf_counter() - started
            new_tokens = int(output.size(1) - idx.size(1))
            text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            samples.append(
                {
                    "prompt": prompt,
                    "prompt_tokens": len(prompt_ids),
                    "new_tokens": new_tokens,
                    "seconds": round(seconds, 3),
                    "tokens_per_second": round(new_tokens / seconds, 2) if seconds > 0 else 0.0,
                    "text": text,
                }
            )
    finally:
        model.train(was_training)
    return samples


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_overrides(TrainingConfig.load(args.config), args)
    torch.set_num_threads(config.num_threads)

    streams = build_token_streams(config.dataset_dir, config.tokenizer_path)
    print(f"run dir       : {config.out_dir}")
    print(f"dataset       : {streams.manifest['dataset_id']}")
    for name, stream in streams.splits.items():
        print(f"{name:<13} : {stream.tokens:,} tokens")
    print(
        f"sampling      : {config.batch_size} x {config.model.context_length} "
        f"x accum {config.grad_accum_steps} = {config.tokens_per_step:,} tokens/step"
    )

    trainer = Trainer(config, streams=streams, verbose=True)
    params = count_parameters(trainer.model)
    estimate = trainer.check_memory(force=args.force)
    print(f"parameters    : {params.summary()}")
    print(
        f"memory budget : estimate {estimate['total_human']} "
        f"(activations {estimate['activations_human']})"
    )

    resume_path = None
    resume_checkpoint = None
    if args.resume is not None:
        resume_path = resolve_resume_arg(args.resume, Path(config.out_dir))
        # Resolve once, before training writes a newer ``latest.json``.
        resume_checkpoint = resolve_checkpoint(resume_path)
        print(f"checkpoint    : {resume_checkpoint}")

    if not args.sample_only:
        report = trainer.fit(
            max_steps=args.max_steps,
            resume_from=resume_checkpoint,
        )
    else:
        if resume_checkpoint is None:
            raise SystemExit("--sample-only requires --resume")
        trainer.resume(resume_checkpoint)
        report = trainer.report()

    tokenizer = BPETokenizer.load(config.tokenizer_path)
    samples = generate_samples(trainer, tokenizer, args)
    report["samples"] = samples
    report["resumed_from"] = None if resume_checkpoint is None else str(resume_checkpoint)

    report_path = Path(config.out_dir) / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.sample_only:
        print(
            f"done          : step {report['step']}, "
            f"{report['tokens_this_run']:,} tokens in {report['wall_seconds']:.1f}s "
            f"({report['tokens_per_second']:,.0f} tok/s)"
        )
        print(
            f"loss          : first step {report['first_step_loss']}, "
            f"first logged train {report['first_train_loss']}, "
            f"final train {report['final_train_loss']}, "
            f"final val {report['final_val_loss']}, best val {report['best_val_loss']}"
        )
    for sample in samples:
        print(f"--- prompt {sample['prompt']!r} ({sample['tokens_per_second']} tok/s)")
        print(sample["text"])
    print(f"report        : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
