"""Inspect and benchmark a from-scratch GPT configuration on this CPU.

Example:
    .venv\\Scripts\\python.exe scripts\\inspect_model.py --preset tiny-5m
    .venv\\Scripts\\python.exe scripts\\inspect_model.py --config configs\\models\\gpt-5m.json

The model produced here is RANDOMLY INITIALISED. Generated text is a smoke test
of the sampling pipeline, not evidence of training (that is Phase 3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import torch  # noqa: E402

from ml.models import (  # noqa: E402
    GPT,
    GenerationConfig,
    GPTConfig,
    count_parameters,
    estimate_memory,
    parameter_breakdown,
    preset,
)

DEFAULT_PRESET = "tiny-5m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset", default=None)
    source.add_argument("--config", default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed repetitions per attention path; the best run is reported to reduce noise",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def module_group(name: str) -> str:
    """Collapse parameter names into coarse groups for a readable table."""
    if name.startswith("token_embedding"):
        return "token_embedding"
    if name.startswith("position_embedding"):
        return "position_embedding"
    if name.startswith("blocks."):
        parts = name.split(".")
        return f"blocks.*.{parts[2]}"  # blocks.*.attn, blocks.*.mlp, ...
    return name.split(".")[0]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config = GPTConfig.from_dict(payload.get("config", payload))
        label = str(args.config)
    else:
        name = args.preset or DEFAULT_PRESET
        config = preset(name)
        label = f"preset:{name}"

    seq = args.seq or config.context_length
    model = GPT(config)
    params = count_parameters(model)

    groups: dict[str, int] = {}
    for row in parameter_breakdown(model):
        group = module_group(str(row["name"]))
        groups[group] = groups.get(group, 0) + int(row["numel"])

    print(f"config            : {label}")
    print(json.dumps(config.to_dict(), indent=2))
    print(f"parameters        : {params.summary()}")
    print(f"parameter groups  : {json.dumps(groups, indent=2)}")
    print(f"theoretical init loss: ln(vocab) = {math.log(config.vocab_size):.3f}")

    for checkpointing in (False, True):
        estimate = estimate_memory(
            config,
            batch_size=args.batch,
            seq_len=seq,
            total_parameters=params.total,
            activation_checkpointing=checkpointing,
        )
        print(
            f"memory estimate   : batch={args.batch} seq={seq} "
            f"checkpointing={checkpointing} -> {estimate['total_human']} "
            f"(weights {estimate['weights_human']}, grads {estimate['gradients_human']}, "
            f"opt {estimate['optimizer_human']}, act {estimate['activations_human']})"
        )

    # --- forward/backward timing on real tensors -----------------------------
    inputs = torch.randint(0, config.vocab_size, (args.batch, seq))
    targets = torch.randint(0, config.vocab_size, (args.batch, seq))
    loss_fn = torch.nn.CrossEntropyLoss()

    def benchmark(fused: bool) -> tuple[float, float, float]:
        """Time AdamW training steps with one attention implementation.

        A fresh model is built from the same seed for each run so both paths
        start from identical weights; a fresh optimiser avoids carrying AdamW
        moments from the other path. Several timed repetitions are taken and
        the best one is reported: this laptop is noisy (other processes, thermal
        variation), and the fastest run is the least contaminated estimate.
        """
        torch.manual_seed(args.seed)
        local_model = GPT(config)
        for block in local_model.blocks:
            block.attn.fused_attention = fused
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=3e-4)

        def step() -> float:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = local_model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            return loss.detach().item()

        step()  # warm-up (allocator, mask cache, thread pool)
        local_model.train()
        timings: list[float] = []
        loss = 0.0
        for _ in range(args.repeats):
            start = time.perf_counter()
            for _ in range(args.iters):
                loss = step()
            timings.append((time.perf_counter() - start) / args.iters * 1000)
        best = min(timings)
        tokens = args.batch * seq
        return best, tokens / (best / 1000), loss

    reference_ms, reference_tps, reference_loss = benchmark(fused=False)
    fused_ms, fused_tps, fused_loss = benchmark(fused=True)
    print(
        f"train step ref    : {reference_ms:.1f} ms/step, {reference_tps:,.0f} tokens/s, "
        f"loss {reference_loss:.3f}, {args.threads} threads"
    )
    print(
        f"train step fused  : {fused_ms:.1f} ms/step, {fused_tps:,.0f} tokens/s, "
        f"loss {fused_loss:.3f}, {reference_ms / fused_ms:.2f}x vs reference"
    )

    # --- untrained generation smoke test ------------------------------------
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    greedy = model.generate(prompt, GenerationConfig(max_new_tokens=16, greedy=True))
    sampled = model.generate(
        prompt, GenerationConfig(max_new_tokens=16, temperature=1.0, top_k=40, top_p=0.95, seed=7)
    )
    print(f"greedy tokens     : {greedy[0].tolist()}")
    print(f"sampled tokens    : {sampled[0].tolist()}")
    print(
        "note              : model is randomly initialised; token ids are meaningless until Phase 3"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
