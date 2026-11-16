# LLM Foundry

A local-first LLM engineering platform: dataset pipeline, from-scratch BPE tokenizer,
from-scratch GPT transformer, pretraining, LoRA fine-tuning, evaluation, quantization,
CPU inference, FastAPI backend and a React dashboard - built to be reproducible and
measured, not simulated.

**Status:** Phase 0 complete (environment verified). Implementation plan and progress
tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Why this exists

The goal is to demonstrate the full LLM lifecycle end to end on constrained hardware:
an i7-1165G7 laptop with 16 GB RAM, no CUDA, plus free cloud GPUs where genuinely
needed. Everything from the tokenizer and attention blocks to the training loop is
implemented and tested in this repository rather than wrapped around an existing model.

## Hardware baseline

| | |
|---|---|
| CPU | i7-1165G7, 4C/8T |
| RAM | 16 GB (no GPU acceleration) |
| Local training target | 1-5M parameter GPT, short context |
| Fine-tuning | small pretrained models (LoRA), cloud GPU preferred |
| Inference | CPU, GGUF / llama.cpp + PyTorch |

## Layout

```
backend/    FastAPI app (app/) and ML code (ml/) + tests
frontend/   React + TypeScript + Vite + Tailwind dashboard
configs/    Training / experiment configs
datasets/   raw and processed data (gitignored)
checkpoints experiments model_registry   artifacts (gitignored)
scripts/    check_env.py and per-phase CLIs
docs/       environment, architecture, benchmarks
```

## Quickstart (Phase 0)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

.venv\Scripts\python.exe scripts\check_env.py
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

Detailed setup, measurements and constraints: [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Rules this project follows

- No fabricated metrics, no hardcoded results, no claiming downloads were trainings.
- Every completed item in the plan is backed by a test or a measured benchmark.
- Large data, model weights and artifacts never enter Git.

## License

TBD.
