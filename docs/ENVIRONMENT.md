# Environment Setup & Verified Baseline

Audited and verified on 2026-11-16. Re-run `scripts/check_env.py` after any change
to Python, drivers or dependencies, then update this file.

## 1. Machine

| Component | Value |
|---|---|
| OS | Windows 11 (build 26200) |
| CPU | Intel Core i7-1165G7, 4 physical / 8 logical cores |
| RAM | 16 GB (2x8 GB DDR4-3200), ~15.77 GiB usable |
| GPU | Intel Iris Xe iGPU (no CUDA, 1 GB shared) |
| Storage | 1 TB NVMe, C: 128.4 GiB free at audit |
| Python | 3.14.3 (system default) **and** 3.12.10 |
| Git | 2.53.0 |
| Node / npm | v24.14.1 / 11.19.1 |

## 2. Decisions

1. **Python 3.12.10 only.** 3.14 has no reliable wheels yet for the ML stack.
   Always invoke `.venv\Scripts\python.exe` (or `py -3.12`), never bare `python`
   (which resolves to 3.14).
2. **CPU-only PyTorch** (`2.15.0+cpu`) from `download.pytorch.org/whl/cpu`.
3. **No `bitsandbytes`** - no Windows/CPU support. Quantization path is
   `torch.ao` dynamic INT8 + GGUF/llama.cpp (Phase 6).
4. **All caches inside the project** so disk usage is auditable:
   `HF_HOME`, `TORCH_HOME`, `HF_DATASETS_CACHE` point into `model_registry/`.
   `scripts/check_env.py` sets them with `setdefault`, so an explicit override wins.
5. **Thread policy:** torch on Windows defaults to *physical* cores (4 here),
   which leaves ~40% throughput on the table. `check_env.apply_thread_policy()`
   defaults to all logical CPUs (8) unless `LLM_FOUNDRY_NUM_THREADS` is set.
   Measured effect is in §4.
6. `npm.ps1` is blocked by the PowerShell execution policy; use `npm.cmd`.

## 3. Setup (reproducible)

```powershell
py -3.12 -m venv C:\LLM_Platform\.venv
& C:\LLM_Platform\.venv\Scripts\python.exe -m pip install --upgrade pip
& C:\LLM_Platform\.venv\Scripts\python.exe -m pip install "torch==2.15.0+cpu" `
    --index-url https://download.pytorch.org/whl/cpu
& C:\LLM_Platform\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Verify:

```powershell
& C:\LLM_Platform\.venv\Scripts\python.exe scripts\check_env.py
& C:\LLM_Platform\.venv\Scripts\python.exe -m pytest
& C:\LLM_Platform\.venv\Scripts\python.exe -m ruff check .
```

Note: `torch` already pulls its own deps; installing `requirements.txt` afterwards
is idempotent and just confirms versions.

## 4. Measured baseline (2026-11-16)

`scripts/check_env.py`, torch 2.15.0+cpu, FP32, no other load:

| Measurement | 4 threads (torch default) | 8 threads (applied policy) |
|---|---|---|
| 2048x2048 FP32 matmul | 135.6 GFLOPS | **221.2 GFLOPS** |
| Provisional 5.29M GPT fwd+bwd step (8x128) | 753.2 ms | **447.0 ms (2291 tok/s)** |
| Process RSS after benchmark | - | 533 MiB |

Acceptance test suite: 11 passed in ~4 s (`backend/tests/test_env.py`).

### What this implies for Phase 2/3

- A 5M-parameter GPT step at batch 8 x 128 tokens takes ~0.45 s => ~2.3k tokens/s
  on CPU. A 10M-token "pretraining run" is therefore ~75 minutes at best; plan
  small corpora (~1-5M tokens) and short contexts (128) first.
- Available RAM at audit was only ~5.8 GiB (other applications running). Budget
  <= 4 GiB for training process, and add explicit memory guards in Phase 3.
- 221 GFLOPS is near the practical AVX2 peak of this 4-core part; there is no
  headroom left on CPU. Larger experiments must go to Colab/Kaggle.

## 5. Disk accounting

| Item | Size |
|---|---|
| `.venv` (torch + numpy/pandas + tooling) | 791 MiB |
| Project excluding `.venv` and `.git` | < 1 MiB |
| Projected full platform (plan §5) | < 12 GiB |

## 6. Known environment risks

- Bare `python` = 3.14 → import failures. Always use the venv interpreter.
- `torch.set_num_threads` must be called in every long-running process
  (API worker, training script); it is not persisted.
- Windows page file can mask RAM exhaustion and tank throughput; treat
  `ram_available_gb < 3` as an unsafe state for model loading.
