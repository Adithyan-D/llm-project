# LLM Foundry — Implementation Plan & Tracking Log

> Single source of truth for engineering progress. `README.md` will be user-facing;
> this file is engineering-facing. Update it in the same session as the work is done.

Last updated: 2026-11-18 — **Phase 2 complete and verified**. Current focus: Phase 3
(pretraining engine).

---

## 0. How To Use This File

1. Pick the next unchecked item in §7 (Phase Roadmap).
2. Set it to `[~]` while working, `[x]` only after verification.
3. Record the proof (command, test name, measured number) in §9 or §8.
4. Add a row to §10 (Session Log) and a row to §8 for every measured number.
5. Any decision that changes the architecture goes into §2 (ADR-lite) with a date.
6. **Phase close routine** (mandatory, every phase):
   `pytest` green → `ruff check .` and `ruff format --check .` clean →
   commit with a conventional message (`feat(phaseN): …`, `test(phaseN): …`,
   `docs: …`, `fix: …`, `chore: …`) → `git push` to `main` →
   record the pushed commit hash in §10.

### Repository

| | |
|---|---|
| Remote | `origin` = https://github.com/Adithyan-D/llm-project |
| Branch | `main` (tracks `origin/main`) |
| License | MIT (`LICENSE`, added at repo creation) |
| Auth | Git Credential Manager (`manager`), user already signed in |

### Status Legend

| Symbol | Meaning |
|---|---|
| `[ ]` | Todo |
| `[~]` | In progress |
| `[x]` | Done **and verified** (test/benchmark recorded) |
| `[!]` | Blocked (reason required) |
| `[-]` | Dropped (reason required) |

Rule: nothing is marked `[x]` without a command, test or benchmark that proves it.

---

## 1. Project Objective

Build **LLM Foundry**: a local-first, open-source LLM engineering platform that
covers the full lifecycle — data → tokenization → from-scratch transformer →
pretraining → fine-tuning (LoRA) → evaluation → quantization → optimized
inference → API → dashboard — with reproducible, measured results.

Non-goals: building the largest model, UI-only demos, API wrappers around hosted LLMs,
hardcoded/fabricated metrics.

---

## 2. Verified Environment (audit 2026-11-16)

| Item | Value | Implication |
|---|---|---|
| OS / Shell | Windows 11, PowerShell 5.1 | `py -3.12`, `npm.cmd` |
| Python | 3.14.3 (default), 3.12.10 (available) | **Use 3.12 venv** — 3.14 lacks ML wheels |
| PyTorch | 2.15.0+cpu in `.venv` (installed Phase 0) | CPU-only, 8-thread policy |
| Git | 2.53.0 | OK |
| Node / npm | v24.14.1 / 11.19.1 | `npm.ps1` blocked by ExecutionPolicy → use `npm.cmd` |
| CPU | i7-1165G7, 4C / 8T @ 2.8 GHz | 8 threads, CPU-only training |
| RAM | 16 GB (2×8 GB DDR4-3200) | max one large model loaded at a time |
| GPU | Intel Iris Xe iGPU, 1 GB shared, driver 31.0.101.4502 | no CUDA; CPU baseline, OpenVINO optional |
| Disk (C:) | 129.7 GB free at audit / 128.4 GB free after P0 | keep all caches inside `C:\LLM_Platform` |
| Workspace | `C:\LLM_Platform` (Phase 0 scaffolding in place) | clean start |

**Measured on this machine (2026-11-16, full detail in `docs/ENVIRONMENT.md`):**
8 threads = 221 GFLOPS FP32 matmul; provisional 5.29M GPT fwd+bwd = 447 ms/step
(2291 tok/s) at batch 8 × ctx 128; pytest 11/11; `.venv` = 791 MiB.

**Hard constraint:** no reads/writes outside `C:\LLM_Platform` without explicit
permission. `HF_HOME`, `TORCH_HOME`, MLflow store, model cache all live inside the project.

---

## 3. Architecture Decisions (ADR-lite)

| # | Decision | Why | Rejected alternative |
|---|---|---|---|
| A1 | Python 3.12 venv at `.venv` | torch/transformers/peft/mlflow wheel coverage | Python 3.14 (immature ML wheels) |
| A2 | torch CPU build from `download.pytorch.org/whl/cpu` | no CUDA hardware; smaller download | default PyPI wheel (pulls CUDA deps) |
| A3 | Skip `bitsandbytes` | no Windows/CPU support | bnb 4/8-bit (requires NVIDIA) |
| A4 | llama.cpp via official prebuilt Windows binaries + subprocess/HTTP | no MSVC/CMake build risk | `llama-cpp-python` source build |
| A5 | MLflow local file store + SQLite (Phase 5) | zero infra, low RAM | MLflow server / hosted tracking |
| A6 | BPE, attention, transformer, training loop, generation, metrics from scratch | core learning objective | wrapping HF implementations |
| A7 | HF libs only from Phase 4+ (pretrained/LoRA/datasets) | avoid premature deps | install everything up front |
| A8 | OpenVINO / Iris Xe acceleration = optional Phase 6 experiment | extra conversion complexity | hard dependency on iGPU |
| A9 | SQLite + SQLAlchemy backend, no auth | local single-user platform | Postgres / OAuth |
| A10 | All caches under `model_registry/` | disk accounting, repo hygiene | user-global `~/.cache` |
| A11 | From-scratch components expose tests before integration | "test each subsystem before integrating" | end-to-end-only testing |
| A12 | Thread policy: all logical CPUs (8) unless `LLM_FOUNDRY_NUM_THREADS` set | measured +63% matmul, +69% step throughput vs torch's 4-thread default | accept torch default |
| A13 | `torch.set_num_threads` must be applied in every long process (scripts, API worker) | setting is not persisted across processes | rely on default |
| A14 | Byte-level BPE (256 byte base tokens) instead of word-level | lossless, no `<unk>`, arbitrary text incl. code/emoji, predictable `256+merges` budget | word-level BPE with `<unk>`, SentencePiece wrapper |
| A15 | Hash-based split assignment instead of shuffle-and-slice | streaming, order-independent, reproducible per seed | buffer + shuffle (needs full corpus in RAM) |
| A16 | Exact-match dedup only | O(1) per record, ~50 B/sample | MinHash/SimHash near-dup (deferred) |
| A17 | Token counts only reported when a real tokenizer is supplied | avoids fake "approx tokens = chars/4" numbers | estimated token counts |
| A18 | Pre-LN transformer blocks (normalise before sublayer) | stable training without LR warmup; post-LN is fragile at tiny batches | post-LN (original 2017 layout) |
| A19 | Learned positional embeddings, not RoPE | at `T ≤ 128` the extra `T·d` params are cheap; RoPE pays off only beyond the trained window | RoPE / ALiBi (deferred until context grows) |
| A20 | Handwritten causal SDPA is the default; fused `F.scaled_dot_product_attention` is opt-in | reference implementation is the learning objective; fused measured only 7-9% faster at 5M and slower at 2M on this CPU | making the fused kernel the default |
| A21 | Causal mask memoised per `(T, device)` | 81.3 µs → 6.9 µs per mask build, six builds per step | rebuilding the mask every call |
| A22 | No KV cache in Phase 2 generation | 128-token contexts make recompute acceptable; caching belongs with the inference engine | premature cache complexity |
| A23 | Memory estimator is analytic and conservative, always paired with real measurements | avoids presenting a projection as a measurement | reporting estimates as measured RSS |
| A24 | Pack documents into one token stream with `<|endoftext|>` separators and non-overlapping windows | zero padding tokens trained on; every token seen once per epoch; EOS makes boundaries visible | padding per document (wasted compute), sliding windows with overlap (duplicates) |
| A25 | Epoch shuffle derived from `(seed + epoch)`; checkpoint stores only `BatchCursor(epoch, batch_index)` | resume is exact without serialising a 3.4k-entry permutation; no permutation drift across NumPy versions in the checkpoint | storing the permutation (larger, version-coupled checkpoints) |
| A26 | Resuming preserves the checkpoint's LR horizon even if `max_steps` grows | prevents an artificial metric jump from remapping the cosine curve; extending a run means "more steps at the floor", not "restart the schedule" | recomputing the schedule for the new horizon |
| A27 | Analytic RAM pre-flight (70% of available) plus OOM-marker translation | CPU allocator errors are generic and can leave the process unusable; a clear refusal is cheaper than a half-written run | hoping the allocator errors cleanly |
| A28 | Atomic checkpoints (temp file + `os.replace`) and a `latest.json` pointer | a crash mid-write cannot corrupt the last good checkpoint; resume path is one argument | in-place `torch.save` (truncated files on crash) |

---

## 4. Target Directory Structure

```
LLM_Platform/
├── backend/
│   ├── app/
│   │   ├── api/          # routers: datasets, tokenizer, models, training, eval, inference
│   │   ├── core/         # settings, paths, logging, device/thread policy
│   │   ├── services/     # orchestration between API and ml/
│   │   ├── schemas/      # Pydantic request/response models
│   │   ├── db/           # SQLAlchemy models + session
│   │   └── main.py
│   ├── ml/
│   │   ├── data/         # loading, validation, dedup, normalization, splitting, stats
│   │   ├── tokenization/ # BPE train/encode/decode/save/load + HF comparison
│   │   ├── models/       # attention, blocks, GPT, generation, param/memory estimation
│   │   ├── training/     # dataset packing, batching, loop, scheduler, ckpt/resume
│   │   ├── finetuning/   # HF model mgmt, LoRA config/train/save/load
│   │   ├── evaluation/   # loss, ppl, latency, throughput, memory, generation eval
│   │   ├── quantization/ # torch.ao dynamic INT8, GGUF conversion/benchmark
│   │   └── inference/    # local engine, generation params, quantized backends
│   ├── tests/
│   └── requirements.txt
├── frontend/             # Vite + React + TS + Tailwind: pages/ components/ services/ hooks/
├── configs/              # YAML experiment/training configs
├── datasets/             # raw/ processed/ (gitignored)
├── checkpoints/          # (gitignored)
├── experiments/          # run metadata, metrics JSON, adapters (gitignored)
├── model_registry/       # hf_cache/ gguf/ metadata (gitignored)
├── notebooks/            # cloud (Colab/Kaggle) notebooks
├── scripts/              # check_env.py, train_tokenizer.py, train_gpt.py, convert_gguf.py
├── docs/                 # ENVIRONMENT.md, ARCHITECTURE.md, BENCHMARKS.md
├── .venv/                # (gitignored)
├── .gitignore
├── IMPLEMENTATION_PLAN.md
└── README.md
```

---

## 5. Resource Budget (projected, C: has ~128.4 GB free after P0)

| Module | Disk | Peak RAM |
|---|---|---|
| venv (P0) | **791 MiB measured** (~0.8 GB) | — |
| tokenizer + processed data | **1.4 MB measured** (1.37 MB dataset + 33.5 KB tokenizer) | ~200 MB peak |
| 1–5M GPT checkpoints | <1 GB | 2–3 GB (533 MiB measured for 5.29M) |
| HF pretrained (360M + 0.5B) | ~2 GB | 3–6 GB |
| LoRA adapters | <50 MB each | +1–2 GB |
| GGUF quantizations (2–3) | ~2 GB | 1–2 GB |
| backend + node_modules | ~1 GB | ~1 GB |
| **Total (projected)** | **< 12 GB** | keep < 12 GB |

Note: available (not total) RAM at audit was 5.6-5.8 GiB with normal desktop load.
Phase 3 must therefore guard against OOM rather than assume 16 GB is free.

---

## 6. Local / Cloud Execution Split

| Work | Local CPU | Free cloud GPU (Colab/Kaggle) |
|---|---|---|
| Dataset + tokenizer + BPE | yes | — |
| Transformer from scratch (1–5M) | yes | — |
| GPT pretraining | yes (small) | optional larger |
| HF model loading + inference | yes | — |
| LoRA on 360M / 0.5B | yes (reduced config, slow) | **primary** |
| LoRA on 1.5B | no | yes |
| Evaluation + benchmarks | yes | GPU comparison rows |
| Quantization / GGUF / CPU inference | yes | — |
| API + dashboard | yes | — |

Cloud rule: every cloud run must export a metrics JSON + adapter + environment record
into `experiments/` so results are reproducible and locally inspectable. All cloud
training must be checkpoint-resumable.

---

## 7. Phase Roadmap

### Phase 0 — Environment & Architecture `[x]`
- [x] `git init` + `.gitignore` (venv, caches, checkpoints, datasets, models, artifacts)
- [x] Skeleton directories with `.gitkeep`
- [x] `py -3.12 -m venv .venv`
- [x] `backend/requirements.txt` — Phase 0 subset only: torch (CPU), numpy, pandas, pytest, ruff
- [x] `scripts/check_env.py` (python/torch versions, thread count, RAM, disk, cache paths)
- [x] Micro-benchmark: matmul GFLOPS + 5M-param GPT fwd/bwd step time
- [x] `backend/tests/test_env.py`
- [x] `pytest` and `ruff check` green
- [x] `docs/ENVIRONMENT.md`, `README.md`
- [x] First commit (clean history starts here)

**Acceptance criteria:** `import torch` succeeds inside `.venv` on Python 3.12;
8 threads detected; benchmark numbers recorded in §8; `pytest` + `ruff` pass;
measured disk delta logged.

**Result (2026-11-16):** met. 11 tests pass, ruff clean; 221 GFLOPS / 447 ms per
5.29M GPT step at 8 threads; `.venv` = 791 MiB. See `docs/ENVIRONMENT.md` §4.

### Phase 1 — Dataset & Tokenizer `[x]`
- [x] Loaders for TXT / JSON / JSONL with schema validation (`ml/data/loaders.py`)
- [x] Malformed-record quarantine, dedup, text normalization (`ml/data/preprocess.py`)
- [x] Chunked/streaming preprocessing (no full-file RAM load; JSON arrays documented as non-streaming)
- [x] Statistics: samples, words, exact tokens (when tokenizer given), avg length, size, split distribution (`ml/data/stats.py`)
- [x] Train/val/test hash-based splitting with saved metadata + dataset version hash
- [x] Custom BPE: incremental pair counting, merges, vocab build, encode, decode, byte-level (no `<unk>` needed)
- [x] Tokenizer save/load (merges + config + specials; vocab rebuilt and validated)
- [x] Tokenization visualization (text → tokens → IDs) — `scripts/tokenize_demo.py`
- [x] Benchmark + comparison against an HF tokenizer (GPT-2; see §8)
- [x] Tests: round-trip encode/decode, unicode/edge text, vocab growth, reload equality, corrupt-file rejection
- [x] Corpus pinning: `scripts/download_corpus.py` with SHA-256 verification
- [x] CLI scripts: prepare_dataset, train_tokenizer, tokenize_demo, bench_tokenizer

**Acceptance:** met. Tiny Shakespeare (1.12 MB) → 25,721 kept samples; BPE-1024 trained
in 2.85 s (768 merges + 256 bytes + 1 special), exact 524,233 tokens, lossless
round-trip. 59 tests pass. Details: `docs/DATASET_AND_TOKENIZER.md`.

### Phase 2 — Transformer From Scratch `[x]`
- [x] Token embeddings + positional embeddings
- [x] Scaled dot-product attention with causal mask
- [x] Multi-head attention (Q/K/V projections, head split/merge)
- [x] MLP/feed-forward block, LayerNorm, residual connections
- [x] Transformer block stack + LM head (weight tying optional, documented)
- [x] Autoregressive generation (temperature, top-k, top-p, repetition penalty)
- [x] Parameter counter + activation/memory estimator
- [x] Configurable vocab size, d_model, heads, layers, context length, dropout
- [x] Tests: shapes, causal leakage, gradient flow, determinism under seed

**Acceptance:** met. `tiny-5m` = 5,034,240 params (262,400 tied), 438.8 MiB
analytic training estimate (143.8 MiB with checkpointing), batch 8 × 128 AdamW
step 757-803 ms/step reference / 695-748 ms/step fused (best of 3 × 5 steps).
115 tests pass; ruff clean. Generated token IDs come from a randomly initialised
model and are explicitly not a quality claim. Details: `docs/TRANSFORMER.md`.

### Phase 3 — Pretraining Engine `[x]`
- [x] Tokenized dataset + input/target pair construction, batching, packing
- [x] Training loop: CE loss, backprop, AdamW, LR schedule, grad clip, grad accumulation
- [x] Validation loop
- [x] Checkpoint save/resume (model, optimizer, scheduler, step, RNG state)
- [x] OOM safeguards (batch/seq auto-guard, clear error messages)
- [x] Training logs: loss, val loss, LR, epoch, duration, tokens processed
- [x] Actually train a small GPT and record real curves

**Acceptance:** met. EOS-separated packed streams (train 440,076 tokens / 20,607 docs,
val 55,491 / 2,580; cache built in 2.1 s), non-overlapping windows, deterministic
`(seed + epoch)` shuffle with a 2-integer cursor. Real `tiny-5m` run: 400 steps x 1,024
tokens = 409,600 tokens in 326.7 s (1,253.7 tok/s, 93.2% of the 429-step epoch); val
loss 3.9425 -> 3.5209 across 8 evaluations (init reference ln(1025)=6.932); checkpoints
60,517,397 B; 144 tests pass; ruff clean. Resume proof: step 400 -> 405 kept the
checkpoint's LR horizon (6e-5), val 3.520929 -> 3.520660 with no jump, and a repeat
resume reproduced 3.520660 exactly. Samples are word salad, as expected at 5M params
and 0.93 epoch; this is stated plainly rather than presented as coherent text.
Details: `docs/TRAINING.md`.

### Phase 4 — Pretrained Models & LoRA `[ ]`
- [ ] HF cache config inside `model_registry/hf_cache`
- [ ] Model selection/download guard (size + disk check before download)
- [ ] Config inspection, parameter counting, size report, tokenizer load, inference smoke test
- [ ] Start with SmolLM2-360M and Qwen2.5-0.5B (1.5B via cloud only)
- [ ] LoRA: target-module discovery, r/alpha/dropout, frozen base, trainable-param report
- [ ] Train LoRA locally (reduced config) and on free GPU (primary)
- [ ] Save/reload adapters; base vs tuned side-by-side evaluation
- [ ] Record execution device/environment per experiment

**Acceptance:** real adapter trained, reloaded, and shown to change outputs; trainable
parameter percentage reported; cloud run exported into `experiments/`.

### Phase 5 — Experiment Tracking & Evaluation `[ ]`
- [ ] MLflow (local file store + SQLite) integration from this phase onward
- [ ] Run registry: unique IDs, config, dataset/tokenizer versions, env, device, duration
- [ ] Loss curve storage + comparison data for UI
- [ ] Metrics: train/val loss, cross-entropy, perplexity, tokens/s, latency, memory
- [ ] Consistent prompt set + fixed generation params for model comparisons
- [ ] Explicit separation of quantitative metrics vs subjective quality notes

**Acceptance:** two distinct runs comparable from stored artifacts; metrics recomputed
from checkpoints match logged values.

### Phase 6 — Quantization & Optimized Inference `[ ]`
- [ ] FP32 vs FP16 (where valid) vs dynamic INT8 (`torch.ao.quantization`)
- [ ] GGUF conversion + llama.cpp CPU inference (prebuilt binaries)
- [ ] Measure: file size, RAM, load time, latency, tokens/s, qualitative output
- [ ] Document which techniques are unavailable on this CPU and why
- [ ] Optional: OpenVINO / Iris Xe experiment, results recorded even if negative

**Acceptance:** at least one quantized path benchmarked against FP32 baseline with a
trade-off table; failures documented honestly.

### Phase 7 — Backend & Dashboard `[ ]`
- [ ] FastAPI app skeleton, settings, logging, error handling, Pydantic validation
- [ ] SQLite + SQLAlchemy models (models, experiments, datasets, tokenizers, runs)
- [ ] Endpoints: model list/info, generate, load/unload, experiments, evaluation, datasets, system info
- [ ] Service layer bridging API ↔ `backend/ml`
- [ ] React + TS + Vite + Tailwind dashboard, 12 sections (Overview → System Information)
- [ ] Training Studio with live progress; Experiment comparison charts (Plotly)
- [ ] Inference Playground with latency/tokens-per-second display
- [ ] No simulated ML results anywhere in UI

**Acceptance:** UI performs a real generation against a locally loaded model and shows
real measured numbers; API inputs validated; errors surfaced clearly.

### Phase 8 — Finalization `[ ]`
- [ ] Integration tests across API + ML services
- [ ] Performance pass and fixes
- [ ] Docs: architecture diagrams, limitations, hardware benchmarks, install guide
- [ ] Example experiments + reproducible configs
- [ ] Professional README (what was built, what was trained here, what was downloaded)
- [ ] Clean Git history and final review

**Acceptance:** a reviewer can reproduce at least one training run, one LoRA run, and
one benchmark from committed configs + docs.

---

## 8. Benchmark & Results Log

Only measured values — never estimated or invented. Include hardware + command.

| Date | What | Config | Result | Command |
|---|---|---|---|---|
| 2026-11-16 | FP32 matmul, torch 2.15.0+cpu | 2048x2048, 10 iters, 4 threads (torch default) | 135.6 GFLOPS | `.venv\Scripts\python.exe scripts\check_env.py` (first run) |
| 2026-11-16 | FP32 matmul, torch 2.15.0+cpu | 2048x2048, 10 iters, 8 threads | 221.2 GFLOPS | same |
| 2026-11-16 | Provisional 5.29M GPT fwd+bwd | batch 8 x 128 ctx, AdamW, 4 threads | 753.2 ms/step, 1359 tok/s | same |
| 2026-11-16 | Provisional 5.29M GPT fwd+bwd | batch 8 x 128 ctx, AdamW, 8 threads | 447.0 ms/step, 2291 tok/s | same |
| 2026-11-16 | Process RSS after benchmark | torch + 5.29M model loaded | 533 MiB | same |
| 2026-11-16 | Test suite | pytest 9.2.0 | 11 passed in 4.09 s | `.venv\Scripts\python.exe -m pytest` |
| 2026-11-16 | Disk: `.venv` size | torch 2.15.0+cpu + numpy/pandas/pytest/ruff | 791 MiB | `Get-ChildItem .venv -Recurse \| Measure-Object Length -Sum` |
| 2026-11-17 | Dataset preprocessing | Tiny Shakespeare 1,115,394 B, line mode, NFKC, dedup, 80/10/10, seed 42 | 25,721 kept / 7,056 dup removed; 1,003,675 chars, 192,862 words; 2.1 s | `.venv\Scripts\python.exe scripts\prepare_dataset.py ...` |
| 2026-11-17 | Split shares achieved | 80/10/10 target, hash split | train 0.8012 / val 0.1003 / test 0.0985 | same (meta.json) |
| 2026-11-17 | BPE training | vocab_size 1024, min_frequency 2, train split (20,607 docs) | 768 merges + 256 bytes + 1 special = 1025; 2.85 s | `.venv\Scripts\python.exe scripts\train_tokenizer.py ...` |
| 2026-11-17 | Exact token count | dataset + BPE-1024 | 524,233 tokens, 1.915 chars/token | `prepare_dataset.py --tokenizer ...` |
| 2026-11-17 | Tokenizer compression (held-out) | 500 test docs, 18,932 chars | local 1.91 chars/token vs HF gpt2 3.82 | `.venv\Scripts\python.exe scripts\bench_tokenizer.py ...` |
| 2026-11-17 | Tokenizer encode throughput | 500 test docs, single thread | local 1.37 M chars/s vs HF gpt2 2.63 M chars/s (Rust) | same |
| 2026-11-17 | Tokenizer round-trip | all splits + unicode/emoji/ANSI/empty | lossless, verified by tests | `pytest backend/tests/test_tokenizer.py` |
| 2026-11-17 | Test suite | pytest 9.2.0 | 59 passed in 9.25 s | `.venv\Scripts\python.exe -m pytest` |
| 2026-11-17 | Artifact sizes | dataset + tokenizer | 1.37 MB processed, 33.5 KB tokenizer | `Get-ChildItem ... \| Measure-Object Length -Sum` |
| 2026-11-18 | GPT parameter count | `tiny-5m`: d=256, H=4, N=6, T=128, V=1025 | 5,034,240 params (262,400 tied) | `.venv\Scripts\python.exe scripts\inspect_model.py --config configs\models\gpt-5m.json` |
| 2026-11-18 | GPT parameter count | `nano-2m`: d=192, H=4, N=4, T=128, V=1025 | 2,001,216 params (196,800 tied) | same, `configs\models\gpt-2m.json` |
| 2026-11-18 | Analytic training memory | `tiny-5m`, batch 8 x 128, AdamW, FP32 | 438.8 MiB (act 362.0; weights 19.2, grads 19.2, opt 38.4); checkpointing model 143.8 MiB | same |
| 2026-11-18 | Init-loss sanity | `tiny-5m`, V=1025 | ln(V)=6.932 printed; 3.954 after 15 AdamW steps on one fixed random batch | same |
| 2026-11-18 | Train step throughput | `tiny-5m`, batch 8 x 128, 8 threads, best of 3 x 5 steps, two runs | reference 756.9-802.8 ms/step (1,276-1,353 tok/s); fused SDPA 694.6-747.7 ms/step (1,370-1,474 tok/s) | `.venv\Scripts\python.exe scripts\inspect_model.py --config configs\models\gpt-5m.json --iters 5 --repeats 3` |
| 2026-11-18 | Train step throughput | `nano-2m`, same protocol | reference 322.7 ms/step (3,173 tok/s); fused 330.8 ms/step (3,096 tok/s) | same, `configs\models\gpt-2m.json` |
| 2026-11-18 | Causal mask cost | 128 x 128 boolean mask, 20,000 calls, 8 threads | build 81.28 us/call; cached 6.94 us/call | inline timing loop, recorded in `docs/TRANSFORMER.md` §2 |
| 2026-11-18 | Test suite | pytest 9.2.0 | 115 passed (13 attention, 16 model, 16 generation, 11 estimation, plus P0/P1) | `.venv\Scripts\python.exe -m pytest` |
| 2026-11-18 | Token stream build | Tiny Shakespeare `971a2b0d5ff5` + BPE-1024, EOS 1024 | train 440,076 tokens / 20,607 docs; val 55,491 / 2,580; 2.1 s cold build | `.venv\Scripts\python.exe scripts\prepare_training_data.py ...` |
| 2026-11-18 | Packing geometry | `tiny-5m`, T=128, batch 8 | 3,437 train windows -> 429 steps/epoch, 439,296 tokens/epoch (5 windows dropped) | token stream manifest + `PackedSampler` |
| 2026-11-18 | Real pretraining throughput | 400 steps x 8 x 128, AdamW, 8 threads, GradClip 1.0 | 409,600 tokens in 326.7 s = 1,253.7 tok/s (~0.82 s/step) | `.venv\Scripts\python.exe scripts\train_gpt.py --config configs\training\pretrain-gpt5m-shakespeare.json` |
| 2026-11-18 | Real val-loss curve | same run, 10 fixed val batches every 50 steps | 3.9425 @50, 3.7263 @100, 3.6617 @150, 3.6178 @200, 3.5827 @250, 3.5540 @300, 3.5380 @350, 3.5209 @400 | same (report.json) |
| 2026-11-18 | Real train-loss anchors | same run | ln(1025)=6.932 init reference; first logged window (steps 1-20) 5.377; final step 3.543 | same |
| 2026-11-18 | Checkpoint size | 5,034,240 params + AdamW m/v + RNG + history | 60,517,397 B (step 200); 60,518,549 B (step 405) | `Get-ChildItem checkpoints\shakespeare-gpt5m` |
| 2026-11-18 | Resume continuity | step 400 -> 405, schedule horizon kept at 400 | 5,120 tokens in 6.7 s (769 tok/s); val 3.520929 -> 3.520660; repeat run reproduced 3.520660 exactly | `train_gpt.py --resume ...\step-000400.pt --max-steps 405` |
| 2026-11-18 | Trained-model generation | temperature 0.8, top-k 40, top-p 0.95, no KV cache | 22 tokens @ 30.3 tok/s; 34 tokens @ 31.7 tok/s | same run (report.json `samples`) |
| 2026-11-18 | Test suite | pytest 9.2.0 | 144 passed (29 training + 29 tokenizer + rest of P0/P1/P2) | `.venv\Scripts\python.exe -m pytest` |

---

## 9. Verification Log

Proof-of-completion entries (test names, outputs, file hashes).

| Date | Phase | Claim | Proof |
|---|---|---|---|
| 2026-11-16 | P0 | Python 3.12 venv with CPU torch works | `test_python_runtime_is_312`, `test_torch_is_cpu_build` pass |
| 2026-11-16 | P0 | Thread policy applies all logical CPUs + env override | `test_thread_policy_applies_all_logical_cpus`, `test_thread_policy_respects_env_override` |
| 2026-11-16 | P0 | All caches stay inside `C:\LLM_Platform` | `test_cache_paths_stay_inside_project` |
| 2026-11-16 | P0 | Gradients flow through LayerNorm+Linear | `test_gradient_flow_through_layernorm_and_linear` |
| 2026-11-16 | P0 | Benchmark model is in 1-20M param range | `test_provisional_benchmark_model_is_in_target_range` |
| 2026-11-16 | P0 | Lint/format clean | `ruff check .` → All checks passed; `ruff format --check .` → 19 files formatted |
| 2026-11-17 | P1 | Loaders stream and quarantine bad records | `test_jsonl_collects_structural_issues_but_keeps_valid_records`, `test_txt_line_mode_skips_blank_lines` |
| 2026-11-17 | P1 | Splits + dataset hash reproducible across runs | `test_split_assignment_is_deterministic`, `test_seed_changes_split_assignment` |
| 2026-11-17 | P1 | Stats match actual split contents | `test_stats_match_split_contents` |
| 2026-11-17 | P1 | Exact token counts equal recount with tokenizer | `test_exact_token_stats_with_tokenizer` |
| 2026-11-17 | P1 | BPE lossless on unicode/emoji/ANSI/empty | `test_encode_decode_roundtrip` (9 parametrised cases) |
| 2026-11-17 | P1 | Save/load preserves behaviour & rejects corruption | `test_save_and_reload_preserves_behaviour`, `test_load_rejects_corrupt_merges`, `test_load_rejects_special_token_collision` |
| 2026-11-17 | P1 | Training deterministic, min_frequency early stop | `test_training_is_deterministic`, `test_min_frequency_stops_merging` |
| 2026-11-17 | P1 | Real pipeline on Tiny Shakespeare | dataset `tinyshakespeare-971a2b0d5ff5`, 524,233 tokens, round-trip true |
| 2026-11-18 | P2 | Attention equals an explicit per-position reference | `test_attention_matches_naive_reference` |
| 2026-11-18 | P2 | Causal masking blocks future-token leakage (attention + full model) | `test_causal_attention_ignores_future_tokens`, `test_logits_cannot_see_future_tokens` |
| 2026-11-18 | P2 | Mask cache returns identical tensor without rebuilding | `test_cached_mask_is_reused_and_matches_fresh_mask` |
| 2026-11-18 | P2 | Fused SDPA and handwritten path agree in value and gradient | `test_fused_attention_matches_reference_path`, `test_fused_attention_gradients_match_reference` |
| 2026-11-18 | P2 | Gradients reach every trainable parameter, all finite | `test_gradients_flow_to_all_trainable_parameters`, `test_mha_shapes_and_gradients` |
| 2026-11-18 | P2 | Seed determinism; dropout stochastic only in train mode | `test_initialisation_is_seed_deterministic`, `test_eval_mode_is_deterministic_with_dropout`, `test_training_mode_dropout_is_stochastic` |
| 2026-11-18 | P2 | Analytic parameter formula matches built model (tied/untied, bias/no-bias) | `test_weight_tying_shares_storage`, `test_untied_head_adds_vocab_projection`, `test_no_bias_variant_counts_match` |
| 2026-11-18 | P2 | Generation controls behave as specified and are reproducible | `test_temperature_zero_matches_greedy`, `test_top_k_one_is_greedy`, `test_sampling_is_seed_reproducible`, `test_repetition_penalty_matches_manual_math` |
| 2026-11-18 | P2 | Memory estimate components sum and scale monotonically | `test_memory_estimate_includes_all_components`, `test_memory_scales_with_batch_and_sequence` |
| 2026-11-18 | P2 | Lint/format clean at phase close | `ruff check .` -> All checks passed; `ruff format --check .` -> all files formatted |
| 2026-11-18 | P3 | Token cache equals a fresh tokenizer encode with EOS after each doc | `test_token_stream_appends_eos_and_matches_tokenizer` |
| 2026-11-18 | P3 | Cache is reused without re-encoding; a changed tokenizer gets its own cache | `test_token_stream_cache_is_reused_without_reencoding`, `test_changing_the_tokenizer_builds_a_separate_cache` |
| 2026-11-18 | P3 | Windows are non-overlapping and targets are inputs shifted by one | `test_sampler_shapes_and_shifted_targets` |
| 2026-11-18 | P3 | Epoch order is deterministic per seed, differs across epochs, covers every window once | `test_sampler_epoch_is_deterministic_and_shuffled`, `test_sampler_covers_each_window_once_per_epoch` |
| 2026-11-18 | P3 | Gradient accumulation equals a single full-batch update | `test_gradient_accumulation_matches_full_batch` |
| 2026-11-18 | P3 | Resume reproduces an uninterrupted run's parameters and tokens exactly | `test_resume_matches_uninterrupted_run` |
| 2026-11-18 | P3 | Extending a resumed run keeps the checkpoint's LR horizon | `test_resume_keeps_checkpoint_schedule_horizon` |
| 2026-11-18 | P3 | Checkpoint restores model, AdamW state, cursor and all three RNGs | `test_checkpoint_roundtrip_restores_model_optimizer_and_rng`, `test_random_module_state_is_restored_by_checkpoint` |
| 2026-11-18 | P3 | `latest.json` resolves from the CWD and from a moved run directory | `test_latest_pointer_resolves_paths_relative_to_the_working_directory`, `test_latest_pointer_falls_back_to_the_run_directory` |
| 2026-11-18 | P3 | Mismatched architecture/format is rejected instead of loaded | `test_checkpoint_rejects_mismatched_model_config`, `test_checkpoint_rejects_unknown_format` |
| 2026-11-18 | P3 | RAM pre-flight refuses over-budget configs; allocator OOM is translated | `test_memory_budget_guard`, `test_oom_during_step_is_translated`, `test_oom_detection` |
| 2026-11-18 | P3 | Real loss falls on real tokens and the curve is archived | val 3.9425 -> 3.5209 over 400 steps; `docs/TRAINING.md` §9 |
| 2026-11-18 | P3 | Resume causes no metric jump and is repeatable | val 3.520929 -> 3.520660 at step 405 (two identical resumes); `docs/TRAINING.md` §9 |
| 2026-11-18 | P3 | Lint/format clean at phase close | `ruff check .` -> All checks passed; `ruff format --check .` -> 58 files formatted |

---

## 10. Session Log

| Date | Work done | Verified by | Next step |
|---|---|---|---|
| 2026-11-16 | Spec analysis, environment audit, plan created | read-only audit | Phase 0 execution |
| 2026-11-16 | Phase 0: git init, skeleton, venv (torch 2.15.0+cpu), check_env.py, 11 tests, ruff config, README + ENVIRONMENT docs, benchmarks recorded | `pytest` 11 passed, `ruff` clean, check_env measurements in §8 | Phase 1: dataset loaders + BPE tokenizer |
| 2026-11-17 | GitHub wiring: rebased Phase 0 onto upstream `main` (LICENSE kept), authored docs commit, pushed | `9af411a` + `f687747` pushed to `origin/main`, remote HEAD confirmed via GitHub API (author `Adithyan-D`) | Phase 1 |
| 2026-11-17 | Phase 1: loaders/preprocess/stats/split pipeline, from-scratch byte-level BPE, persistence, visualization, HF benchmark, 5 CLI scripts, 48 new tests, docs/DATASET_AND_TOKENIZER.md, real Tiny Shakespeare run | 59 tests pass, ruff clean, dataset `tinyshakespeare-971a2b0d5ff5` (524,233 tokens), BPE-1024 (768 merges, 2.85 s) | Phase 2: transformer from scratch |
| 2026-11-17 | Phase 1 pushed to origin/main | 3fba6c2 (feat) + cdb3cfc (docs) verified on remote | Phase 2 |
| 2026-11-18 | Phase 2: from-scratch GPT (embeddings, causal SDPA, MHA, pre-LN blocks, MLP, tied LM head), sampling (greedy/temperature/top-k/top-p/repetition), parameter + memory estimator, mask cache, opt-in fused SDPA, configs + inspect/benchmark CLI, docs/TRANSFORMER.md, 56 new tests | 115 tests pass, ruff clean; 5,034,240-param model at 757-803 ms/step (batch 8 x 128, 8 threads); fused parity asserted | Phase 3: pretraining engine |
| 2026-11-18 | Phase 2 pushed to origin/main | 39ea566 (feat) + 6e536bd (docs) verified equal to `git ls-remote origin refs/heads/main` | Phase 3 |
| 2026-11-18 | Phase 3: token-stream cache, packed non-overlapping windows, deterministic epoch sampler, warmup+cosine schedule, AdamW + clipping + accumulation, fixed val loop, atomic checkpoints with cursor/RNG, RAM pre-flight + OOM translation, train/prepare CLIs, config, 29 new tests, docs/TRAINING.md | 144 tests pass, ruff clean; 400 steps on real Tiny Shakespeare tokens: 409,600 tokens in 326.7 s (1,253.7 tok/s), val 3.9425 -> 3.5209, resume 400 -> 405 with no jump | Phase 4: pretrained models + LoRA |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CPU-only training too slow | low motivation / stalled project | 1–5M params, short contexts, small corpus, honest timing |
| RAM OOM with HF models | crashes, corrupted runs | one model at a time, size guards before load, GGUF fallback |
| Colab/Kaggle session loss | lost work | checkpoint every N steps, resume, export adapters + metrics |
| Disk growth from model cache | storage exhaustion | cache inside project, size checks, cleanup script |
| llama.cpp Windows build issues | blocked inference path | prebuilt binaries; PyTorch CPU fallback path |
| Python 3.14 hijack of default interpreter | broken env | always invoke `.venv\Scripts\python.exe` / `py -3.12` |
| MLflow heaviness | RAM/setup friction | local file+SQLite store, introduced only at Phase 5 |
| Reproducibility drift | invalid comparisons | seeds, saved configs, env records, dataset hashes |
| Scope creep | unfinished platform | phase gates + acceptance criteria enforced here |

---

## 12. Definition of Done (Project)

- [ ] From-scratch transformer + tokenizer + pretraining pipeline, all tested
- [ ] Pretrained model integration with real LoRA fine-tuning artifacts
- [ ] Experiment tracking with comparable, reproducible runs
- [ ] Evaluation metrics + quantization trade-off study with real measurements
- [ ] Working FastAPI backend + React dashboard using actual ML services
- [ ] Documentation synchronized with implementation
- [ ] No fabricated metrics; every number traceable to a command in §8
