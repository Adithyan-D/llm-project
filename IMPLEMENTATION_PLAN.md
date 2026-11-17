# LLM Foundry — Implementation Plan & Tracking Log

> Single source of truth for engineering progress. `README.md` will be user-facing;
> this file is engineering-facing. Update it in the same session as the work is done.

Last updated: 2026-11-16 — **Phase 0 complete and verified**. Current focus: Phase 1
(dataset management + BPE tokenizer).

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
| tokenizer + processed data | <500 MB | ~1 GB |
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

### Phase 1 — Dataset & Tokenizer `[ ]`
- [ ] Loaders for TXT / JSON / JSONL with schema validation
- [ ] Malformed-record quarantine, dedup, text normalization
- [ ] Chunked/streaming preprocessing (no full-file RAM load for large inputs)
- [ ] Statistics: samples, words, approx tokens, avg length, size, split distribution
- [ ] Train/val/test splitting with saved metadata + dataset version hash
- [ ] Custom BPE: pair counting, merges, vocab build, encode, decode, unknown handling
- [ ] Tokenizer save/load (JSON vocab + merges + config)
- [ ] Tokenization visualization (text → tokens → IDs)
- [ ] Benchmark + comparison against an HF tokenizer
- [ ] Tests: round-trip encode/decode, unicode/edge text, vocab growth, reload equality

**Acceptance:** trained tokenizer on a small corpus; round-trip identity test passes;
stats endpoint data reproducible from saved config.

### Phase 2 — Transformer From Scratch `[ ]`
- [ ] Token embeddings + positional embeddings
- [ ] Scaled dot-product attention with causal mask
- [ ] Multi-head attention (Q/K/V projections, head split/merge)
- [ ] MLP/feed-forward block, LayerNorm, residual connections
- [ ] Transformer block stack + LM head (weight tying optional, documented)
- [ ] Autoregressive generation (temperature, top-k, top-p, repetition penalty)
- [ ] Parameter counter + activation/memory estimator
- [ ] Configurable vocab size, d_model, heads, layers, context length, dropout
- [ ] Tests: shapes, causal leakage, gradient flow, determinism under seed

**Acceptance:** 1–5M param model builds on CPU; fwd/bwd runs within budget; all unit
tests pass; generation produces tokens (quality evaluated only after Phase 3).

### Phase 3 — Pretraining Engine `[ ]`
- [ ] Tokenized dataset + input/target pair construction, batching, packing
- [ ] Training loop: CE loss, backprop, AdamW, LR schedule, grad clip, grad accumulation
- [ ] Validation loop
- [ ] Checkpoint save/resume (model, optimizer, scheduler, step, RNG state)
- [ ] OOM safeguards (batch/seq auto-guard, clear error messages)
- [ ] Training logs: loss, val loss, LR, epoch, duration, tokens processed
- [ ] Actually train a small GPT and record real curves

**Acceptance:** loss decreases on real data; checkpoint resumes without metric jump;
saved model generates coherent-looking text for its tiny scale; curves archived.

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

---

## 10. Session Log

| Date | Work done | Verified by | Next step |
|---|---|---|---|
| 2026-11-16 | Spec analysis, environment audit, plan created | read-only audit | Phase 0 execution |
| 2026-11-16 | Phase 0: git init, skeleton, venv (torch 2.15.0+cpu), check_env.py, 11 tests, ruff config, README + ENVIRONMENT docs, benchmarks recorded | `pytest` 11 passed, `ruff` clean, check_env measurements in §8 | Phase 1: dataset loaders + BPE tokenizer |

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
