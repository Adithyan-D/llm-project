# Pretraining Engine (Phase 3)

The training loop that turns the Phase 1 token streams and the Phase 2 GPT into a
trained checkpoint: packed batches, cross-entropy, AdamW, warmup + cosine decay,
gradient clipping and accumulation, a fixed validation set, atomic checkpoints and
exact resume. Every number below was measured on the machine in
`docs/ENVIRONMENT.md` (i7-1165G7, 8 threads, CPU only).

| Code | Tests |
|---|---|
| `backend/ml/training/data.py` — token streams, packing, sampler, eval batches | `backend/tests/test_training.py` (29 total) |
| `backend/ml/training/config.py` — validated run config, nested `GPTConfig` | same |
| `backend/ml/training/scheduler.py` — warmup + cosine/linear/constant | same |
| `backend/ml/training/checkpoint.py` — atomic save, `latest.json`, load/resume | same |
| `backend/ml/training/safety.py` — RAM pre-flight, OOM translation | same |
| `backend/ml/training/trainer.py` — step loop, eval, logging, report | same |
| `scripts/prepare_training_data.py` — build the token cache | run manually, § 2 |
| `scripts/train_gpt.py` — train / resume / sample-only CLI | run manually, § 9 |
| `configs/training/pretrain-gpt5m-shakespeare.json` — the real run | § 9 |

Suite at the end of Phase 3: **144 tests pass**, `ruff check` and
`ruff format --check` clean.

---

## 1. Pipeline

```
datasets/processed/<dataset_id>/{train,val}.jsonl
   + model_registry/tokenizers/<name>/tokenizer.json
        │  encode once, append <|endoftext|> after every document
        ▼
tokens-b3ae97b72816-eos1024/{train,val}.npy        int32, 1-D
        │  slice into non-overlapping (context_length + 1) windows
        ▼
PackedSampler(epoch, batch_index) → (inputs, targets) shifted by one token
        │
        ▼
Trainer.fit() → loss/val/LR logs, step-XXXXXX.pt, latest.json, report.json
```

## 2. Token streams and packing

`build_token_streams()` is a pure function of the dataset and tokenizer *files*.
The cache directory is keyed by both hashes:

```
datasets/processed/tinyshakespeare-971a2b0d5ff5/tokens-b3ae97b72816-eos1024/
    train.npy    440,076 tokens    20,607 documents
    val.npy       55,491 tokens     2,580 documents
    manifest.json
```

- **Single stream per split, documents separated by `<|endoftext|>` (id 1024).**
  Packing concatenates documents and fills every window completely, so no padding
  token is ever trained on. The cost is that a window may straddle a document
  boundary; the EOS token makes that boundary visible instead of hiding it.
- **No overlaps.** Windows start at `0, T, 2T, …`, so each token is seen once per
  epoch. Tiny Shakespeare's train split has 3,437 windows; with batch 8 that is
  **429 steps/epoch** (3,432 windows used, 5 dropped) and 439,296 tokens/epoch.
- **Cache hit is a file check.** The manifest records dataset hash, tokenizer
  SHA-256, EOS id, per-split token counts and file hashes; a mismatch rebuilds
  into a different directory rather than mixing vocabularies. Building this cache
  took **2.1 s** for 23,187 documents (the encode loop is ~1.4 M chars/s from
  Phase 1).
- **`int32`, not `int64`.** Token ids fit in 16 bits, and the cache is 2× smaller;
  conversion to `torch.long` happens once per batch.

Targets are the same window shifted by one token: `targets[i] = tokens[s+1 : s+T+1]`.
The model's loss is therefore "predict the next token at every position", with no
extra target file on disk.

## 3. Sampling: deterministic shuffling without storing a permutation

`PackedSampler` derives the window order for epoch `e` from
`numpy.random.default_rng(seed + e)`. Consequences:

- The same `(seed, epoch, batch_index)` always produces the same batch — on any
  machine and after any restart.
- A checkpoint only needs `BatchCursor(epoch, batch_index)`; the permutation is
  regenerated, not serialised.
- Epochs differ (the permutation changes), so the model does not see one fixed
  order forever.
- Resuming mid-epoch continues at exactly the next unused batch.

Validation uses `evaluation_batches()`: a fixed set of 10 batches of 8 windows,
chosen evenly across the whole validation split (`np.linspace` over window
indices), not the first 80 windows. Fixing them makes val loss comparable across
steps; spreading them stops the number from reflecting only the opening lines.

## 4. Loss, optimiser and schedule

- **Loss**: PyTorch `cross_entropy` on flattened logits (`B·T, V`) against
  flattened targets. Reported loss is the mean over the micro-batches of an
  optimizer step.
- **Gradient accumulation**: each micro-batch loss is divided by the number of
  micro-batches before `backward()`, so the accumulated gradient equals the
  gradient of the mean loss over the full batch. Asserted by
  `test_gradient_accumulation_matches_full_batch`.
- **AdamW**: `lr=6e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`. This is
  torch's decoupled-weight-decay AdamW; like torch's default, decay applies to
  every parameter including LayerNorm scales and biases. No parameter-group split
  is used yet — a deliberate simplification, noted in § 12.
- **Schedule** (`scheduler.py`), step numbering zero-based:

  ```
  step < warmup:        lr = base_lr · (step + 1) / warmup
  otherwise:            p  = (step − warmup) / (max_steps − warmup)
                        lr = min_lr + ½(base_lr − min_lr)(1 + cos(πp))
  ```

  with `min_lr = base_lr · 0.1`. The warmup starts at `base_lr/warmup` rather
  than 0, so the first update is not a no-op. The schedule is a pure function of
  the step, so a resumed run computes the same LR as an uninterrupted one.
- **Gradient clipping**: global norm clipped to 1.0 after accumulation, before
  `optimizer.step()`. The pre-clip norm is logged so instability is visible.
- **Threads**: `torch.set_num_threads(8)` from the config (Phase 0 policy).

## 5. Validation

`Trainer.evaluate()` switches to eval mode, runs `torch.no_grad()` over the fixed
validation batches and restores the previous mode. No dropout (dropout is 0.0 in
the shipped config), no gradient state, no sampler advance. It runs every
`eval_interval` steps and its result is what the curve in § 9 plots.

## 6. Checkpoints and resume

Each `step-XXXXXX.pt` contains:

| Key | Contents |
|---|---|
| `step`, `tokens_seen`, `best_val_loss` | progress |
| `cursor` | `{epoch, batch_index}` of the next batch |
| `model_config`, `training_config` | full architecture + run config |
| `stream_info` | dataset hash, tokenizer SHA-256, EOS id, split sizes |
| `model` | `state_dict()` |
| `optimizer` | AdamW moments + step counters |
| `rng` | torch, Python `random`, NumPy states |
| `history` | last 200 log records |

- **Atomicity**: `torch.save` goes to `.step-XXXXXX.pt.tmp`, then `os.replace`
  renames it. `latest.json` is written the same way. A crash mid-write leaves the
  previous checkpoint and pointer intact.
- **Resume checks the data**: `dataset_hash`, `tokenizer_sha256` and `eos_token_id`
  must match the current streams, and the model config must match exactly
  (`load_state_dict(..., strict=True)`), so a checkpoint cannot be silently
  attached to a different vocabulary or architecture.
- **Resume restores RNG and cursor**, so the next batches and any dropout masks
  are the ones the uninterrupted run would have used. `test_resume_matches_uninterrupted_run`
  goes further: a 6-step run stopped at step 3 and resumed ends with parameters
  equal to the uninterrupted run (`atol=1e-7`).
- **The LR horizon is preserved.** If you resume a 400-step run with
  `--max-steps 405`, the schedule still ends at 400 — otherwise the same step
  number would suddenly map to a different point on the cosine curve and the
  metric would jump for a scheduling reason rather than a modelling one. The
  trainer prints `keeping checkpoint horizon of 400 steps`
  (`test_resume_keeps_checkpoint_schedule_horizon`).
- **`latest.json` holds the path as the trainer saw it** (relative to the working
  directory in the shipped config). Resolution tries that path, then the run
  directory, then the same filename next to the pointer, so a copied or moved run
  directory still loads. Found the hard way: the first resume attempt produced
  `run/run/step-000400.pt`.

A checkpoint is a pickle. Only load files this project produced.

## 7. Memory and OOM guards

CPU torch raises a generic `RuntimeError` from the allocator, and a failed
allocation can leave the process unusable. The trainer therefore checks before
step 1:

- The Phase 2 analytic estimate for this config: **438.8 MiB** (weights 19.2,
  gradients 19.2, AdamW state 38.4, activations 362.0).
- Available physical RAM via `GlobalMemoryStatusEx` on Windows (POSIX
  `sysconf` elsewhere).
- If the estimate exceeds **70 %** of available RAM, refuse to start with a
  message naming the knobs to reduce; `--force` overrides. The margin covers
  allocator slack, validation tensors and the desktop.
- Allocator failures during training are caught (markers such as
  `DefaultCPUAllocator: can't allocate memory`) and re-raised as
  `TrainingOOMError` with the analytic estimate and remedies.

The estimate is a planning bound, not a measurement: peak RSS for the shipped
run was not instrumented, so this document does not claim one.

## 8. Logging and report

Every `log_interval` steps, one JSON line is appended to
`<out_dir>/train_log.jsonl`:

```json
{"step": 200, "epoch": 0, "tokens_seen": 204800, "lr": 0.00035452,
 "train_loss": 3.636153, "val_loss": 3.61778, "grad_norm": 0.819944,
 "elapsed_s": 161.233, "tokens_per_s": 1270.2}
```

`train_loss` is the mean over the steps since the last record (all of them, not a
single noisy batch); `elapsed_s` and `tokens_per_s` cover the current `fit()`
call. `report.json` is rewritten at the end of every run with the same history
plus parameter counts, config, environment, checkpoint path, `samples` and
`resumed_from`.

`first_train_loss` is the first logged window, which does **not** include step 1.
`first_step_loss` is the literal first optimizer step's loss (for a resumed run,
the first step after resume — the checkpoint is where the original run's value
would be). Quote the relevant one explicitly.

## 9. The real run (Tiny Shakespeare)

Config: `configs/training/pretrain-gpt5m-shakespeare.json` — `tiny-5m`
(5,034,240 params, `V=1025`, `T=128`, `d=256`, 4 heads, 6 layers), batch 8 × 1
accumulation, 400 steps, AdamW `lr=6e-4`, warmup 20, cosine floor 0.1, wd 0.1,
clip 1.0, eval every 50 steps on 10 fixed batches, seed 42, 8 threads.

```
.venv\Scripts\python.exe scripts\prepare_training_data.py ^
    --dataset datasets\processed\tinyshakespeare-971a2b0d5ff5 ^
    --tokenizer model_registry\tokenizers\tinyshakespeare-bpe-1024

.venv\Scripts\python.exe scripts\train_gpt.py ^
    --config configs\training\pretrain-gpt5m-shakespeare.json
```

### Measured curve

400 steps × 1,024 tokens = **409,600 tokens** in **326.7 s** = **1,253.7 tok/s**
(93.2 % of one epoch; the epoch is 429 steps). Reference for an untrained model
with `V=1025` is `ln(1025) = 6.932`; the first logged window (steps 1–20) was
**5.377**.

| Step | Train loss | Val loss | LR | Tokens seen |
|---:|---:|---:|---:|---:|
| 50 | 4.0421 | 3.9425 | 5.92e-4 | 51,200 |
| 100 | 3.7732 | 3.7263 | 5.44e-4 | 102,400 |
| 150 | 3.6664 | 3.6617 | 4.60e-4 | 153,600 |
| 200 | 3.6362 | 3.6178 | 3.55e-4 | 204,800 |
| 250 | 3.5988 | 3.5827 | 2.44e-4 | 256,000 |
| 300 | 3.5713 | 3.5540 | 1.49e-4 | 307,200 |
| 350 | 3.5700 | 3.5380 | 8.36e-5 | 358,400 |
| 400 | 3.5817 | **3.5209** | 6.00e-5 | 409,600 |

Val loss is monotone across every evaluation point; train loss is noisy but the
trend matches. The curve is still descending when the run stops — 400 steps is
0.93 epoch, not convergence.

### Samples (verbatim, this checkpoint)

Prompted with a prepended `<|endoftext|>`, temperature 0.8, top-k 40, top-p 0.95,
80 new tokens requested, EOS allowed to stop generation:

| Prompt | New tokens | Speed | Output |
|---|---:|---:|---|
| `ROMEO:` | 22 | 30.3 tok/s | `ROMEO: me as fiil be the I of with I,` |
| `To be, or not to be` | 34 | 31.7 tok/s | `To be, or not to be wes the your a it his so the your and to to in and wa,` |

Read plainly: the model has learned casing, speaker-label shape, punctuation and
a stock of common words, but this is **not coherent English**. At 5M parameters
and 0.93 epoch that is the honest result; the acceptance claim is that a saved
checkpoint generates text of the kind its scale allows, not that the text is
good. Generation has no KV cache (Phase 2 decision), hence ~30 tok/s on CPU.

### Resume proof

Continuing the finished run by 5 steps, from the step-400 checkpoint:

```
.venv\Scripts\python.exe scripts\train_gpt.py ^
    --config configs\training\pretrain-gpt5m-shakespeare.json ^
    --resume checkpoints\shakespeare-gpt5m\step-000400.pt ^
    --max-steps 405 --eval-interval 5 --log-interval 5 --checkpoint-interval 500
```

```
checkpoint    : ...\step-000400.pt
schedule      : keeping checkpoint horizon of 400 steps
resumed       : step 400, 409,600 tokens seen
step  405/405 | train 3.5249 | val 3.5207 | lr 6.00e-05 | tok 414,720 | 800 tok/s
done          : step 405, 5,120 tokens in 6.7s (769 tok/s)
```

| | Step 400 | Step 405 | Δ |
|---|---:|---:|---:|
| Val loss | 3.520929 | 3.520660 | **−0.00027** |
| LR | 6.0009e-5 | 6.0e-5 | unchanged (floor) |
| Checkpoint | 60,517,397 B | 60,518,549 B | +1,152 B (history only) |

No jump: the first post-resume validation is a continuation of the same curve.
Running the same 5-step resume a second time reproduced val `3.520660` exactly,
which is the RNG + cursor restoration doing its job. All resume numbers above
come from that second, correctly-pathed run.

### Artifacts (gitignored, local)

```
checkpoints/shakespeare-gpt5m/
    step-000200.pt   step-000400.pt   step-000405.pt
    latest.json      report.json      train_log.jsonl
experiments/phase3_train_stdout.txt
experiments/phase3_resume_stdout.txt
```

Checkpoints are ~57.7 MiB: 5.0M FP32 params × 4 B, AdamW m/v × 8 B, plus
optimizer step counters, RNG states and history — hence 12 B/param.

## 10. What the tests prove

| Claim | Test(s) |
|---|---|
| Streams equal a fresh tokenizer encode with EOS after every doc | `test_token_stream_appends_eos_and_matches_tokenizer` |
| Cache is reused byte-for-byte; new tokenizer → new cache | `test_token_stream_cache_is_reused_without_reencoding`, `test_changing_the_tokenizer_builds_a_separate_cache` |
| Missing separator is an error, not silent text | `test_missing_separator_is_rejected` |
| Windows are non-overlapping; targets are inputs + 1 | `test_sampler_shapes_and_shifted_targets` |
| Same seed/epoch → same batch; epochs differ | `test_sampler_epoch_is_deterministic_and_shuffled` |
| Every window appears once per epoch; cursor wraps | `test_sampler_covers_each_window_once_per_epoch`, `test_cursor_wraps_and_advances` |
| Too-short streams and out-of-range indices are rejected | `test_sampler_rejects_streams_that_cannot_fill_a_batch` |
| Val batches are fixed, spread across the split | `test_validation_batches_are_fixed_and_spread` |
| Warmup peaks at `warmup_steps`, decays to the floor | `test_schedule_warmup_peak_and_decay`, `test_schedule_kinds_and_validation` |
| Checkpoint restores model, optimizer, RNG, cursor, metadata | `test_checkpoint_roundtrip_restores_model_optimizer_and_rng` |
| `latest.json` resolves from CWD and from a moved run dir | `test_latest_pointer_resolves_paths_relative_to_the_working_directory`, `test_latest_pointer_falls_back_to_the_run_directory` |
| Wrong architecture / unknown format are rejected | `test_checkpoint_rejects_mismatched_model_config`, `test_checkpoint_rejects_unknown_format` |
| Python + NumPy RNG states survive the round trip | `test_random_module_state_is_restored_by_checkpoint` |
| Memory guard refuses an over-budget run | `test_memory_budget_guard`, `test_available_memory_is_a_positive_number_on_this_machine` |
| OOM errors are recognised and translated | `test_oom_detection`, `test_oom_during_step_is_translated` |
| Config rejects bad values and round-trips through JSON | `test_config_validation_and_json_roundtrip` |
| Loss falls on learnable data with real logs and a checkpoint | `test_trainer_reduces_loss_on_learnable_data` |
| Resume equals an uninterrupted run (parameters, tokens, loss) | `test_resume_matches_uninterrupted_run` |
| Extending a run keeps the checkpoint's LR horizon | `test_resume_keeps_checkpoint_schedule_horizon` |
| Accumulation gradient equals one full-batch gradient | `test_gradient_accumulation_matches_full_batch` |
| Two identically seeded runs are bit-identical | `test_seed_everything_makes_runs_identical` |
| Memory estimate uses the real parameter count | `test_memory_estimate_uses_real_parameter_count` |

## 11. Complexity and cost

- One step: `O(B·T·N·d²)` for the projection/MLP work plus `O(B·T²·d)` for
  attention scores and head outputs, times ~3 for forward + backward. At
  `B=8, T=128, d=256, N=6` the measured step is ~0.8 s on this CPU, matching
  Phase 2.
- Packing is `O(total tokens)` per epoch plus `O(windows)` memory for the
  permutation (`int64` indices, 3,437 × 8 B ≈ 27 KiB) — negligible.
- Evaluation is 10 forward-only passes every 50 steps. Wall time already includes
  it; it was not timed in isolation, so no overhead percentage is claimed.
- Checkpointing writes ~58 MiB every 200 steps. Logged throughput at steps 180,
  200 and 220 was 1,277.8 / 1,270.2 / 1,272.1 tok/s, i.e. no visible dip beyond
  the run's normal variation.

## 12. Limitations (known and deliberate)

- **0.93 epoch.** The run stops mid-first-epoch; val loss 3.5209 is not a
  converged estimate and should not be compared to published Shakespeare numbers.
- **One corpus, one seed.** No seed-variance study yet, so a 0.01 loss difference
  between configs would not be interpretable.
- **No LR/weight-decay parameter groups.** Weight decay touches LayerNorm and
  biases, contrary to common practice; measurable but not yet measured here.
- **No mixed precision, no activation checkpointing code** (the memory model for
  it exists, the implementation does not).
- **No KV cache**, so generation is ~30 tok/s and grows with context.
- **No gradient-norm history in `report.json`** beyond the logged records; the
  per-record values are in `train_log.jsonl`.
- **Peak RSS is not instrumented**; the 438.8 MiB figure is analytic.
- **Checkpoints are pickles** and carry no signature; only load your own.
- `latest.json` stores working-directory-relative paths; resolution has fallbacks
  but an absolute pointer would be simpler if runs are moved often.

## 13. Reproduce

```powershell
# full suite
.venv\Scripts\python.exe -m pytest

# build / verify the token cache (2.1 s cold, file check when warm)
.venv\Scripts\python.exe scripts\prepare_training_data.py ^
    --dataset datasets\processed\tinyshakespeare-971a2b0d5ff5 ^
    --tokenizer model_registry\tokenizers\tinyshakespeare-bpe-1024

# the 400-step run (~5.5 min on this machine)
.venv\Scripts\python.exe scripts\train_gpt.py ^
    --config configs\training\pretrain-gpt5m-shakespeare.json

# resume 5 steps and generate from the result
.venv\Scripts\python.exe scripts\train_gpt.py ^
    --config configs\training\pretrain-gpt5m-shakespeare.json ^
    --resume checkpoints\shakespeare-gpt5m\step-000400.pt ^
    --max-steps 405 --eval-interval 5 --log-interval 5 --checkpoint-interval 500

# generation only, extra prompts
.venv\Scripts\python.exe scripts\train_gpt.py ^
    --config configs\training\pretrain-gpt5m-shakespeare.json ^
    --resume checkpoints\shakespeare-gpt5m\latest.json --sample-only ^
    --prompts "ROMEO:" "JULIET:" "To be, or not to be" --sample-tokens 60
```
