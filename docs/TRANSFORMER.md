# Transformer From Scratch (Phase 2)

A decoder-only GPT built entirely from primitives in this repository — no
`transformers`, no `torch.nn.MultiheadAttention`, no pretrained weights. Every
number in this document was measured on the machine described in
`docs/ENVIRONMENT.md` (i7-1165G7, 8 threads, CPU only).

| Code | Tests |
|---|---|
| `backend/ml/models/attention.py` — causal SDPA, MHA, mask cache | `backend/tests/test_attention.py` (13) |
| `backend/ml/models/layers.py` — LayerNorm, MLP, pre-LN block | `backend/tests/test_model.py` (16) |
| `backend/ml/models/gpt.py` — GPT, forward/loss, `estimate_loss` | `backend/tests/test_model.py` |
| `backend/ml/models/generation.py` — sampling policies | `backend/tests/test_generation.py` (16) |
| `backend/ml/models/estimation.py` — param counts, memory estimate | `backend/tests/test_estimation.py` (11) |
| `backend/ml/models/config.py` — validated config + presets | `backend/tests/test_model.py` |
| `scripts/inspect_model.py` — inspect + benchmark a config | run manually, see §7 |

Suite at the end of Phase 2: **115 tests pass**, `ruff check` and
`ruff format --check` clean.

---

## 1. Architecture

```
token ids (B, T)
  → token embedding (V, d) + learned position embedding (T_ctx, d)
  → dropout
  → N × TransformerBlock:
        x = x + MHA(LayerNorm(x))          # pre-LN attention
        x = x + MLP(LayerNorm(x))          # pre-LN feed-forward
  → LayerNorm
  → LM head (d → V), weights tied to the token embedding by default
  → logits (B, T, V)
```

Shape flow inside one attention layer:

| Tensor | Shape |
|---|---|
| input | `(B, T, d)` |
| QKV projection | `(B, T, 3d)` |
| Q, K, V per head | `(B, H, T, d/H)` |
| scores `QKᵀ/√d_h` | `(B, H, T, T)` |
| weights (softmax) | `(B, H, T, T)` |
| head output | `(B, H, T, d/H)` → merged `(B, T, d)` |

Design choices:

- **Pre-LN** (normalise → sublayer → add residual). Post-LN needs warmup and is
  noticeably less stable at tiny batch sizes; pre-LN trains from step 1 with a
  constant LR, which matters because Phase 3 has no GPU to waste on restarts.
- **Learned positional embeddings**, as in GPT-2. At `T ≤ 128` the extra
  `T_ctx·d` parameters (32,768 at `d=256`) are cheap and simpler than RoPE, which
  only pays off beyond the training window — a Phase 3+ decision if context grows.
- **Fused QKV**: one `Linear(d, 3d)` instead of three `Linear(d, d)`; fewer
  kernel launches and better cache behaviour on CPU, at identical parameter count.
- **Weight tying**: the LM head shares storage with the token embedding
  (`lm_head.weight is token_embedding.weight`). This removes `V·d` parameters
  (262,400 at the 5M preset) and ties input/output token semantics together.
- **Bias on by default**; every bias can be disabled with `bias=False`, and the
  analytic parameter formula is asserted against the built model in both modes.

Init: `N(0, 0.02)` for all weights, zero biases, and GPT-2-style residual scaling
(`std = 0.02/√(2N)`) on `attn.proj` and `mlp.fc_out`, so residual variance does not
grow with depth.

## 2. Scaled dot-product attention

```
scores  = Q Kᵀ / √d_h                    # (B, H, T, T)
scores  = masked_fill(~causal, -inf)     # row i may see columns 0..i
weights = softmax(scores, dim=-1)
out     = weights V
```

- **Why scale by `1/√d_h`**: dot products of `d_h`-dimensional random vectors have
  variance `d_h`; without scaling, logits grow with `d_h` and softmax saturates,
  killing gradients. Dividing by the standard deviation keeps unit-scale logits.
- **Causal mask**: a boolean lower-triangular matrix. `-inf` (not a large negative
  constant) guarantees exactly zero softmax weight for future positions. The mask
  is memoised per `(T, device)` — it is constant and gradient-free, so rebuilding
  it six times per training step (once per layer) is pure overhead. Measured:
  **81.3 µs** per fresh 128×128 mask vs **6.9 µs** cached (20,000-call loop), i.e.
  ≈0.49 ms vs ≈0.04 ms per step — real but small next to the ~0.8 s step.
- **Complexity**: `O(T²·d)` time and `O(T²)` score memory per head. Doubling
  context quadruples attention cost, which is why this project caps context at
  128 for CPU training.
- **Fused path**: `MultiHeadAttention(fused_attention=True)` dispatches to
  `F.scaled_dot_product_attention` (same maths, specialised kernel). It is
  **off by default**; the handwritten implementation is the reference and the
  default. Both paths are asserted equal in value and gradient
  (`test_fused_attention_matches_reference_path`,
  `test_fused_attention_gradients_match_reference`).

## 3. LayerNorm, MLP and the block

`LayerNorm` is written directly: mean and biased variance over the last
dimension, `(x - μ)/√(σ² + ε)`, then a learned per-channel scale and optional
shift. The epsilon is inside the square root, matching PyTorch semantics, and the
implementation is checked implicitly by the model-level gradient and
determinism tests.

`FeedForward` is the standard 4× MLP (`d → 4d → GELU → d`) with dropout on the
output. At `d=256` it holds 525,568 parameters per layer — the largest block
component, and the reason `d` dominates parameter count.

Each block therefore has `12d²` weight parameters (`qkv 3d² + proj d² + MLP 8d²`)
plus biases and two LayerNorms. The analytic counter in `estimation.py` encodes
this and is asserted to match the instantiated model for tied/untied and
bias/no-bias variants.

## 4. Generation

`generate()` supports greedy decoding, temperature scaling, top-k, nucleus
(top-p), and a CTRL-style repetition penalty over unique tokens. The model runs
in eval mode during generation and its previous mode is restored afterwards.
Seeded sampling uses a dedicated `torch.Generator`, so runs are reproducible
without touching global RNG state.

**No KV cache yet.** Every new token re-runs the full window, so per-token cost
grows with context length and generation is `O(T²)` per step. That is acceptable
at 128 tokens and was deliberately deferred to the inference phase; the estimator
explicitly excludes KV cache memory.

## 5. Parameter counts (measured)

`scripts/inspect_model.py --config configs/models/gpt-5m.json`:

| Preset | Config | Params | Tied |
|---|---|---|---|
| `nano-2m` | `d=192, H=4, N=4, T=128, V=1025` | **2,001,216** | 196,800 |
| `tiny-5m` | `d=256, H=4, N=6, T=128, V=1025` | **5,034,240** | 262,400 |

Breakdown for `tiny-5m` (grouped by module):

| Group | Params |
|---|---|
| token embedding | 262,400 |
| position embedding | 32,768 |
| `blocks.*.attn` (qkv + proj + biases) | 1,579,008 |
| `blocks.*.mlp` (fc_in + fc_out + biases) | 3,153,408 |
| `blocks.*.ln_attn` | 3,072 |
| `blocks.*.ln_mlp` | 3,072 |
| final LayerNorm | 512 |
| LM head | 0 (tied) |

Initial loss of an untrained model should be close to `ln(V) = 6.932` for uniform
predictions; the benchmark prints this as a sanity reference. Observed loss after
15 AdamW steps on a fixed random batch was 3.954 — the model is memorising that
batch, not learning language.

## 6. Memory estimate

`estimate_memory()` is **analytic**, not measured. For `tiny-5m`, batch 8 × 128:

| Component | Size |
|---|---|
| weights (FP32) | 19.2 MiB |
| gradients (FP32) | 19.2 MiB |
| AdamW state (m, v — 8 B/param) | 38.4 MiB |
| activations | 362.0 MiB |
| **total** | **438.8 MiB** |
| with activation checkpointing | **143.8 MiB** |

Assumptions:

- Activations are budgeted as `B·T·56·d` elements per layer plus
  `2·B·H·T²` for scores and softmax probabilities, plus `2·B·T·d` for
  embeddings. The `56d` figure is deliberately conservative: roughly 18 distinct
  `d`-sized block tensors are live for backward, and the factor of ~3 leaves room
  for backward temporaries and allocator slack. Real peak RSS is always measured
  separately; the estimate is a planning bound, not a promise.
- Checkpointing stores block boundaries only and recomputes internals, so the
  per-layer activation term collapses to `2·B·T·d` + `(N−1)·B·T·d`. It is
  modelled, not implemented yet — there is no checkpointing code in Phase 2.
- Excludes KV cache, allocator fragmentation, and the tokenizer/dataset footprint.
- `optimizer="sgd"` reports zero optimizer state; unknown optimizers are rejected.

## 7. Measured throughput

Command (`8` threads, batch 8 × 128, AdamW lr 3e-4, 5 steps per repetition, best
of 3 repetitions; a fresh identically-seeded model per path):

```
.venv\Scripts\python.exe scripts\inspect_model.py --config configs\models\gpt-5m.json --iters 5 --repeats 3
```

| Model | Reference attention | Fused SDPA | Ratio |
|---|---|---|---|
| `nano-2m` | 322.7 ms/step, 3,173 tok/s | 330.8 ms/step, 3,096 tok/s | 0.98× |
| `tiny-5m` run A | 802.8 ms/step, 1,276 tok/s | 747.7 ms/step, 1,370 tok/s | 1.07× |
| `tiny-5m` run B | 756.9 ms/step, 1,353 tok/s | 694.6 ms/step, 1,474 tok/s | 1.09× |

Interpretation, stated plainly:

- On this CPU the fused kernel is **not a clear win at 2M params** and is only
  ~7–9% faster at 5M. Run-to-run spread on a shared laptop is larger than either
  effect. That is why the reference path remains the default and the fused path
  is an opt-in flag rather than an assumed optimisation.
- An earlier Phase 2 reading of 1204.5 ms/step was taken in a noisy session with
  a different timing loop (single run, no warm-up repetition, no best-of). It is
  superseded by the table above; the two are not directly comparable.
- The Phase 0 provisional benchmark (447 ms/step for a 5.29M model) used a
  minimal forward/backward without the full block stack and generation plumbing,
  so it is a floor, not a target. Phase 3 will use the real loop and report its
  own numbers.

## 8. What the tests prove

| Claim | Test(s) |
|---|---|
| Mask is lower-triangular, rejects `T=0` | `test_causal_mask_is_lower_triangular` |
| Cache returns the same tensor, equal to a fresh mask | `test_cached_mask_is_reused_and_matches_fresh_mask` |
| Attention matches an explicit per-position loop | `test_attention_matches_naive_reference` |
| Changing future values cannot change past outputs | `test_causal_attention_ignores_future_tokens`, `test_logits_cannot_see_future_tokens` |
| Non-causal and explicit masks behave as specified | `test_non_causal_attention_sees_everything`, `test_explicit_attention_mask_can_hide_keys` |
| Head split/merge is lossless and slices channels correctly | `test_split_merge_heads_roundtrip`, `test_head_split_preserves_per_head_slices` |
| Gradients reach every trainable parameter and are finite | `test_gradients_flow_to_all_trainable_parameters`, `test_mha_shapes_and_gradients` |
| Fused and reference paths agree in value and gradient | `test_fused_attention_matches_reference_path`, `test_fused_attention_gradients_match_reference` |
| Same seed → identical parameters; eval mode is deterministic | `test_initialisation_is_seed_deterministic`, `test_eval_mode_is_deterministic_with_dropout` |
| Dropout is stochastic only in training mode | `test_training_mode_dropout_is_stochastic` |
| Tied head shares storage; untied adds exactly `V·d` | `test_weight_tying_shares_storage`, `test_untied_head_adds_vocab_projection` |
| Analytic parameter count matches the model (bias on/off) | `test_weight_tying_shares_storage`, `test_no_bias_variant_counts_match` |
| Greedy/temperature-0/top-k=1 agree; sampling is seed-reproducible | `test_temperature_zero_matches_greedy`, `test_top_k_one_is_greedy`, `test_sampling_is_seed_reproducible` |
| Top-k/top-p keep the promised support and mass | `test_top_k_limits_support_to_k_tokens`, `test_top_p_keeps_mass_and_at_least_one_token` |
| Repetition penalty matches the hand-computed values | `test_repetition_penalty_matches_manual_math` |
| EOS stops generation; prompt cropped to context; batch rows independent | `test_eos_stops_generation`, `test_prompt_is_cropped_to_context_window`, `test_generate_batch_rows_are_independent` |
| Memory estimate components sum, scale correctly, reject bad input | `test_memory_estimate_includes_all_components`, `test_memory_scales_with_batch_and_sequence`, `test_invalid_shapes_rejected` |

## 9. Limitations (known and deliberate)

- **Untrained generation is meaningless.** `inspect_model.py` prints token IDs
  from a randomly initialised model only to prove the sampling path executes.
  No language-quality claim is made until Phase 3 trains on real tokens.
- No KV cache, no paged attention, no speculative decoding.
- No flash-attention/OpenVINO path; the fused SDPA kernel is the only alternative
  implementation on this hardware.
- No activation checkpointing code yet — only the memory model for it.
- Context length is fixed at construction; extending it requires a new positional
  table and retraining.
- `float32` only; mixed precision has no payoff on CPU without bf16 support.

## 10. Reproduce

```powershell
# full test suite
.venv\Scripts\python.exe -m pytest

# parameter counts, memory estimate, ref-vs-fused throughput, sampling smoke test
.venv\Scripts\python.exe scripts\inspect_model.py --config configs\models\gpt-5m.json --iters 5 --repeats 3

# smaller preset
.venv\Scripts\python.exe scripts\inspect_model.py --config configs\models\gpt-2m.json --iters 5 --repeats 3
```
