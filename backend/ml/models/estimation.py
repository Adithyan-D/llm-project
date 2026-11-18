"""Parameter counting and memory estimation for a GPT configuration.

The estimates are analytic, not measured: they state the *lower bound* a run
needs for weights, gradients, AdamW state and a conservative activation budget.
Real peak RSS is always measured separately and recorded next to the estimate
so the two can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .config import GPTConfig


@dataclass
class ParameterCounts:
    total: int
    trainable: int
    frozen: int
    tied_parameters: int
    unique_tensors: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "unique_tensors": self.unique_tensors,
            "tied_parameters": self.tied_parameters,
        }

    def summary(self) -> str:
        return (
            f"{self.total:,} params ({self.trainable:,} trainable, "
            f"{self.frozen:,} frozen; {self.tied_parameters:,} tied)"
        )


def count_parameters(model: nn.Module) -> ParameterCounts:
    """Count parameters, counting tied tensors once.

    ``model.parameters()`` already de-duplicates shared tensors (the LM head and
    token embedding share storage when ``tie_weights=True``), so iterating it
    once gives the unique count. The tied amount is derived explicitly.
    """
    unique = list(model.parameters())
    total = sum(p.numel() for p in unique)
    trainable = sum(p.numel() for p in unique if p.requires_grad)
    shared = 0
    seen: dict[int, int] = {}
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            pointer = parameter.data_ptr()
            if pointer in seen:
                shared += parameter.numel()
            else:
                seen[pointer] = parameter.numel()
    return ParameterCounts(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        tied_parameters=shared,
        unique_tensors=len(unique),
    )


def parameter_breakdown(model: nn.Module) -> list[dict[str, object]]:
    """Per-parameter table for architecture inspection in the dashboard/CLI.

    Walks modules directly (not ``named_parameters``) so tied tensors appear
    explicitly and are marked as duplicates rather than silently dropped.
    """
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            pointer = parameter.data_ptr()
            tied = pointer in seen
            seen.add(pointer)
            rows.append(
                {
                    "name": full_name,
                    "shape": list(parameter.shape),
                    "numel": 0 if tied else parameter.numel(),
                    "tied_to_earlier": tied,
                    "trainable": parameter.requires_grad,
                }
            )
    return rows


def estimate_memory(
    config: GPTConfig,
    *,
    batch_size: int = 8,
    seq_len: int | None = None,
    dtype_bytes: int = 4,
    optimizer: str = "adamw",
    activation_checkpointing: bool = False,
    total_parameters: int | None = None,
) -> dict[str, object]:
    """Analytic memory estimate in bytes for one training step.

    Breakdown:
    * weights: ``params * dtype_bytes``
    * gradients: one tensor per trainable parameter
    * AdamW state: two FP32 tensors per parameter (m, v)
    * activations: per layer ``B*T*(56*d)`` plus attention ``2*B*H*T^2``
      elements (scores + probabilities), plus embeddings. This is deliberately
      conservative; in-place ops and caching make real usage lower.

    KV-cache memory for inference is *not* included (not implemented yet).
    """
    seq_len = seq_len if seq_len is not None else config.context_length
    if seq_len > config.context_length:
        raise ValueError(f"seq_len={seq_len} exceeds context_length={config.context_length}")
    if batch_size < 1 or seq_len < 1:
        raise ValueError("batch_size and seq_len must be >= 1")

    params = _analytic_parameter_count(config) if total_parameters is None else total_parameters

    weights = params * dtype_bytes
    grads = params * dtype_bytes
    if optimizer == "adamw":
        optimizer_state = params * 8  # FP32 exp_avg + exp_avg_sq
    elif optimizer == "sgd":
        optimizer_state = 0
    else:
        raise ValueError(f"unsupported optimizer {optimizer!r}")

    d = config.d_model
    heads = config.n_heads
    tokens = batch_size * seq_len
    per_layer = tokens * 56 * d * dtype_bytes + 2 * batch_size * heads * seq_len**2 * dtype_bytes
    embedding_activations = 2 * tokens * d * dtype_bytes
    if activation_checkpointing:
        # Only block boundaries are stored; recomputed internals are transient.
        activations = (
            embedding_activations + per_layer + (config.n_layers - 1) * tokens * d * dtype_bytes
        )
    else:
        activations = embedding_activations + config.n_layers * per_layer

    total = weights + grads + optimizer_state + activations
    return {
        "parameters": params,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dtype_bytes": dtype_bytes,
        "optimizer": optimizer,
        "activation_checkpointing": activation_checkpointing,
        "weights_bytes": weights,
        "gradients_bytes": grads,
        "optimizer_bytes": optimizer_state,
        "activations_bytes": activations,
        "total_bytes": total,
        "weights_human": human_bytes(weights),
        "gradients_human": human_bytes(grads),
        "optimizer_human": human_bytes(optimizer_state),
        "activations_human": human_bytes(activations),
        "total_human": human_bytes(total),
        "note": "analytic estimate, excludes KV cache and allocator slack",
    }


def _analytic_parameter_count(config: GPTConfig) -> int:
    """Closed-form parameter count, matching ``GPT`` exactly (asserted in tests).

    Per block: 12*d^2 weights (qkv 3d^2, proj d^2, MLP 8d^2) + 9*d biases
    (qkv 3d, proj d, fc_in 4d, fc_out d) + two LayerNorms of (d weight + d bias).
    """
    d = config.d_model
    has_bias = 1 if config.bias else 0
    per_layer = 12 * d * d + (9 * d) * has_bias + 2 * (d + d * has_bias)
    embedding = config.vocab_size * d + config.context_length * d
    final_norm = d + d * has_bias
    head = 0 if config.tie_weights else config.vocab_size * d
    return embedding + config.n_layers * per_layer + final_norm + head


def human_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


__all__ = [
    "ParameterCounts",
    "count_parameters",
    "estimate_memory",
    "human_bytes",
    "parameter_breakdown",
]
