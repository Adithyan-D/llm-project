"""GPT-style decoder-only transformer built from the primitives in this package."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import MultiHeadAttention  # noqa: F401 - re-exported for inspection
from .config import GPTConfig
from .generation import GenerationConfig, generate
from .layers import LayerNorm, TransformerBlock


class GPT(nn.Module):
    """Decoder-only transformer language model.

    Architecture (pre-LN):

        token emb + learned positional emb
        -> N x [LN -> causal MHA -> +residual -> LN -> MLP -> +residual]
        -> final LN -> linear LM head (optionally tied to the token embedding)

    Parameter count is O(V*d + T_ctx*d + N*d^2): vocabulary and context terms are
    small at this scale, so depth and width dominate. Attention compute is
    O(B*N*T^2*d) and is the reason context length is capped at 128 here.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                bias=config.bias,
                ln_eps=config.ln_eps,
                fused_attention=config.fused_attention,
            )
            for _ in range(config.n_layers)
        )
        self.ln_final = LayerNorm(config.d_model, eps=config.ln_eps, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # GPT-2 residual scaling: keeps the residual stream variance stable with depth.
        residual_std = config.init_std / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.fc_out.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    @property
    def device(self) -> torch.device:
        return self.token_embedding.weight.device

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(logits, loss)``; loss is ``None`` when targets are omitted.

        ``logits`` has shape (batch, seq, vocab_size), ``loss`` is mean
        cross-entropy over the shifted pairs supplied by the caller.
        """
        if idx.dim() != 2:
            raise ValueError(f"expected (batch, seq) token ids, got {tuple(idx.shape)}")
        batch, seq = idx.shape
        if seq > self.config.context_length:
            raise ValueError(
                f"sequence length {seq} exceeds context_length {self.config.context_length}"
            )
        if targets is not None and targets.shape != idx.shape:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} != input shape {tuple(idx.shape)}"
            )

        positions = torch.arange(seq, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    def generate(self, idx: torch.Tensor, config: GenerationConfig | None = None) -> torch.Tensor:
        """Autoregressive continuation; see ``ml.models.generation``."""
        return generate(self, idx, config)

    @torch.no_grad()
    def estimate_loss(
        self,
        batches: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        ignore_index: int = -100,
    ) -> dict[str, float]:
        """Mean loss over pre-built batches, in eval mode (used by Phases 3/5)."""
        if not batches:
            raise ValueError("batches must not be empty")
        was_training = self.training
        self.eval()
        losses: list[float] = []
        try:
            for inputs, targets in batches:
                _, loss = self(inputs, targets, ignore_index=ignore_index)
                if loss is not None:
                    losses.append(float(loss.detach()))
        finally:
            self.train(was_training)
        return {"loss": sum(losses) / len(losses), "batches": float(len(losses))}


__all__ = ["GPT", "GPTConfig"]
