"""Q/V LoRA injection for SAM1 ViT-B fused QKV attention layers.

SAM1 uses a fused `self.attn.qkv = nn.Linear(dim, dim * 3)` that projects
input to Q, K, V in a single matmul.  This module injects LoRA adapters on
the Q-channel and V-channel while leaving K untouched.

Design (per block):
  x ──► [frozen qkv] ──► split ──► [Q | K | V]
         (dim→3*dim)                │       │
                                    │       └── frozen
                                    │
                      ┌─────────────┘
                      │
              ┌───────┴───────┐
              │  Q-LoRA       │  V-LoRA
              │  A(dim,r)     │  A(dim,r)
              │  B(r,dim)     │  B(r,dim)
              └───┬───────────┴───┬───┘
                  │               │
                  └─── add ───────┘
                       │
              [Q+ΔQ | K | V+ΔV] ──► downstream
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Dict, List, Tuple


class LoRALinear(nn.Module):
    """Single LoRA branch: W + B @ A, with A kaiming-uniform, B zero init."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = alpha / max(rank, 1)
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

    def forward(self, x: Tensor) -> Tensor:
        # x: [*, in_features]
        return self.scaling * (x @ self.A.t() @ self.B.t())


class FusedQKVLoRA(nn.Module):
    """Wraps SAM1's fused `self.attn.qkv` and adds Q and V LoRA branches.

    Original qkv.weight is kept frozen.  Forward:
      1. y_qkv = frozen_qkv(x)          # [*, 3*dim]
      2. Split into q, k, v             # each [*, dim]
      3. q = q + q_lora(self.dropout(x))
      4. v = v + v_lora(self.dropout(x))
      5. Return cat(q, k, v, dim=-1)
    """

    def __init__(
        self,
        qkv_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        dim = qkv_linear.in_features
        if qkv_linear.out_features != dim * 3:
            raise ValueError(
                f"Expected qkv.out_features == {dim}*3, got {qkv_linear.out_features}"
            )
        self.dim = dim
        self.qkv = qkv_linear  # frozen by caller

        self.q_lora = LoRALinear(dim, dim, rank=rank, alpha=alpha, dropout=dropout)
        self.v_lora = LoRALinear(dim, dim, rank=rank, alpha=alpha, dropout=dropout)

    @property
    def rank(self) -> int:
        return self.q_lora.rank

    @property
    def scaling(self) -> float:
        return self.q_lora.scaling

    def forward(self, x: Tensor) -> Tensor:
        # Original fused QKV projection (no grad)
        with torch.no_grad():
            y = self.qkv(x)  # [*, 3*dim]

        q, k, v = y.split(self.dim, dim=-1)

        # LoRA on Q and V only; K stays frozen
        x_drop = self.q_lora.dropout(x)
        q = q + self.q_lora(x_drop)
        v = v + self.v_lora(x_drop)

        return torch.cat([q, k, v], dim=-1)

    # ------------------------------------------------------------------
    # Training-mode helpers
    # ------------------------------------------------------------------
    def set_lora_trainable(self, trainable: bool = True):
        for p in self.q_lora.parameters():
            p.requires_grad = trainable
        for p in self.v_lora.parameters():
            p.requires_grad = trainable

    def set_lora_dropout_enabled(self, enabled: bool):
        if isinstance(self.q_lora.dropout, nn.Dropout):
            self.q_lora.dropout.train(enabled)
        if isinstance(self.v_lora.dropout, nn.Dropout):
            self.v_lora.dropout.train(enabled)

    def num_lora_params(self) -> int:
        return sum(
            p.numel()
            for m in (self.q_lora, self.v_lora)
            for p in m.parameters()
            if p.requires_grad
        )


# ------------------------------------------------------------------
# Injection / removal
# ------------------------------------------------------------------

def inject_qv_lora_into_sam1(
    sam_model: nn.Module,
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.05,
    target_blocks: List[int] | None = None,
) -> Dict[int, FusedQKVLoRA]:
    """Inject Q/V LoRA into SAM1 ViT image encoder blocks.

    Args:
        sam_model: A fully-initialised Sam instance.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout probability.
        target_blocks: Block indices to inject (0-indexed).  None → all 12.

    Returns:
        Dict mapping block index → FusedQKVLoRA module.
    """
    encoder = sam_model.image_encoder
    if target_blocks is None:
        target_blocks = list(range(len(encoder.blocks)))

    injected: Dict[int, FusedQKVLoRA] = {}
    for idx in target_blocks:
        blk = encoder.blocks[idx]
        original_qkv = blk.attn.qkv

        # Freeze original
        for p in original_qkv.parameters():
            p.requires_grad = False

        lora = FusedQKVLoRA(original_qkv, rank=rank, alpha=alpha, dropout=dropout)
        blk.attn.qkv = lora
        injected[idx] = lora

    return injected


def remove_lora_from_sam1(sam_model: nn.Module) -> None:
    """Restore original `nn.Linear` qkv layers."""
    encoder = sam_model.image_encoder
    for blk in encoder.blocks:
        if isinstance(blk.attn.qkv, FusedQKVLoRA):
            blk.attn.qkv = blk.attn.qkv.qkv


def count_lora_parameters(sam_model: nn.Module) -> Tuple[int, Dict[int, int]]:
    """Count LoRA parameters per block and total.

    Returns:
        (total, {block_idx: count_per_block, ...})
    """
    per_block: Dict[int, int] = {}
    total = 0
    encoder = sam_model.image_encoder
    for idx, blk in enumerate(encoder.blocks):
        if isinstance(blk.attn.qkv, FusedQKVLoRA):
            n = blk.attn.qkv.num_lora_params()
            per_block[idx] = n
            total += n
    return total, per_block
