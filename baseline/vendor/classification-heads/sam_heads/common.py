from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm used by the original SAM mask decoder."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d expects [B,C,H,W], got {tuple(x.shape)}")
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    """SAM-style MLP with ReLU on all but the final layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim, num_layers) <= 0:
            raise ValueError("MLP dimensions and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        hidden = [hidden_dim] * (num_layers - 1)
        dims_in = [input_dim] + hidden
        dims_out = hidden + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims_in, dims_out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x, inplace=False)
                x = self.dropout(x)
        return x


class TokenMLPClassifier(nn.Module):
    """Identical classifier used by the Mask-token and IoU-token baselines.

    Both baselines must consume the raw transformer token, not the output of
    SAM's pre-existing IoU prediction MLP. This guarantees identical depth and
    parameter count when input dimensions are equal.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: Optional[int] = None,
        num_linear_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("input_dim must be positive and num_classes must exceed one")
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_linear_layers,
            dropout=dropout,
        )

    def forward(self, token: Tensor) -> Tensor:
        if token.ndim != 2:
            raise ValueError(f"Token classifier expects [N,D], got {tuple(token.shape)}")
        if token.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected token dim {self.input_dim}, received {token.shape[-1]}"
            )
        return self.classifier(self.norm(token))


@dataclass(frozen=True)
class SelectionResult:
    token: Tensor
    index: Optional[Tensor]


def select_mask_token(
    mask_tokens: Tensor,
    *,
    iou_scores: Optional[Tensor] = None,
    strategy: str = "first",
    fixed_index: int = 0,
) -> SelectionResult:
    """Select one mask token for each prompted instance.

    Args:
        mask_tokens: [N,D] or [N,M,D].
        iou_scores: [N,M], required by strategy='best_iou'.
        strategy: 'first', 'fixed', 'best_iou', or 'mean'.
    """
    if mask_tokens.ndim == 2:
        return SelectionResult(mask_tokens, None)
    if mask_tokens.ndim != 3:
        raise ValueError(f"mask_tokens must be [N,D] or [N,M,D], got {tuple(mask_tokens.shape)}")

    n, m, _ = mask_tokens.shape
    if m == 0:
        raise ValueError("mask_tokens contains zero mask tokens")

    if strategy == "mean":
        return SelectionResult(mask_tokens.mean(dim=1), None)
    if strategy == "first":
        idx = torch.zeros(n, dtype=torch.long, device=mask_tokens.device)
    elif strategy == "fixed":
        if not 0 <= fixed_index < m:
            raise IndexError(f"fixed_index={fixed_index} outside [0,{m})")
        idx = torch.full((n,), fixed_index, dtype=torch.long, device=mask_tokens.device)
    elif strategy == "best_iou":
        if iou_scores is None:
            raise ValueError("iou_scores is required for strategy='best_iou'")
        if iou_scores.shape != (n, m):
            raise ValueError(f"Expected iou_scores {(n,m)}, got {tuple(iou_scores.shape)}")
        idx = iou_scores.argmax(dim=1)
    else:
        raise ValueError(f"Unknown token selection strategy: {strategy}")

    selected = mask_tokens[torch.arange(n, device=mask_tokens.device), idx]
    return SelectionResult(selected, idx)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
