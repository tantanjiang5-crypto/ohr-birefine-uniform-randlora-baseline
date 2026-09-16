from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def multiclass_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float = 2.0,
    alpha: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Expected logits [N,C] and targets [N]")
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    loss = (1.0 - pt).pow(gamma) * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be mean, sum, or none")


def dice_loss_with_logits(logits: Tensor, targets: Tensor, eps: float = 1.0) -> Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"Mask logits/targets shape mismatch: {logits.shape} vs {targets.shape}")
    probs = logits.sigmoid().flatten(start_dim=1)
    targets = targets.float().flatten(start_dim=1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


class MultiHeadClassificationLoss(nn.Module):
    """Unified closed-set loss for the five-head horizontal comparison."""

    def __init__(
        self,
        *,
        class_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        head_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.label_smoothing = float(label_smoothing)
        self.head_weights = dict(head_weights or {})

    def forward(self, logits_by_head: Mapping[str, Tensor], targets: Tensor) -> Dict[str, Tensor]:
        if not logits_by_head:
            raise ValueError("logits_by_head must not be empty")
        losses: Dict[str, Tensor] = {}
        weighted_sum = targets.new_zeros((), dtype=torch.float32)
        total_weight = 0.0
        for name, logits in logits_by_head.items():
            loss = F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            weight = float(self.head_weights.get(name, 1.0))
            if weight < 0:
                raise ValueError(f"Negative head weight for {name}")
            losses[f"loss_cls_{name}"] = loss
            weighted_sum = weighted_sum + weight * loss
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("At least one head must have positive weight")
        losses["loss_cls_total"] = weighted_sum / total_weight
        return losses


@dataclass(frozen=True)
class JointLossWeights:
    classification: float = 1.0
    mask_bce: float = 1.0
    mask_dice: float = 1.0
    cwsam_dense: float = 0.0


class JointSAMLoss(nn.Module):
    """Joint classification + binary mask + optional CWSAM dense CE loss."""

    def __init__(
        self,
        classification_loss: MultiHeadClassificationLoss,
        *,
        weights: JointLossWeights = JointLossWeights(),
        dense_ignore_index: int = -100,
        dense_class_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.classification_loss = classification_loss
        self.weights = weights
        self.dense_ignore_index = int(dense_ignore_index)
        if dense_class_weights is not None:
            self.register_buffer("dense_class_weights", dense_class_weights.float())
        else:
            self.dense_class_weights = None

    def forward(
        self,
        *,
        logits_by_head: Mapping[str, Tensor],
        class_targets: Tensor,
        mask_logits: Optional[Tensor] = None,
        mask_targets: Optional[Tensor] = None,
        cwsam_dense_logits: Optional[Tensor] = None,
        dense_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        losses = self.classification_loss(logits_by_head, class_targets)
        total = self.weights.classification * losses["loss_cls_total"]

        if (mask_logits is None) != (mask_targets is None):
            raise ValueError("mask_logits and mask_targets must be supplied together")
        if mask_logits is not None:
            bce = F.binary_cross_entropy_with_logits(mask_logits, mask_targets.float())
            dice = dice_loss_with_logits(mask_logits, mask_targets)
            losses["loss_mask_bce"] = bce
            losses["loss_mask_dice"] = dice
            total = total + self.weights.mask_bce * bce + self.weights.mask_dice * dice

        if (cwsam_dense_logits is None) != (dense_targets is None):
            raise ValueError("cwsam_dense_logits and dense_targets must be supplied together")
        if cwsam_dense_logits is not None:
            dense = F.cross_entropy(
                cwsam_dense_logits,
                dense_targets.long(),
                weight=self.dense_class_weights,
                ignore_index=self.dense_ignore_index,
            )
            losses["loss_cwsam_dense"] = dense
            total = total + self.weights.cwsam_dense * dense

        losses["loss_total"] = total
        return losses
