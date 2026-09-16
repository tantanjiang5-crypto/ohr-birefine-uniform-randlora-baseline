from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import TQMOTPOutput


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax CE: CE(logits + log(train_class_counts), target)."""

    def __init__(self, class_counts: Tensor, reduction: str = "mean") -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or (counts <= 0).any():
            raise ValueError("class_counts must be a positive one-dimensional tensor")
        self.register_buffer("log_counts", torch.log(counts))
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.shape[-1] != self.log_counts.numel():
            raise ValueError("logit class dimension does not match class_counts")
        return F.cross_entropy(logits + self.log_counts.to(logits.dtype), target, reduction=self.reduction)


@dataclass(frozen=True)
class TQMOTPLossConfig:
    final_cls_weight: float = 1.0
    ot_aux_weight: float = 0.05
    token_aux_weight: float = 0.05
    ot_margin_weight: float = 0.15
    ot_margin: float = 0.10
    quality_weight: float = 0.05
    prototype_diversity_weight: float = 0.01
    prototype_similarity_threshold: float = 0.80
    quality_iou_low: float = 0.50
    quality_iou_high: float = 0.90


class TQMOTPLoss(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        class_counts: Optional[Tensor] = None,
        config: Optional[TQMOTPLossConfig] = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.config = config or TQMOTPLossConfig()
        self.classification = (
            BalancedSoftmaxLoss(class_counts) if class_counts is not None else nn.CrossEntropyLoss()
        )

    @staticmethod
    def _mask_iou_target(pred_logits: Tensor, gt_masks: Tensor) -> Tensor:
        if pred_logits.ndim == 3:
            pred_logits = pred_logits.unsqueeze(1)
        if gt_masks.ndim == 3:
            gt_masks = gt_masks.unsqueeze(1)
        if pred_logits.ndim != 4 or gt_masks.ndim != 4:
            raise ValueError("mask tensors must be [N,1,H,W] or [N,H,W]")
        if gt_masks.shape[-2:] != pred_logits.shape[-2:]:
            gt_masks = F.interpolate(gt_masks.float(), size=pred_logits.shape[-2:], mode="nearest")
        pred = (pred_logits.detach().sigmoid() >= 0.5).float()
        gt = (gt_masks.detach() >= 0.5).float()
        inter = (pred * gt).sum(dim=(-1, -2, -3))
        union = (pred + gt - pred * gt).sum(dim=(-1, -2, -3)).clamp_min(1.0e-6)
        return inter / union

    def _ot_margin(self, distances: Tensor, labels: Tensor) -> Tensor:
        positive = distances.gather(1, labels[:, None]).squeeze(1)
        negative = distances.masked_fill(
            F.one_hot(labels, num_classes=self.num_classes).bool(), float("inf")
        ).min(dim=1).values
        return F.relu(self.config.ot_margin + positive - negative).mean()

    def _prototype_diversity(self, prototypes: Mapping[str, Tensor]) -> Tensor:
        losses = []
        threshold = self.config.prototype_similarity_threshold
        for tensor in prototypes.values():
            p = F.normalize(tensor, dim=-1)
            sim = torch.matmul(p, p.transpose(-1, -2))
            eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool)
            off_diag = sim.masked_select(~eye.unsqueeze(0))
            if off_diag.numel() > 0:
                losses.append(F.relu(off_diag - threshold).mean())
        if not losses:
            first = next(iter(prototypes.values()))
            return first.new_zeros(())
        return torch.stack(losses).mean()

    def forward(
        self,
        output: TQMOTPOutput,
        labels: Tensor,
        *,
        pred_mask_logits: Optional[Tensor] = None,
        gt_masks: Optional[Tensor] = None,
        prototypes: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        if output.logits.shape[0] != labels.shape[0]:
            raise ValueError("output/label instance count mismatch")
        if output.logits.shape[-1] != self.num_classes:
            raise ValueError("output class dimension mismatch")

        loss_final = self.classification(output.logits, labels)
        loss_ot_aux = self.classification(output.ot_logits, labels)
        loss_token_aux = self.classification(output.token_logits, labels)
        loss_margin = self._ot_margin(output.ot_distances, labels)

        loss_quality = output.logits.new_zeros(())
        quality_target = None
        if pred_mask_logits is not None and gt_masks is not None:
            true_iou = self._mask_iou_target(pred_mask_logits, gt_masks)
            low, high = self.config.quality_iou_low, self.config.quality_iou_high
            quality_target = ((true_iou - low) / max(high - low, 1.0e-6)).clamp(0.0, 1.0)
            loss_quality = F.smooth_l1_loss(output.alpha.squeeze(-1), quality_target)

        proto_map = prototypes if prototypes is not None else output.normalized_prototypes
        loss_diversity = self._prototype_diversity(proto_map)

        total = (
            self.config.final_cls_weight * loss_final
            + self.config.ot_aux_weight * loss_ot_aux
            + self.config.token_aux_weight * loss_token_aux
            + self.config.ot_margin_weight * loss_margin
            + self.config.quality_weight * loss_quality
            + self.config.prototype_diversity_weight * loss_diversity
        )
        result = {
            "loss_cls_final": loss_final,
            "loss_cls_ot_aux": loss_ot_aux,
            "loss_cls_token_aux": loss_token_aux,
            "loss_ot_margin": loss_margin,
            "loss_quality_gate": loss_quality,
            "loss_prototype_diversity": loss_diversity,
            "loss_tqmotp_total": total,
        }
        if quality_target is not None:
            result["quality_target_mean"] = quality_target.detach().mean()
        return result
