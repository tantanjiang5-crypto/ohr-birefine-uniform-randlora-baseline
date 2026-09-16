from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MQRSRegularizerConfig
from .quality_targets import MaskQualityOutput, MaskQualityTargetBuilder
from .rank_sort import MultiClassRankSortLoss, RankSortOutput


@dataclass
class MQRSRegularizerOutput:
    loss: Tensor
    loss_mqrs: Tensor
    loss_mqrs_unweighted: Tensor
    loss_qj2: Tensor
    effective_lambda_rs: float
    quality: MaskQualityOutput
    rank_sort: RankSortOutput

    def logging_dict(self, prefix: str = "mqrs") -> dict[str, float | int | str]:
        def scalar(value: Tensor) -> float:
            return float(value.detach().float().cpu().item())

        return {
            f"{prefix}/loss": scalar(self.loss),
            f"{prefix}/loss_rs": scalar(self.loss_mqrs),
            f"{prefix}/rank_error": scalar(self.rank_sort.ranking_error),
            f"{prefix}/sort_error": scalar(self.rank_sort.sorting_error),
            f"{prefix}/loss_qj2": scalar(self.loss_qj2),
            f"{prefix}/lambda_rs": self.effective_lambda_rs,
            f"{prefix}/quality_mean": scalar(self.quality.raw_quality.mean()) if self.quality.raw_quality.numel() else 0.0,
            f"{prefix}/iou_mean": scalar(self.quality.iou.mean()) if self.quality.iou.numel() else 0.0,
            f"{prefix}/bf1_mean": scalar(self.quality.boundary_f1.mean()) if self.quality.boundary_f1.numel() else 0.0,
            f"{prefix}/completeness_mean": scalar(self.quality.completeness.mean()) if self.quality.completeness.numel() else 0.0,
            f"{prefix}/positive_count": self.rank_sort.positive_count,
            f"{prefix}/relevant_negative_count": self.rank_sort.relevant_negative_count,
            f"{prefix}/active_sort_classes": self.rank_sort.active_sort_classes,
            f"{prefix}/score_mode": self.rank_sort.score_mode,
            f"{prefix}/sort_scope": self.rank_sort.sort_scope,
        }


class MQRSRegularizer(nn.Module):
    """Training-only MQ-RS regularizer for TQ-MOTP + optional QJ-2.

    This module adds no parameters and is absent from inference. The mask-based
    quality target is always detached. QJ-2 regression is optional because an
    existing project may already supervise QJ-2 elsewhere; keeping it disabled
    avoids accidental duplicate loss terms.
    """

    def __init__(self, config: MQRSRegularizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or MQRSRegularizerConfig()
        self.quality_builder = MaskQualityTargetBuilder(self.config.quality)
        self.rank_sort_loss = MultiClassRankSortLoss(self.config.rank_sort)

    def effective_lambda(self, global_step: int | None) -> float:
        cfg = self.config
        if global_step is None:
            return cfg.lambda_rs
        if global_step < cfg.warmup_steps:
            return 0.0
        if cfg.ramp_steps <= 0:
            return cfg.lambda_rs
        progress = min(1.0, max(0.0, (global_step - cfg.warmup_steps) / cfg.ramp_steps))
        return cfg.lambda_rs * progress

    def _qj2_loss(self, quality_logit: Tensor | None, quality_target: Tensor) -> Tensor:
        cfg = self.config
        if cfg.qj2_loss_type == "none" or cfg.qj2_loss_weight == 0:
            if quality_logit is not None:
                return quality_logit.sum() * 0.0
            return quality_target.sum() * 0.0
        if quality_logit is None:
            raise ValueError("qj2_quality_logit is required when QJ-2 loss is enabled")
        quality_logit = quality_logit.float().reshape(-1)
        if quality_logit.shape != quality_target.shape:
            raise ValueError("qj2_quality_logit and quality target must be shape-aligned")
        if cfg.qj2_loss_type == "bce":
            base = F.binary_cross_entropy_with_logits(quality_logit, quality_target)
        else:
            prediction = quality_logit.sigmoid()
            if cfg.qj2_loss_type == "smooth_l1":
                base = F.smooth_l1_loss(prediction, quality_target)
            elif cfg.qj2_loss_type == "mse":
                base = F.mse_loss(prediction, quality_target)
            else:
                raise RuntimeError(f"unsupported qj2_loss_type: {cfg.qj2_loss_type}")
        return cfg.qj2_loss_weight * base

    def forward(
        self,
        *,
        class_logits: Tensor,
        labels: Tensor,
        pred_mask_logits: Tensor,
        gt_masks: Tensor,
        qj2_quality_logit: Tensor | None = None,
        global_step: int | None = None,
    ) -> MQRSRegularizerOutput:
        quality = self.quality_builder(pred_mask_logits, gt_masks)
        rank_sort = self.rank_sort_loss(class_logits, labels, quality.quality)
        lambda_rs = self.effective_lambda(global_step)
        loss_mqrs = rank_sort.loss * lambda_rs
        loss_qj2 = self._qj2_loss(qj2_quality_logit, quality.quality)
        total = loss_mqrs + loss_qj2
        return MQRSRegularizerOutput(
            loss=total,
            loss_mqrs=loss_mqrs,
            loss_mqrs_unweighted=rank_sort.loss,
            loss_qj2=loss_qj2,
            effective_lambda_rs=lambda_rs,
            quality=quality,
            rank_sort=rank_sort,
        )
