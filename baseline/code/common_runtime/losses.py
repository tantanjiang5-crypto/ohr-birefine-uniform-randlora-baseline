from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from mq_rs_sam.config import MaskQualityConfig, MQRSRegularizerConfig, RankSortConfig
from mq_rs_sam.project_adapter import compute_for_tqmotp_qj2_output, flatten_instance_targets
from mq_rs_sam.regularizer import MQRSRegularizer, MQRSRegularizerOutput


@dataclass
class LossBundle:
    total: Tensor
    parts: dict[str, Tensor]
    mqrs_output: MQRSRegularizerOutput | None


def selected_mask_metrics(pred_mask_logits: Tensor, gt_masks: Tensor) -> dict[str, Tensor]:
    if pred_mask_logits.shape != gt_masks.shape:
        raise ValueError(f"mask shape mismatch: {pred_mask_logits.shape} vs {gt_masks.shape}")
    probability = pred_mask_logits.float().sigmoid()
    target = gt_masks.float().clamp(0, 1)
    hard = (pred_mask_logits.detach() > 0.0).float()
    intersection_hard = (hard * target).sum(dim=(-2, -1))
    union_hard = ((hard + target) > 0).float().sum(dim=(-2, -1))
    iou = intersection_hard / union_hard.clamp_min(1e-6)
    dice_hard = 2.0 * intersection_hard / (hard.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))).clamp_min(1e-6)
    intersection_soft = (probability * target).sum(dim=(-2, -1))
    dice_loss = (
        1.0
        - (2.0 * intersection_soft + 1.0)
        / (probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1.0)
    ).mean()
    bce = F.binary_cross_entropy_with_logits(pred_mask_logits.float(), target)
    return {
        "mask_bce": bce,
        "mask_dice_loss": dice_loss,
        "hard_iou_per_instance": iou.detach(),
        "hard_dice_per_instance": dice_hard.detach(),
    }


def build_mqrs_regularizer(config: dict[str, Any]) -> MQRSRegularizer:
    quality_cfg = config.get("quality", {})
    rank_cfg = config.get("rank_sort", {})
    regularizer_config = MQRSRegularizerConfig(
        lambda_rs=float(config.get("lambda_rs", 0.3)),
        warmup_steps=int(config.get("warmup_steps", 0)),
        ramp_steps=int(config.get("ramp_steps", 1000)),
        qj2_loss_type="none",
        qj2_loss_weight=0.0,
        quality=MaskQualityConfig(
            iou_weight=float(quality_cfg.get("iou_weight", 0.5)),
            boundary_f1_weight=float(quality_cfg.get("boundary_f1_weight", 0.3)),
            completeness_weight=float(quality_cfg.get("completeness_weight", 0.2)),
            region_mode=str(quality_cfg.get("region_mode", "hard")),
            threshold=float(quality_cfg.get("threshold", 0.5)),
            boundary_width=int(quality_cfg.get("boundary_width", 1)),
            boundary_tolerance=int(quality_cfg.get("boundary_tolerance", 2)),
        ),
        rank_sort=RankSortConfig(
            delta=float(rank_cfg.get("delta", 0.5)),
            score_mode=str(rank_cfg.get("score_mode", "log_probability")),
            sort_scope=str(rank_cfg.get("sort_scope", "class")),
            rank_weight=float(rank_cfg.get("rank_weight", 1.0)),
            sort_weight=float(rank_cfg.get("sort_weight", 1.0)),
        ),
    )
    return MQRSRegularizer(regularizer_config)


def compute_joint_losses(
    *,
    output: Any,
    labels_list: list[Tensor],
    gt_masks_list: list[Tensor],
    tqmotp_loss_fn: Any,
    loss_cfg: dict[str, Any],
    mqrs_enabled: bool,
    mqrs_regularizer: MQRSRegularizer,
    global_step: int,
) -> LossBundle:
    targets = flatten_instance_targets(labels_list, gt_masks_list, device=output.pred_masks.device)
    if targets.labels.numel() == 0:
        zero = output.pred_masks.sum() * 0.0
        return LossBundle(total=zero, parts={"total": zero}, mqrs_output=None)
    if output.class_logits.shape[0] != targets.labels.numel():
        raise RuntimeError("classification/target instance count mismatch")

    mask_metrics = selected_mask_metrics(output.pred_masks, targets.gt_masks)
    hard_iou = mask_metrics["hard_iou_per_instance"]
    qj2_loss = F.smooth_l1_loss(output.effective_quality.float(), hard_iou)
    native_iou_loss = F.smooth_l1_loss(output.original_pred_iou.float(), hard_iou)

    tq_losses = tqmotp_loss_fn(
        output.tqmotp_output,
        targets.labels,
        pred_mask_logits=output.pred_masks,
        gt_masks=targets.gt_masks,
    )
    if "loss_tqmotp_total" not in tq_losses:
        raise KeyError(f"TQMOTPLoss output lacks loss_tqmotp_total; keys={sorted(tq_losses)}")

    base_total = (
        float(loss_cfg.get("mask_bce_weight", 1.0)) * mask_metrics["mask_bce"]
        + float(loss_cfg.get("mask_dice_weight", 1.0)) * mask_metrics["mask_dice_loss"]
        + float(loss_cfg.get("qj2_weight", 0.1)) * qj2_loss
        + float(loss_cfg.get("native_iou_weight", 0.0)) * native_iou_loss
        + float(loss_cfg.get("tqmotp_weight", 1.0)) * tq_losses["loss_tqmotp_total"]
    )

    mqrs_output: MQRSRegularizerOutput | None = None
    if mqrs_enabled:
        mqrs_output = compute_for_tqmotp_qj2_output(
            mqrs_regularizer, output, targets, global_step=global_step
        )
        total = base_total + mqrs_output.loss
    else:
        total = base_total

    parts: dict[str, Tensor] = {
        "total": total,
        "base_total": base_total,
        "mask_bce": mask_metrics["mask_bce"],
        "mask_dice_loss": mask_metrics["mask_dice_loss"],
        "qj2_loss": qj2_loss,
        "native_iou_loss": native_iou_loss,
        "mean_train_hard_iou": hard_iou.mean(),
        "mean_train_hard_dice": mask_metrics["hard_dice_per_instance"].mean(),
    }
    for name, value in tq_losses.items():
        if torch.is_tensor(value) and value.ndim == 0:
            parts[f"tqmotp/{name}"] = value
    if mqrs_output is not None:
        parts.update(
            {
                "mqrs/loss": mqrs_output.loss,
                "mqrs/loss_weighted": mqrs_output.loss_mqrs,
                "mqrs/loss_unweighted": mqrs_output.loss_mqrs_unweighted,
                "mqrs/rank_error": mqrs_output.rank_sort.ranking_error,
                "mqrs/sort_error": mqrs_output.rank_sort.sorting_error,
                "mqrs/quality_mean": mqrs_output.quality.raw_quality.mean(),
            }
        )
    else:
        zero = base_total.detach() * 0.0
        parts.update(
            {
                "mqrs/loss": zero,
                "mqrs/loss_weighted": zero,
                "mqrs/loss_unweighted": zero,
                "mqrs/rank_error": zero,
                "mqrs/sort_error": zero,
                "mqrs/quality_mean": zero,
            }
        )
    return LossBundle(total=total, parts=parts, mqrs_output=mqrs_output)
