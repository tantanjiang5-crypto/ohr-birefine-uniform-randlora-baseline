from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MaskQualityConfig


@dataclass
class MaskQualityOutput:
    quality: Tensor
    raw_quality: Tensor
    iou: Tensor
    boundary_f1: Tensor
    completeness: Tensor
    boundary_precision: Tensor
    boundary_recall: Tensor


def _ensure_nhw(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim == 4:
        if tensor.shape[1] != 1:
            raise ValueError(f"{name} with 4 dimensions must have channel size 1, got {tuple(tensor.shape)}")
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [N,H,W] or [N,1,H,W], got {tuple(tensor.shape)}")
    return tensor


def _dilate(mask: Tensor, radius: int) -> Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _erode(mask: Tensor, radius: int) -> Tensor:
    if radius <= 0:
        return mask
    return 1.0 - _dilate(1.0 - mask, radius)


def binary_boundary(mask: Tensor, width: int = 1) -> Tensor:
    """Return a binary morphological boundary for masks shaped [N,1,H,W]."""

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [N,1,H,W], got {tuple(mask.shape)}")
    mask = (mask > 0.5).to(dtype=torch.float32)
    boundary = (_dilate(mask, width) - _erode(mask, width)).clamp_(0, 1)
    return boundary


def _safe_ratio(numerator: Tensor, denominator: Tensor, eps: float) -> Tensor:
    return numerator / denominator.clamp_min(eps)


def boundary_f1_score(
    pred_binary: Tensor,
    gt_binary: Tensor,
    *,
    width: int = 1,
    tolerance: int = 2,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-instance boundary F1 with a pixel tolerance band.

    Both inputs must be [N,1,H,W] binary-like tensors. A prediction boundary
    pixel is correct when it lies within ``tolerance`` pixels of a GT boundary,
    and vice versa for recall.
    """

    pred_boundary = binary_boundary(pred_binary, width=width)
    gt_boundary = binary_boundary(gt_binary, width=width)
    pred_match_band = _dilate(gt_boundary, tolerance)
    gt_match_band = _dilate(pred_boundary, tolerance)

    dims = (1, 2, 3)
    pred_count = pred_boundary.sum(dims)
    gt_count = gt_boundary.sum(dims)
    matched_pred = (pred_boundary * pred_match_band).sum(dims)
    matched_gt = (gt_boundary * gt_match_band).sum(dims)

    precision = _safe_ratio(matched_pred, pred_count, eps)
    recall = _safe_ratio(matched_gt, gt_count, eps)

    both_empty = (pred_count <= eps) & (gt_count <= eps)
    precision = torch.where(both_empty, torch.ones_like(precision), precision)
    recall = torch.where(both_empty, torch.ones_like(recall), recall)
    f1 = _safe_ratio(2 * precision * recall, precision + recall, eps)
    f1 = torch.where(both_empty, torch.ones_like(f1), f1)
    return f1.clamp(0, 1), precision.clamp(0, 1), recall.clamp(0, 1)


class MaskQualityTargetBuilder(nn.Module):
    """Build detached IoU + Boundary-F1 + completeness quality targets.

    ``completeness`` is mask recall, |P∩G| / |G|. The inverse expression
    |G| / |P∩G| is intentionally not used because it is unbounded and reverses
    the desired quality direction.
    """

    def __init__(self, config: MaskQualityConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskQualityConfig()

    @torch.no_grad()
    def forward(self, pred_mask_logits: Tensor, gt_masks: Tensor) -> MaskQualityOutput:
        pred_mask_logits = _ensure_nhw(pred_mask_logits, "pred_mask_logits")
        gt_masks = _ensure_nhw(gt_masks, "gt_masks")
        if pred_mask_logits.shape != gt_masks.shape:
            raise ValueError(
                "pred_mask_logits and gt_masks must be instance-aligned and spatially equal; "
                f"got {tuple(pred_mask_logits.shape)} and {tuple(gt_masks.shape)}"
            )
        if pred_mask_logits.numel() == 0:
            empty = pred_mask_logits.new_empty((0,), dtype=torch.float32)
            return MaskQualityOutput(empty, empty, empty, empty, empty, empty, empty)

        cfg = self.config
        pred_prob = pred_mask_logits.float().sigmoid()
        gt_float = gt_masks.float().clamp(0, 1)
        pred_binary = (pred_prob >= cfg.threshold).float()
        gt_binary = (gt_float >= 0.5).float()

        if cfg.region_mode == "soft":
            pred_region = pred_prob
            gt_region = gt_float
            intersection = (pred_region * gt_region).sum(dim=(1, 2))
            union = (pred_region + gt_region - pred_region * gt_region).sum(dim=(1, 2))
            gt_area = gt_region.sum(dim=(1, 2))
        else:
            intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
            union = ((pred_binary + gt_binary) > 0).float().sum(dim=(1, 2))
            gt_area = gt_binary.sum(dim=(1, 2))

        pred_area = pred_binary.sum(dim=(1, 2))
        both_empty = (union <= cfg.eps) & (gt_area <= cfg.eps) & (pred_area <= cfg.eps)
        iou = _safe_ratio(intersection, union, cfg.eps)
        iou = torch.where(both_empty, torch.ones_like(iou), iou).clamp(0, 1)

        # Completeness is recall: predicted coverage of the GT region.
        completeness = _safe_ratio(intersection, gt_area, cfg.eps)
        completeness = torch.where(both_empty, torch.ones_like(completeness), completeness).clamp(0, 1)

        boundary_f1, boundary_precision, boundary_recall = boundary_f1_score(
            pred_binary[:, None],
            gt_binary[:, None],
            width=cfg.boundary_width,
            tolerance=cfg.boundary_tolerance,
            eps=cfg.eps,
        )

        wiou, wbf1, wcomplete = cfg.normalized_weights
        raw_quality = (wiou * iou + wbf1 * boundary_f1 + wcomplete * completeness).clamp(0, 1)
        quality = raw_quality.pow(cfg.quality_power).clamp(cfg.min_positive_quality, 1.0)
        return MaskQualityOutput(
            quality=quality.detach(),
            raw_quality=raw_quality.detach(),
            iou=iou.detach(),
            boundary_f1=boundary_f1.detach(),
            completeness=completeness.detach(),
            boundary_precision=boundary_precision.detach(),
            boundary_recall=boundary_recall.detach(),
        )
