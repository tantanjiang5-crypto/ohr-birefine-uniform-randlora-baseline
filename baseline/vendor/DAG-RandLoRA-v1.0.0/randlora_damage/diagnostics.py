from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .core import FusedQKVRandLoRA, RandLoRALinear


@dataclass(frozen=True)
class RankDiagnostics:
    numerical_rank: int
    max_rank: int
    effective_rank: float
    spectral_energy_90_rank: int
    frobenius_norm: float


def matrix_rank_diagnostics(delta: torch.Tensor, *, tol: Optional[float] = None) -> RankDiagnostics:
    if delta.ndim != 2:
        raise ValueError(f"delta must be 2-D, got {tuple(delta.shape)}")
    if tol is not None and (not isinstance(tol, (int, float)) or tol < 0):
        raise ValueError("tol must be non-negative")
    matrix = delta.detach().float().cpu()
    if not torch.isfinite(matrix).all():
        raise ValueError("delta contains NaN or Inf")
    singular = torch.linalg.svdvals(matrix)
    numerical = int(
        torch.linalg.matrix_rank(matrix).item()
        if tol is None
        else torch.linalg.matrix_rank(matrix, tol=tol).item()
    )
    eps = torch.finfo(singular.dtype).eps
    raw_total = singular.sum()
    energy = singular.square()
    raw_energy = energy.sum()
    if raw_total <= eps or raw_energy <= eps:
        effective = 0.0
        energy90 = 0
    else:
        prob = singular / raw_total
        entropy = -(prob * prob.clamp_min(eps).log()).sum()
        effective = float(entropy.exp().item())
        cumulative = energy.cumsum(0) / raw_energy
        energy90 = int(
            torch.searchsorted(cumulative, torch.tensor(0.9, dtype=cumulative.dtype)).item()
        ) + 1
    return RankDiagnostics(
        numerical_rank=numerical,
        max_rank=min(matrix.shape),
        effective_rank=effective,
        spectral_energy_90_rank=energy90,
        frobenius_norm=float(matrix.norm().item()),
    )


def collect_update_rank_diagnostics(model: nn.Module) -> Dict[str, RankDiagnostics]:
    results: Dict[str, RankDiagnostics] = {}
    for name, module in model.named_modules():
        if isinstance(module, FusedQKVRandLoRA):
            for target, coeff in module.coefficients.items():
                results[f"{name}.{target}"] = matrix_rank_diagnostics(coeff.delta_weight())
        elif isinstance(module, RandLoRALinear):
            results[name] = matrix_rank_diagnostics(module.coefficients.delta_weight())
    return results


def audit_trainable_image_encoder(model_or_encoder: nn.Module) -> Dict[str, object]:
    encoder = getattr(model_or_encoder, "image_encoder", model_or_encoder)
    trainable = {name: p.numel() for name, p in encoder.named_parameters() if p.requires_grad}
    unexpected = {
        name: count
        for name, count in trainable.items()
        if "randlora_lambda" not in name and "randlora_gamma" not in name
    }
    return {
        "trainable_parameters": trainable,
        "trainable_total": sum(trainable.values()),
        "unexpected_trainables": unexpected,
        "pass": not unexpected and bool(trainable),
    }


def _as_binary_mask(
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    name: str = "mask",
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if mask.dtype == torch.bool:
        return mask
    if not torch.is_floating_point(mask):
        return mask != 0
    if not torch.isfinite(mask).all():
        raise ValueError(f"{name} contains NaN or Inf")
    values = mask.sigmoid() if from_logits else mask
    if not from_logits and (values.min() < 0 or values.max() > 1):
        raise ValueError(
            f"{name} is floating point but outside [0,1]; pass from_logits=True "
            "instead of relying on .bool(), which treats every non-zero logit as foreground"
        )
    return values >= threshold


def _validate_same_shape(*masks: torch.Tensor) -> None:
    shapes = {tuple(mask.shape) for mask in masks}
    if len(shapes) != 1:
        raise ValueError(f"mask shapes differ: {sorted(shapes)}")
    if masks[0].ndim < 3:
        raise ValueError("masks must include batch and at least two spatial dimensions")


def occupancy(gt_mask: torch.Tensor, box_mask: torch.Tensor) -> torch.Tensor:
    _validate_same_shape(gt_mask, box_mask)
    gt = _as_binary_mask(gt_mask, name="gt_mask")
    box = _as_binary_mask(box_mask, name="box_mask")
    dims = tuple(range(1, gt.ndim))
    return (gt & box).sum(dims).float() / box.sum(dims).clamp_min(1).float()


def box_background_false_positive_rate(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    box_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    pred_from_logits: bool = False,
) -> torch.Tensor:
    _validate_same_shape(pred_mask, gt_mask, box_mask)
    pred = _as_binary_mask(
        pred_mask, threshold=threshold, from_logits=pred_from_logits, name="pred_mask"
    )
    gt = _as_binary_mask(gt_mask, name="gt_mask")
    box = _as_binary_mask(box_mask, name="box_mask")
    dims = tuple(range(1, pred.ndim))
    background = box & ~gt
    false_positive = pred & background
    return false_positive.sum(dims).float() / background.sum(dims).clamp_min(1).float()


def oversegmentation_ratio(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    pred_from_logits: bool = False,
) -> torch.Tensor:
    _validate_same_shape(pred_mask, gt_mask)
    pred = _as_binary_mask(
        pred_mask, threshold=threshold, from_logits=pred_from_logits, name="pred_mask"
    )
    gt = _as_binary_mask(gt_mask, name="gt_mask")
    dims = tuple(range(1, pred.ndim))
    return pred.sum(dims).float() / gt.sum(dims).clamp_min(1).float()


def false_positive_area_ratio(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    pred_from_logits: bool = False,
) -> torch.Tensor:
    """False-positive area divided by GT foreground area."""
    _validate_same_shape(pred_mask, gt_mask)
    pred = _as_binary_mask(
        pred_mask, threshold=threshold, from_logits=pred_from_logits, name="pred_mask"
    )
    gt = _as_binary_mask(gt_mask, name="gt_mask")
    dims = tuple(range(1, pred.ndim))
    fp = pred & ~gt
    return fp.sum(dims).float() / gt.sum(dims).clamp_min(1).float()
