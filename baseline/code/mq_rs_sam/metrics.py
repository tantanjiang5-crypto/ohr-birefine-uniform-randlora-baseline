from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class QualityCalibrationMetrics:
    spearman: float
    pearson: float
    brier: float
    soft_ece: float
    count: int


def _average_ranks(values: Tensor) -> Tensor:
    values = values.detach().float().reshape(-1)
    n = values.numel()
    if n == 0:
        return values
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty(n, device=values.device, dtype=torch.float32)
    start = 0
    while start < n:
        end = start + 1
        while end < n and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        average = (start + end - 1) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def pearson_correlation(x: Tensor, y: Tensor, eps: float = 1e-12) -> Tensor:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    if x.shape != y.shape or x.numel() < 2:
        return x.new_tensor(float("nan"))
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = torch.sqrt((xc.square().sum()) * (yc.square().sum())).clamp_min(eps)
    return (xc * yc).sum() / denominator


def spearman_correlation(x: Tensor, y: Tensor, eps: float = 1e-12) -> Tensor:
    if x.numel() < 2:
        return x.new_tensor(float("nan"), dtype=torch.float32)
    return pearson_correlation(_average_ranks(x), _average_ranks(y), eps=eps)


def soft_expected_calibration_error(
    confidence: Tensor,
    target: Tensor,
    *,
    bins: int = 15,
) -> Tensor:
    """ECE with a continuous target in [0,1], suitable for mask quality."""

    confidence = confidence.float().reshape(-1).clamp(0, 1)
    target = target.float().reshape(-1).clamp(0, 1)
    if confidence.shape != target.shape:
        raise ValueError("confidence and target must have the same shape")
    if confidence.numel() == 0:
        return confidence.new_tensor(0.0)
    boundaries = torch.linspace(0, 1, bins + 1, device=confidence.device)
    ece = confidence.new_tensor(0.0)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        count = in_bin.sum()
        if count > 0:
            weight = count.float() / confidence.numel()
            ece = ece + weight * (confidence[in_bin].mean() - target[in_bin].mean()).abs()
    return ece


def quality_calibration_metrics(
    class_logits: Tensor,
    labels: Tensor,
    quality: Tensor,
    *,
    bins: int = 15,
) -> QualityCalibrationMetrics:
    labels = labels.to(class_logits.device, dtype=torch.long).reshape(-1)
    quality = quality.to(class_logits.device, dtype=torch.float32).reshape(-1)
    probability = class_logits.float().softmax(dim=-1)
    true_class_probability = probability.gather(1, labels[:, None]).squeeze(1)
    return QualityCalibrationMetrics(
        spearman=float(spearman_correlation(true_class_probability, quality).cpu().item()),
        pearson=float(pearson_correlation(true_class_probability, quality).cpu().item()),
        brier=float((true_class_probability - quality).square().mean().cpu().item()) if quality.numel() else 0.0,
        soft_ece=float(soft_expected_calibration_error(true_class_probability, quality, bins=bins).cpu().item()),
        count=int(quality.numel()),
    )
