from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class ExperimentMetrics:
    d_ap75: float
    low_occupancy_mean_iou: float
    classification_accuracy: float
    box_background_fpr: float
    oversegmentation_ratio: float
    classification_metric: Optional[float] = None

    def validate(self, name: str = "metrics") -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"{name}.{field.name} must be finite")
        if not 0.0 <= self.d_ap75 <= 1.0:
            raise ValueError(f"{name}.d_ap75 must be in [0,1]")
        if not 0.0 <= self.low_occupancy_mean_iou <= 1.0:
            raise ValueError(f"{name}.low_occupancy_mean_iou must be in [0,1]")
        if not 0.0 <= self.classification_accuracy <= 1.0:
            raise ValueError(f"{name}.classification_accuracy must be in [0,1]")
        if not 0.0 <= self.box_background_fpr <= 1.0:
            raise ValueError(f"{name}.box_background_fpr must be in [0,1]")
        if self.oversegmentation_ratio < 0:
            raise ValueError(f"{name}.oversegmentation_ratio must be non-negative")


@dataclass(frozen=True)
class GateDecision:
    pass_gate: bool
    stop_early: bool
    reasons: tuple[str, ...]


def evaluate_experiment_gate(
    baseline: ExperimentMetrics,
    candidate: ExperimentMetrics,
    *,
    epoch: int,
) -> GateDecision:
    """Encode the proposed pass/stop rules without controlling the trainer."""
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    baseline.validate("baseline")
    candidate.validate("candidate")

    ap75_gain = candidate.d_ap75 - baseline.d_ap75
    low_occ_gain = candidate.low_occupancy_mean_iou - baseline.low_occupancy_mean_iou
    accuracy_drop = candidate.classification_accuracy - baseline.classification_accuracy
    fpr_change = candidate.box_background_fpr - baseline.box_background_fpr
    overseg_change = (
        candidate.oversegmentation_ratio / max(baseline.oversegmentation_ratio, 1e-12) - 1.0
    )

    quality_pass = ap75_gain >= 0.010 or low_occ_gain >= 0.020
    safeguards_pass = accuracy_drop >= -0.005 and fpr_change <= 0.0
    reasons = [
        "mask quality threshold reached"
        if quality_pass
        else "mask quality threshold not reached"
    ]
    if accuracy_drop < -0.005:
        reasons.append("classification accuracy dropped by more than 0.5 percentage points")
    if fpr_change > 0:
        reasons.append("box-background false-positive rate increased")

    stop = False
    if epoch >= 15 and ap75_gain < 0.005:
        stop = True
        reasons.append("epoch>=15 and D AP75 gain is below 0.005")
    if overseg_change > 0.05:
        stop = True
        reasons.append("oversegmentation ratio increased by more than 5%")
    if (
        candidate.classification_metric is not None
        and baseline.classification_metric is not None
        and candidate.classification_metric > baseline.classification_metric
        and ap75_gain <= 0
        and low_occ_gain <= 0
    ):
        stop = True
        reasons.append("only classification improved; mask metrics did not improve")

    return GateDecision(
        pass_gate=quality_pass and safeguards_pass,
        stop_early=stop,
        reasons=tuple(reasons),
    )
