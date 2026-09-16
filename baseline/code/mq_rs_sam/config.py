from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RegionMode = Literal["hard", "soft"]
SortScope = Literal["class", "global"]
ScoreMode = Literal["log_probability", "raw_logit", "probability"]
QJ2LossType = Literal["none", "bce", "smooth_l1", "mse"]


@dataclass(frozen=True)
class MaskQualityConfig:
    """Configuration for the detached multi-dimensional mask-quality target."""

    iou_weight: float = 0.5
    boundary_f1_weight: float = 0.3
    completeness_weight: float = 0.2
    region_mode: RegionMode = "hard"
    threshold: float = 0.5
    boundary_width: int = 1
    boundary_tolerance: int = 2
    eps: float = 1e-6
    min_positive_quality: float = 1e-4
    quality_power: float = 1.0

    def __post_init__(self) -> None:
        weights = (self.iou_weight, self.boundary_f1_weight, self.completeness_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("quality weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one quality weight must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if self.boundary_width < 1:
            raise ValueError("boundary_width must be >= 1")
        if self.boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be >= 0")
        if not 0 <= self.min_positive_quality < 1:
            raise ValueError("min_positive_quality must be in [0, 1)")
        if self.quality_power <= 0:
            raise ValueError("quality_power must be positive")

    @property
    def normalized_weights(self) -> tuple[float, float, float]:
        total = self.iou_weight + self.boundary_f1_weight + self.completeness_weight
        return (
            self.iou_weight / total,
            self.boundary_f1_weight / total,
            self.completeness_weight / total,
        )


@dataclass(frozen=True)
class RankSortConfig:
    """Configuration for MQ-RS classification-score ranking and sorting."""

    delta: float = 0.5
    score_mode: ScoreMode = "log_probability"
    sort_scope: SortScope = "class"
    rank_weight: float = 1.0
    sort_weight: float = 1.0
    eps: float = 1e-10

    def __post_init__(self) -> None:
        if self.delta < 0:
            raise ValueError("delta must be non-negative")
        if self.rank_weight < 0 or self.sort_weight < 0:
            raise ValueError("rank/sort weights must be non-negative")
        if self.rank_weight + self.sort_weight <= 0:
            raise ValueError("at least one of rank_weight and sort_weight must be positive")


@dataclass(frozen=True)
class MQRSRegularizerConfig:
    """Top-level configuration for the training-only MQ-RS regularizer."""

    lambda_rs: float = 0.3
    warmup_steps: int = 0
    ramp_steps: int = 0
    qj2_loss_type: QJ2LossType = "none"
    qj2_loss_weight: float = 0.0
    quality: MaskQualityConfig = MaskQualityConfig()
    rank_sort: RankSortConfig = RankSortConfig()

    def __post_init__(self) -> None:
        if self.lambda_rs < 0:
            raise ValueError("lambda_rs must be non-negative")
        if self.warmup_steps < 0 or self.ramp_steps < 0:
            raise ValueError("warmup_steps and ramp_steps must be non-negative")
        if self.qj2_loss_weight < 0:
            raise ValueError("qj2_loss_weight must be non-negative")
        if self.qj2_loss_type == "none" and self.qj2_loss_weight != 0:
            raise ValueError("qj2_loss_weight must be zero when qj2_loss_type='none'")

    def to_dict(self) -> dict:
        return asdict(self)
