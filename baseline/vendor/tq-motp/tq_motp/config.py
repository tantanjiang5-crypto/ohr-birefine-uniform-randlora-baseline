from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class TQMOTPConfig:
    """Configuration for the TQ-MOTP classification head.

    The defaults target SAM/SAM2 decoder tokens with dimension 256 and the
    current 8-class vehicle-damage setup. All spatial coordinates passed to
    the head must use the padded SAM input coordinate system.
    """

    num_classes: int = 8
    token_dim: int = 256
    image_channels: int = 256
    roi_size: int = 28
    context_scale: float = 1.25

    semantic_channels: int = 128
    detail_channels: int = 128
    fused_channels: int = 256
    ot_dim: int = 128
    quality_token_dim: int = 32
    shape_dim: int = 64
    classifier_hidden_dim: int = 512
    dropout: float = 0.10

    use_detail_stem: bool = True
    detail_roi_scale: int = 2
    detach_mask_for_classification: bool = True
    pred_iou_is_logit: bool = False

    descriptor_counts: Dict[str, int] = field(
        default_factory=lambda: {"fg": 16, "bd": 12, "ctx": 8}
    )
    prototype_counts: Dict[str, int] = field(
        default_factory=lambda: {"fg": 12, "bd": 8, "ctx": 6}
    )
    rho_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "fg": (0.10, 0.95),
            "bd": (0.05, 0.85),
            "ctx": (0.05, 0.65),
        }
    )
    rho_init: Dict[str, float] = field(
        default_factory=lambda: {"fg": 0.72, "bd": 0.52, "ctx": 0.32}
    )

    mask_temperature: float = 1.0
    descriptor_temperature: float = 0.20
    token_similarity_weight: float = 0.50
    region_eps: float = 1.0e-6

    sinkhorn_epsilon: float = 0.07
    sinkhorn_iterations: int = 20
    sinkhorn_tolerance: float = 0.0
    dustbin_cost: float = 0.15
    ot_temperature_init: float = 0.07
    ot_temperature_min: float = 0.01

    fusion_alpha_init: float = 0.40
    return_transport: bool = True

    def validate(self) -> None:
        regions = {"fg", "bd", "ctx"}
        if set(self.descriptor_counts) != regions:
            raise ValueError("descriptor_counts must contain exactly fg, bd, ctx")
        if set(self.prototype_counts) != regions:
            raise ValueError("prototype_counts must contain exactly fg, bd, ctx")
        if set(self.rho_ranges) != regions or set(self.rho_init) != regions:
            raise ValueError("rho configuration must contain exactly fg, bd, ctx")
        if self.roi_size <= 1:
            raise ValueError("roi_size must be > 1")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        if self.ot_dim <= 0 or self.token_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        for region in regions:
            if self.descriptor_counts[region] <= 0:
                raise ValueError(f"descriptor count for {region} must be positive")
            if self.prototype_counts[region] <= 0:
                raise ValueError(f"prototype count for {region} must be positive")
            lo, hi = self.rho_ranges[region]
            init = self.rho_init[region]
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(f"invalid rho range for {region}: {(lo, hi)}")
            if not (lo <= init <= hi):
                raise ValueError(f"rho_init for {region} must lie inside its range")
