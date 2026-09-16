from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class RegionDescriptorOutput:
    descriptors: Tensor                 # [N, Kd, D]
    source_mass: Tensor                 # [N, Kd], sums to one
    selected_indices: Tensor            # [N, Kd]
    selected_region_weights: Tensor     # [N, Kd]
    region_coverage: Tensor             # [N, 1]


@dataclass
class PartialOTOutput:
    distance: Tensor                    # [N, C]
    plan: Optional[Tensor] = None        # [N, C, Kd, Kp+1]
    real_mass: Optional[Tensor] = None   # [N, C]
    dustbin_mass: Optional[Tensor] = None
    entropy: Optional[Tensor] = None     # [N, C]
    row_error: Optional[Tensor] = None
    col_error: Optional[Tensor] = None


@dataclass
class TQMOTPOutput:
    logits: Tensor                      # [N, C]
    ot_logits: Tensor                   # [N, C]
    token_logits: Tensor                # [N, C]
    ot_distances: Tensor                # [N, C]
    alpha: Tensor                       # [N, 1]
    region_weights: Tensor              # [N, 3]
    rho: Tensor                         # [N, 3]
    quality_features: Tensor            # [N, Q]
    mask_prob_roi: Tensor               # [N, 1, R, R]
    region_masks: Dict[str, Tensor]
    descriptor_outputs: Dict[str, RegionDescriptorOutput]
    region_ot: Dict[str, PartialOTOutput]
    normalized_prototypes: Dict[str, Tensor]
    diagnostics: Dict[str, Tensor] = field(default_factory=dict)
