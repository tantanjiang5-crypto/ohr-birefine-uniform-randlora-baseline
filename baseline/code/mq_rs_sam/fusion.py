from __future__ import annotations

import torch
from torch import Tensor


def fuse_class_and_quality_scores(
    class_probability: Tensor,
    quality_probability: Tensor,
    *,
    class_exponent: float = 0.5,
    quality_exponent: float = 1.5,
    eps: float = 1e-8,
) -> Tensor:
    """Generalized geometric score fusion used for D-score ranking.

    The legacy project setting is class^0.5 * quality^1.5. Because MQ-RS makes
    class probability itself quality-aware, this function exposes both
    exponents so the legacy, balanced (1,1), and class-only (1,0) variants can
    be evaluated without changing the model architecture.
    """

    if class_exponent < 0 or quality_exponent < 0:
        raise ValueError("fusion exponents must be non-negative")
    class_probability = class_probability.float().clamp(eps, 1)
    quality_probability = quality_probability.float().clamp(eps, 1)
    return class_probability.pow(class_exponent) * quality_probability.pow(quality_exponent)
