from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .types import PartialOTOutput


def _safe_log(x: Tensor, eps: float = 1.0e-12) -> Tensor:
    return torch.log(x.clamp_min(eps))


def partial_sinkhorn_prototype_distance(
    descriptors: Tensor,
    source_mass: Tensor,
    prototypes: Tensor,
    rho: Tensor,
    *,
    epsilon: float = 0.07,
    iterations: int = 20,
    dustbin_cost: float = 0.15,
    tolerance: float = 0.0,
    return_plan: bool = True,
) -> PartialOTOutput:
    """Differentiable typed Partial OT using a fixed-mass dustbin column.

    Args:
        descriptors: normalized [N,Kd,D]
        source_mass: [N,Kd], positive and row-normalized
        prototypes: normalized [C,Kp,D]
        rho: [N,1] or [N], requested real-prototype transported mass in (0,1)

    The target marginal contains Kp equal real-prototype masses summing to rho,
    plus one dustbin mass 1-rho. Each class solves an independent OT problem.
    """
    if descriptors.ndim != 3 or prototypes.ndim != 3:
        raise ValueError("descriptors and prototypes must be rank-3")
    n, kd, dim = descriptors.shape
    classes, kp, pdim = prototypes.shape
    if dim != pdim:
        raise ValueError("descriptor/prototype dimension mismatch")
    if source_mass.shape != (n, kd):
        raise ValueError("source_mass must have shape [N,Kd]")
    if rho.ndim == 1:
        rho = rho.unsqueeze(-1)
    if rho.shape != (n, 1):
        raise ValueError("rho must have shape [N] or [N,1]")
    if epsilon <= 0 or iterations <= 0:
        raise ValueError("epsilon and iterations must be positive")

    if n == 0:
        empty_nc = descriptors.new_empty((0, classes))
        return PartialOTOutput(distance=empty_nc)

    # Keep OT arithmetic in float32 even under AMP.
    original_dtype = descriptors.dtype
    d = F.normalize(descriptors.float(), dim=-1)
    p = F.normalize(prototypes.float(), dim=-1)
    a = source_mass.float().clamp_min(1.0e-12)
    a = a / a.sum(dim=-1, keepdim=True)
    rho32 = rho.float().clamp(1.0e-4, 1.0 - 1.0e-4)

    real_cost = 1.0 - torch.einsum("nkd,cpd->nckp", d, p)  # [N,C,Kd,Kp]
    dust = torch.full((n, classes, kd, 1), float(dustbin_cost), device=d.device, dtype=d.dtype)
    cost = torch.cat([real_cost, dust], dim=-1)

    target_real = rho32 / float(kp)
    b_real = target_real[:, None, :].expand(n, classes, kp)
    b_dust = (1.0 - rho32)[:, None, :].expand(n, classes, 1)
    b = torch.cat([b_real, b_dust], dim=-1)  # [N,C,Kp+1]
    a_nc = a[:, None, :].expand(n, classes, kd)

    flat_cost = cost.reshape(n * classes, kd, kp + 1)
    flat_a = a_nc.reshape(n * classes, kd)
    flat_b = b.reshape(n * classes, kp + 1)

    log_k = -flat_cost / float(epsilon)
    log_a = _safe_log(flat_a)
    log_b = _safe_log(flat_b)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(iterations):
        old_u: Optional[Tensor] = log_u if tolerance > 0 else None
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(1), dim=2)
        log_v = log_b - torch.logsumexp(log_k + log_u.unsqueeze(2), dim=1)
        if tolerance > 0 and old_u is not None:
            if (log_u - old_u).abs().amax().item() < tolerance:
                break

    log_plan = log_u.unsqueeze(2) + log_k + log_v.unsqueeze(1)
    plan = torch.exp(log_plan).reshape(n, classes, kd, kp + 1)

    real_plan = plan[..., :kp]
    actual_real_mass = real_plan.sum(dim=(-1, -2)).clamp_min(1.0e-8)
    numerator = (real_plan * real_cost).sum(dim=(-1, -2))
    distance = numerator / actual_real_mass

    dustbin_mass = plan[..., -1].sum(dim=-1)
    entropy = -(plan.clamp_min(1.0e-12) * torch.log(plan.clamp_min(1.0e-12))).sum(dim=(-1, -2))
    row_error = (plan.sum(dim=-1) - a_nc).abs().amax(dim=-1)
    col_error = (plan.sum(dim=-2) - b).abs().amax(dim=-1)

    return PartialOTOutput(
        distance=distance.to(original_dtype),
        plan=plan.to(original_dtype) if return_plan else None,
        real_mass=actual_real_mass.to(original_dtype),
        dustbin_mass=dustbin_mass.to(original_dtype),
        entropy=entropy.to(original_dtype),
        row_error=row_error.to(original_dtype),
        col_error=col_error.to(original_dtype),
    )
