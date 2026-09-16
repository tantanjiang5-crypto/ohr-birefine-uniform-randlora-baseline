from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from .allocation import allocate_block_ranks, allocate_target_ranks
from .config import (
    RandLoRADamageConfig,
    canonical_target_key,
    parse_target_key,
    randlora_trainable_params,
)
from .core import FusedQKVRandLoRA
from .sam_injector import _resolve_image_encoder


DEFAULT_RANK_CANDIDATES = (56, 64, 70, 80, 96)


def _resolve_fused_qkv_linear(block: nn.Module) -> nn.Linear:
    qkv = block.attn.qkv
    if isinstance(qkv, FusedQKVRandLoRA):
        if qkv.merged:
            raise RuntimeError("gradient profiling requires an unmerged RandLoRA wrapper")
        qkv = qkv.base_layer
    if not isinstance(qkv, nn.Linear) or qkv.out_features != 3 * qkv.in_features:
        raise TypeError("DAG profiler requires official SAM1 fused qkv nn.Linear with shape [3d,d]")
    return qkv


class FusedQKVWeightGradientProfiler:
    """Collect exact Q/V full-weight gradients without updating the encoder.

    The profiler temporarily enables ``requires_grad`` only on the frozen fused
    QKV base weights for selected SAM blocks. It never adds them to an optimizer
    and restores their original ``requires_grad`` state on close. After each
    backward pass call ``update_after_backward(sample_weight=N)``. If the loss
    is a mean over N selected instances, weighting by N yields an approximate
    per-instance aggregate across uneven batches.

    This is preferable to activation hooks for DAG-RandLoRA because the stored
    matrix is exactly the gradient of the fused linear weight, sliced into Q/V.
    """

    def __init__(
        self,
        model_or_encoder: nn.Module,
        block_indices: Sequence[int] = tuple(range(4, 12)),
        *,
        targets: Sequence[str] = ("q", "v"),
        accumulator_device: str | torch.device = "same",
        accumulator_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.encoder = _resolve_image_encoder(model_or_encoder)
        if not hasattr(self.encoder, "blocks"):
            raise AttributeError("expected image_encoder.blocks")
        self.block_indices = tuple(int(x) for x in block_indices)
        if not self.block_indices or len(set(self.block_indices)) != len(self.block_indices):
            raise ValueError("block_indices must be non-empty and unique")
        if any(i < 0 or i >= len(self.encoder.blocks) for i in self.block_indices):
            raise ValueError("block index out of range")
        self.targets = tuple(str(x) for x in targets)
        if not self.targets or not set(self.targets).issubset({"q", "v"}):
            raise ValueError("targets must contain q and/or v only")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets contains duplicates")
        self.accumulator_device = accumulator_device
        self.accumulator_dtype = accumulator_dtype

        self._linears: Dict[int, nn.Linear] = {}
        self._original_requires_grad: Dict[int, bool] = {}
        self._sum: Dict[str, torch.Tensor] = {}
        self._weight_sum: Dict[str, float] = {}
        self._updates = 0
        self._closed = False

        for block_idx in self.block_indices:
            linear = _resolve_fused_qkv_linear(self.encoder.blocks[block_idx])
            self._linears[block_idx] = linear
            self._original_requires_grad[block_idx] = bool(linear.weight.requires_grad)
            linear.weight.requires_grad_(True)
            linear.weight.grad = None

    def _destination(self, grad: torch.Tensor) -> torch.device:
        if self.accumulator_device == "same":
            return grad.device
        return torch.device(self.accumulator_device)

    @torch.no_grad()
    def zero_probe_grads(self) -> None:
        for linear in self._linears.values():
            linear.weight.grad = None

    @torch.no_grad()
    def update_after_backward(self, *, sample_weight: float = 1.0) -> None:
        if self._closed:
            raise RuntimeError("profiler is closed")
        sample_weight = float(sample_weight)
        if not math.isfinite(sample_weight) or sample_weight <= 0:
            raise ValueError("sample_weight must be finite and positive")
        for block_idx, linear in self._linears.items():
            grad = linear.weight.grad
            if grad is None:
                raise RuntimeError(
                    f"no qkv weight gradient for block {block_idx}; ensure the probe loss "
                    "depends on the SAM image encoder and call loss.backward() before update"
                )
            if grad.ndim != 2 or grad.shape[0] != 3 * grad.shape[1]:
                raise RuntimeError(f"unexpected fused qkv gradient shape {tuple(grad.shape)}")
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f"non-finite qkv gradient in block {block_idx}")
            dim = grad.shape[1]
            slices = {"q": grad[:dim], "v": grad[2 * dim : 3 * dim]}
            for target in self.targets:
                key = canonical_target_key(block_idx, target)
                value = slices[target].detach().to(
                    device=self._destination(grad), dtype=self.accumulator_dtype
                )
                weighted = value * sample_weight
                if key not in self._sum:
                    self._sum[key] = weighted.clone()
                    self._weight_sum[key] = sample_weight
                else:
                    self._sum[key].add_(weighted)
                    self._weight_sum[key] += sample_weight
        self._updates += 1
        self.zero_probe_grads()

    def matrices(self, *, clone: bool = True) -> Dict[str, torch.Tensor]:
        expected = {
            canonical_target_key(block, target)
            for block in self.block_indices
            for target in self.targets
        }
        missing = expected - set(self._sum)
        if missing:
            raise RuntimeError(f"no gradient observations for targets {sorted(missing)}")
        result = {
            key: self._sum[key] / max(self._weight_sum[key], 1e-12) for key in sorted(self._sum)
        }
        return {key: value.clone() for key, value in result.items()} if clone else result

    @property
    def updates(self) -> int:
        return self._updates

    def close(self) -> None:
        if self._closed:
            return
        for block_idx, linear in self._linears.items():
            linear.weight.grad = None
            linear.weight.requires_grad_(self._original_requires_grad[block_idx])
        self._closed = True

    def __enter__(self) -> "FusedQKVWeightGradientProfiler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def save_gradient_profile(
    path: str | Path,
    matrices: Mapping[str, torch.Tensor],
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    if not matrices:
        raise ValueError("matrices cannot be empty")
    payload = {
        "format": "dag-randlora-gradient-profile",
        "version": 1,
        "matrices": {k: v.detach().float().cpu().clone() for k, v in matrices.items()},
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_gradient_profile(path: str | Path) -> tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("format") != "dag-randlora-gradient-profile":
        raise ValueError("not a DAG-RandLoRA gradient profile")
    matrices = payload.get("matrices")
    if not isinstance(matrices, Mapping) or not matrices:
        raise ValueError("profile contains no matrices")
    result: Dict[str, torch.Tensor] = {}
    for key, tensor in matrices.items():
        parse_target_key(str(key))
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"invalid gradient matrix for {key}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite gradient matrix for {key}")
        result[str(key)] = tensor.float()
    return result, dict(payload.get("metadata") or {})


def entropy_effective_rank(matrix: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Entropy effective rank used as Gradient Intrinsic Dimensionality (GID)."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    with torch.no_grad():
        singular = torch.linalg.svdvals(matrix.float())
        total = singular.sum()
        if float(total.item()) <= eps:
            return 0.0
        p = singular / total
        entropy = -(p * torch.log(p.clamp_min(eps))).sum()
        return float(torch.exp(entropy).item())


def _norm(matrix: torch.Tensor) -> float:
    value = float(torch.linalg.matrix_norm(matrix.float(), ord="fro").item())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite gradient norm")
    return value


def _zscore(values: Mapping[str, float], *, eps: float = 1e-12) -> Dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("scores must be finite")
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    if float(std.item()) <= eps:
        return {k: 0.0 for k in keys}
    z = (tensor - mean) / std
    return {k: float(v) for k, v in zip(keys, z.tolist())}


def _softmax_scores(values: Mapping[str, float], *, temperature: float = 1.0) -> Dict[str, float]:
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    keys = sorted(values)
    tensor = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64) / temperature
    probs = torch.softmax(tensor, dim=0)
    return {k: float(v) for k, v in zip(keys, probs.tolist())}


def target_profile_statistics(
    hard_gradients: Mapping[str, torch.Tensor],
    *,
    general_gradients: Optional[Mapping[str, torch.Tensor]] = None,
    classification_gradients: Optional[Mapping[str, torch.Tensor]] = None,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, float]]:
    if not hard_gradients:
        raise ValueError("hard_gradients cannot be empty")
    keys = set(hard_gradients)
    for other, name in (
        (general_gradients, "general_gradients"),
        (classification_gradients, "classification_gradients"),
    ):
        if other is not None and set(other) != keys:
            raise ValueError(f"{name} keys must match hard_gradients")
    stats: Dict[str, Dict[str, float]] = {}
    for key in sorted(keys):
        hard = hard_gradients[key]
        parse_target_key(key)
        hard_norm = _norm(hard)
        gid = entropy_effective_rank(hard, eps=eps)
        item = {
            "hard_norm": hard_norm,
            "gid": gid,
            "gid_fraction": gid / max(min(hard.shape), 1),
        }
        if general_gradients is not None:
            general_norm = _norm(general_gradients[key])
            item["general_norm"] = general_norm
            item["hard_specificity"] = hard_norm / (general_norm + eps)
        if classification_gradients is not None:
            cls_norm = _norm(classification_gradients[key])
            item["classification_norm"] = cls_norm
            item["classification_to_hard"] = cls_norm / (hard_norm + eps)
        stats[key] = item
    return stats


def dag_priority_scores(
    hard_gradients: Mapping[str, torch.Tensor],
    *,
    general_gradients: Mapping[str, torch.Tensor],
    classification_gradients: Optional[Mapping[str, torch.Tensor]] = None,
    gid_weight: float = 1.0,
    specificity_weight: float = 0.5,
    classification_penalty: float = 0.0,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute target-level DAG priority from GID and hard-instance specificity.

    R2 default intentionally sets classification_penalty=0 because the user's
    formal encoder experiment keeps the classification head DETACH. A temporary
    non-detached classification probe can be supplied later as a separate
    context-protection ablation, not silently mixed into the first R2 run.
    """
    for value, name in (
        (gid_weight, "gid_weight"),
        (specificity_weight, "specificity_weight"),
        (classification_penalty, "classification_penalty"),
    ):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if classification_penalty > 0 and classification_gradients is None:
        raise ValueError("classification_gradients are required when classification_penalty > 0")

    stats = target_profile_statistics(
        hard_gradients,
        general_gradients=general_gradients,
        classification_gradients=classification_gradients,
        eps=eps,
    )
    log_gid = {k: math.log(max(v["gid"], eps)) for k, v in stats.items()}
    log_specificity = {
        k: math.log(max(v.get("hard_specificity", 1.0), eps)) for k, v in stats.items()
    }
    z_gid = _zscore(log_gid)
    z_spec = _zscore(log_specificity)
    if classification_gradients is not None:
        log_cls = {
            k: math.log(max(v.get("classification_to_hard", 0.0), eps)) for k, v in stats.items()
        }
        z_cls = _zscore(log_cls)
    else:
        z_cls = {k: 0.0 for k in stats}

    raw = {
        key: gid_weight * z_gid[key]
        + specificity_weight * z_spec[key]
        - classification_penalty * z_cls[key]
        for key in stats
    }
    scores = _softmax_scores(raw, temperature=temperature)
    for key in stats:
        stats[key]["z_log_gid"] = z_gid[key]
        stats[key]["z_log_specificity"] = z_spec[key]
        stats[key]["z_log_classification_to_hard"] = z_cls[key]
        stats[key]["raw_priority"] = raw[key]
        stats[key]["priority"] = scores[key]
    return scores, stats


def _target_shapes(gradients: Mapping[str, torch.Tensor]) -> Dict[str, tuple[int, int]]:
    return {key: (int(value.shape[1]), int(value.shape[0])) for key, value in gradients.items()}


def _pattern_params(
    target_shapes: Mapping[str, tuple[int, int]], pattern: Mapping[str, int]
) -> int:
    return sum(randlora_trainable_params(*target_shapes[key], int(pattern[key])) for key in pattern)


@dataclass(frozen=True)
class DAGAllocationPlan:
    stage: str
    baseline_rank: int
    target_total_params: int
    actual_total_params: int
    budget_relative_error: float
    rank_candidates: tuple[int, ...]
    block_rank_pattern: Dict[int, int]
    target_rank_pattern: Dict[str, int]
    scores: Dict[str, float]
    statistics: Dict[str, Dict[str, float]]
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["rank_candidates"] = list(self.rank_candidates)
        data["block_rank_pattern"] = {str(k): int(v) for k, v in self.block_rank_pattern.items()}
        return data

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DAGAllocationPlan":
        payload = dict(data)
        payload["rank_candidates"] = tuple(int(x) for x in payload.get("rank_candidates", ()))
        payload["block_rank_pattern"] = {
            int(k): int(v) for k, v in dict(payload.get("block_rank_pattern", {})).items()
        }
        payload["target_rank_pattern"] = {
            str(k): int(v) for k, v in dict(payload.get("target_rank_pattern", {})).items()
        }
        payload["scores"] = {str(k): float(v) for k, v in dict(payload.get("scores", {})).items()}
        payload["statistics"] = {
            str(k): {str(sk): float(sv) for sk, sv in dict(v).items()}
            for k, v in dict(payload.get("statistics", {})).items()
        }
        payload["metadata"] = dict(payload.get("metadata", {}))
        return cls(**payload)

    @classmethod
    def load_json(cls, path: str | Path) -> "DAGAllocationPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("allocation JSON must contain an object")
        return cls.from_dict(data)


def build_r1_oc_plan(
    hard_gradients: Mapping[str, torch.Tensor],
    *,
    general_gradients: Mapping[str, torch.Tensor],
    baseline_rank: int,
    target_total_params: int,
    rank_candidates: Sequence[int] = (64, 70, 80),
    budget_tolerance: float = 0.01,
    general_penalty: float = 0.20,
    metadata: Optional[Mapping[str, object]] = None,
) -> DAGAllocationPlan:
    """R1: occupancy-conditioned block-level capacity allocation.

    Q and V share one rank within a block. Priority is the normalized hard
    gradient norm minus a small normalized general-gradient penalty.
    """
    if set(hard_gradients) != set(general_gradients):
        raise ValueError("hard/general gradient keys must match")
    block_hard: Dict[int, float] = {}
    block_general: Dict[int, float] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for key in sorted(hard_gradients):
        block, _target = parse_target_key(key)
        h = _norm(hard_gradients[key])
        g = _norm(general_gradients[key])
        block_hard[block] = block_hard.get(block, 0.0) + h
        block_general[block] = block_general.get(block, 0.0) + g
        stats[key] = {"hard_norm": h, "general_norm": g}

    def normalize_int(values: Mapping[int, float]) -> Dict[int, float]:
        total = sum(max(float(v), 0.0) for v in values.values())
        if total <= 0:
            return {k: 1.0 / len(values) for k in values}
        return {k: max(float(v), 0.0) / total for k, v in values.items()}

    hard_n = normalize_int(block_hard)
    general_n = normalize_int(block_general)
    raw = {block: max(hard_n[block] - general_penalty * general_n[block], 0.0) for block in hard_n}
    total_raw = sum(raw.values())
    block_scores = (
        {k: v / total_raw for k, v in raw.items()}
        if total_raw > 0
        else {k: 1.0 / len(raw) for k in raw}
    )
    dim = next(iter(hard_gradients.values())).shape[0]
    block_pattern = allocate_block_ranks(
        block_scores,
        dim=dim,
        adapters_per_block=2,
        baseline_rank=baseline_rank,
        target_total_params=target_total_params,
        budget_tolerance=budget_tolerance,
        rank_candidates=rank_candidates,
        max_search_combinations=1_000_000,
    )
    shapes = _target_shapes(hard_gradients)
    target_pattern = {
        key: int(block_pattern[parse_target_key(key)[0]]) for key in sorted(hard_gradients)
    }
    actual = _pattern_params(shapes, target_pattern)
    score_strings = {str(k): float(v) for k, v in block_scores.items()}
    return DAGAllocationPlan(
        stage="R1-OC",
        baseline_rank=int(baseline_rank),
        target_total_params=int(target_total_params),
        actual_total_params=int(actual),
        budget_relative_error=abs(actual - target_total_params) / target_total_params,
        rank_candidates=tuple(int(x) for x in rank_candidates),
        block_rank_pattern={int(k): int(v) for k, v in block_pattern.items()},
        target_rank_pattern={},
        scores=score_strings,
        statistics=stats,
        metadata=dict(metadata or {}),
    )


def build_r2_dag_plan(
    hard_gradients: Mapping[str, torch.Tensor],
    *,
    general_gradients: Mapping[str, torch.Tensor],
    baseline_rank: int,
    target_total_params: int,
    rank_candidates: Sequence[int] = DEFAULT_RANK_CANDIDATES,
    budget_tolerance: float = 0.01,
    gid_weight: float = 1.0,
    specificity_weight: float = 0.5,
    temperature: float = 1.0,
    classification_gradients: Optional[Mapping[str, torch.Tensor]] = None,
    classification_penalty: float = 0.0,
    metadata: Optional[Mapping[str, object]] = None,
) -> DAGAllocationPlan:
    """R2: Q/V-specific GID + hard-specificity fixed-budget allocation."""
    scores, stats = dag_priority_scores(
        hard_gradients,
        general_gradients=general_gradients,
        classification_gradients=classification_gradients,
        gid_weight=gid_weight,
        specificity_weight=specificity_weight,
        classification_penalty=classification_penalty,
        temperature=temperature,
    )
    shapes = _target_shapes(hard_gradients)
    pattern = allocate_target_ranks(
        scores,
        target_shapes=shapes,
        baseline_rank=baseline_rank,
        target_total_params=target_total_params,
        budget_tolerance=budget_tolerance,
        rank_candidates=rank_candidates,
    )
    actual = _pattern_params(shapes, pattern)
    return DAGAllocationPlan(
        stage="R2-DAG",
        baseline_rank=int(baseline_rank),
        target_total_params=int(target_total_params),
        actual_total_params=int(actual),
        budget_relative_error=abs(actual - target_total_params) / target_total_params,
        rank_candidates=tuple(int(x) for x in rank_candidates),
        block_rank_pattern={},
        target_rank_pattern={str(k): int(v) for k, v in pattern.items()},
        scores={str(k): float(v) for k, v in scores.items()},
        statistics=stats,
        metadata=dict(metadata or {}),
    )


def config_from_allocation_plan(
    base_config: RandLoRADamageConfig,
    plan: DAGAllocationPlan,
) -> RandLoRADamageConfig:
    cfg = copy.deepcopy(base_config)
    cfg.target_trainable_params = int(plan.target_total_params)
    cfg.block_rank_pattern = dict(plan.block_rank_pattern)
    cfg.target_rank_pattern = dict(plan.target_rank_pattern)
    return cfg
