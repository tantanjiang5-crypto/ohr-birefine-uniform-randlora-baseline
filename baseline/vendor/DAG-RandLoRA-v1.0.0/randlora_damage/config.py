from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


_ALLOWED_QKV_TARGETS = {"q", "v"}
_ALLOWED_EXTRA_TARGETS = {"attn.proj", "mlp.lin1", "mlp.lin2"}
_ALLOWED_FORWARD_MODES = {"auto", "materialized", "factorized"}
_ALLOWED_ADAPTER_DTYPES = {"base", "float32"}


def _require_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def canonical_target_key(block_idx: int, target: str) -> str:
    if not isinstance(block_idx, int) or isinstance(block_idx, bool) or block_idx < 0:
        raise ValueError("block_idx must be a non-negative integer")
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")
    return f"{block_idx}:{target}"


def parse_target_key(key: str) -> tuple[int, str]:
    if not isinstance(key, str) or ":" not in key:
        raise ValueError(f"invalid target key {key!r}; expected '<block>:<target>'")
    block_text, target = key.split(":", 1)
    try:
        block_idx = int(block_text)
    except ValueError as exc:
        raise ValueError(f"invalid target key {key!r}") from exc
    if block_idx < 0 or not target:
        raise ValueError(f"invalid target key {key!r}")
    return block_idx, target


def randlora_trainable_params(in_features: int, out_features: int, rank: int) -> int:
    """Trainable lambda+gamma parameters for one RandLoRA-adapted matrix."""
    in_features = _require_positive_int(in_features, "in_features")
    out_features = _require_positive_int(out_features, "out_features")
    rank = _require_positive_int(rank, "rank")
    min_dim = min(in_features, out_features)
    num_bases = math.ceil(min_dim / rank)
    return num_bases * (rank + min_dim)


def lora_trainable_params(in_features: int, out_features: int, rank: int) -> int:
    in_features = _require_positive_int(in_features, "in_features")
    out_features = _require_positive_int(out_features, "out_features")
    rank = _require_positive_int(rank, "rank")
    return rank * (in_features + out_features)


def total_randlora_params(matrix_shapes: Sequence[Tuple[int, int]], rank: int) -> int:
    if not matrix_shapes:
        raise ValueError("matrix_shapes cannot be empty")
    return sum(randlora_trainable_params(i, o, rank) for i, o in matrix_shapes)


def find_rank_for_budget(
    matrix_shapes: Sequence[Tuple[int, int]],
    target_params: int,
    *,
    max_rank: Optional[int] = None,
) -> Tuple[int, int]:
    """Return ``(rank, resulting_params)`` closest to a trainable budget."""
    target_params = _require_positive_int(target_params, "target_params")
    if not matrix_shapes:
        raise ValueError("matrix_shapes cannot be empty")
    smallest = min(min(i, o) for i, o in matrix_shapes)
    if max_rank is not None:
        max_rank = _require_positive_int(max_rank, "max_rank")
    upper = min(max_rank or smallest, smallest)
    best: Optional[Tuple[int, int, int, int]] = None
    best_rank = 0
    for rank in range(1, upper + 1):
        total = total_randlora_params(matrix_shapes, rank)
        candidate = (abs(total - target_params), int(total > target_params), total, -rank)
        if best is None or candidate < best:
            best = candidate
            best_rank = rank
    assert best is not None
    return best_rank, best[2]


@dataclass
class RandLoRADamageConfig:
    """Configuration for official SAM1 ViT image-encoder RandLoRA injection.

    ``target_rank_pattern`` is the DAG-RandLoRA extension. Keys use
    ``'<block>:<target>'`` (for example ``'8:q'`` or ``'8:v'``). A target-level
    rank overrides ``block_rank_pattern``; a block-level rank overrides the
    resolved uniform rank. This preserves v1.2 behavior when the new mapping is
    empty.
    """

    block_indices: Tuple[int, ...] = tuple(range(4, 12))
    qkv_targets: Tuple[str, ...] = ("q", "v")
    extra_targets: Tuple[str, ...] = ()

    rank: Optional[int] = None
    target_trainable_params: Optional[int] = 147_456
    auto_match_budget: bool = True
    budget_warning_tolerance: float = 0.02

    block_rank_pattern: Dict[int, int] = field(default_factory=dict)
    target_rank_pattern: Dict[str, int] = field(default_factory=dict)

    alpha: Optional[float] = None
    alpha_multiplier: float = 2.0
    dropout: float = 0.0

    forward_mode: str = "auto"
    cache_eval_delta: bool = True
    adapter_dtype: str = "float32"

    seed: int = 42
    save_basis: bool = True
    sparse_basis: bool = False
    very_sparse_basis: bool = False
    freeze_image_encoder: bool = True

    def validate(self, *, num_blocks: Optional[int] = None) -> None:
        if num_blocks is not None:
            _require_positive_int(num_blocks, "num_blocks")
        for name in (
            "auto_match_budget",
            "cache_eval_delta",
            "save_basis",
            "sparse_basis",
            "very_sparse_basis",
            "freeze_image_encoder",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not self.block_indices:
            raise ValueError("block_indices cannot be empty")
        if any(not isinstance(i, int) or isinstance(i, bool) for i in self.block_indices):
            raise TypeError("block_indices must contain integers")
        if len(set(self.block_indices)) != len(self.block_indices):
            raise ValueError("block_indices contains duplicates")
        if any(i < 0 for i in self.block_indices):
            raise ValueError("block indices must be non-negative")
        if num_blocks is not None and any(i >= num_blocks for i in self.block_indices):
            raise ValueError(
                f"block index out of range: encoder has {num_blocks} blocks, requested {self.block_indices}"
            )
        if any(not isinstance(target, str) for target in self.qkv_targets):
            raise TypeError("qkv_targets must contain strings")
        if any(not isinstance(target, str) for target in self.extra_targets):
            raise TypeError("extra_targets must contain strings")
        if len(set(self.qkv_targets)) != len(self.qkv_targets):
            raise ValueError("qkv_targets contains duplicates")
        if len(set(self.extra_targets)) != len(self.extra_targets):
            raise ValueError("extra_targets contains duplicates")
        unknown_qkv = set(self.qkv_targets) - _ALLOWED_QKV_TARGETS
        if unknown_qkv:
            raise ValueError(f"unsupported qkv targets: {sorted(unknown_qkv)}")
        if not self.qkv_targets and not self.extra_targets:
            raise ValueError("at least one target is required")
        unknown_extra = set(self.extra_targets) - _ALLOWED_EXTRA_TARGETS
        if unknown_extra:
            raise ValueError(f"unsupported extra targets: {sorted(unknown_extra)}")
        if self.forward_mode not in _ALLOWED_FORWARD_MODES:
            raise ValueError(f"forward_mode must be one of {_ALLOWED_FORWARD_MODES}")
        if self.adapter_dtype not in _ALLOWED_ADAPTER_DTYPES:
            raise ValueError(f"adapter_dtype must be one of {_ALLOWED_ADAPTER_DTYPES}")
        if not isinstance(self.dropout, (int, float)) or not math.isfinite(float(self.dropout)):
            raise TypeError("dropout must be a finite number")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rank is not None:
            _require_positive_int(self.rank, "rank")
        if self.target_trainable_params is not None:
            _require_positive_int(self.target_trainable_params, "target_trainable_params")
        if not self.auto_match_budget and self.rank is None:
            raise ValueError("rank is required when auto_match_budget=False")
        if self.auto_match_budget and self.target_trainable_params is None and self.rank is None:
            raise ValueError("rank or target_trainable_params must be provided")
        if self.auto_match_budget and self.target_trainable_params is not None and self.rank is not None:
            raise ValueError(
                "rank and target_trainable_params are ambiguous when auto_match_budget=True; "
                "set rank=None for budget matching or auto_match_budget=False for a fixed rank"
            )
        if not isinstance(self.budget_warning_tolerance, (int, float)) or not math.isfinite(float(self.budget_warning_tolerance)):
            raise TypeError("budget_warning_tolerance must be a finite number")
        if self.budget_warning_tolerance < 0:
            raise ValueError("budget_warning_tolerance must be non-negative")
        if self.alpha is not None:
            if not isinstance(self.alpha, (int, float)) or not math.isfinite(float(self.alpha)):
                raise TypeError("alpha must be a finite number")
            if self.alpha <= 0:
                raise ValueError("alpha must be positive")
        if not isinstance(self.alpha_multiplier, (int, float)) or not math.isfinite(float(self.alpha_multiplier)):
            raise TypeError("alpha_multiplier must be a finite number")
        if self.alpha_multiplier <= 0:
            raise ValueError("alpha_multiplier must be positive")

        selected_targets = set(self.qkv_targets) | set(self.extra_targets)
        for block_idx, rank in self.block_rank_pattern.items():
            if not isinstance(block_idx, int) or isinstance(block_idx, bool):
                raise TypeError("block_rank_pattern keys must be integer block indices")
            if block_idx not in self.block_indices:
                raise ValueError(f"block_rank_pattern contains unselected block {block_idx}")
            _require_positive_int(rank, f"rank for block {block_idx}")
        for key, rank in self.target_rank_pattern.items():
            block_idx, target = parse_target_key(key)
            if block_idx not in self.block_indices:
                raise ValueError(f"target_rank_pattern contains unselected block {block_idx}")
            if target not in selected_targets:
                raise ValueError(f"target_rank_pattern contains unselected target {target!r}")
            _require_positive_int(rank, f"rank for target {key}")

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if self.sparse_basis and self.very_sparse_basis:
            raise ValueError("sparse_basis and very_sparse_basis are mutually exclusive")

    def alpha_for_rank(self, rank: int) -> float:
        return float(self.alpha if self.alpha is not None else self.alpha_multiplier * rank)

    def rank_for_block(self, block_idx: int, default_rank: int) -> int:
        return int(self.block_rank_pattern.get(block_idx, default_rank))

    def rank_for_target(self, block_idx: int, target: str, default_rank: int) -> int:
        key = canonical_target_key(block_idx, target)
        if key in self.target_rank_pattern:
            return int(self.target_rank_pattern[key])
        return self.rank_for_block(block_idx, default_rank)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["block_indices"] = list(self.block_indices)
        data["qkv_targets"] = list(self.qkv_targets)
        data["extra_targets"] = list(self.extra_targets)
        data["block_rank_pattern"] = {str(k): int(v) for k, v in self.block_rank_pattern.items()}
        data["target_rank_pattern"] = {str(k): int(v) for k, v in self.target_rank_pattern.items()}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RandLoRADamageConfig":
        payload = dict(data)
        for key in ("block_indices", "qkv_targets", "extra_targets"):
            if key in payload:
                payload[key] = tuple(payload[key])
        if "block_rank_pattern" in payload:
            payload["block_rank_pattern"] = {int(k): int(v) for k, v in payload["block_rank_pattern"].items()}
        payload.setdefault("target_rank_pattern", {})
        if "target_rank_pattern" in payload:
            payload["target_rank_pattern"] = {str(k): int(v) for k, v in payload["target_rank_pattern"].items()}
        payload.setdefault("budget_warning_tolerance", 0.02)
        payload.setdefault("adapter_dtype", "float32")
        if payload.get("forward_mode") not in _ALLOWED_FORWARD_MODES:
            payload["forward_mode"] = "auto"
        return cls(**payload)
