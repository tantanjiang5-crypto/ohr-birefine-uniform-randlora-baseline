from __future__ import annotations

import itertools
import math
from typing import Dict, Mapping, Sequence

import torch
from torch import nn

from .config import randlora_trainable_params


class EncoderGradientSensitivityCollector:
    """Collect block activation×gradient sensitivity in an offline pass.

    Run separate collectors for low-occupancy mask loss, the general mask
    distribution, and classification loss. ``score_mode='normalized'`` reduces
    scale bias across residual blocks.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        block_indices: Sequence[int],
        *,
        score_mode: str = "normalized",
        eps: float = 1e-12,
    ) -> None:
        if score_mode not in {"raw", "normalized"}:
            raise ValueError("score_mode must be 'raw' or 'normalized'")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.image_encoder = image_encoder
        self.block_indices = tuple(int(index) for index in block_indices)
        if not self.block_indices:
            raise ValueError("block_indices cannot be empty")
        if len(set(self.block_indices)) != len(self.block_indices):
            raise ValueError("block_indices contains duplicates")
        if not hasattr(image_encoder, "blocks"):
            raise TypeError("image_encoder must expose a blocks sequence")
        if any(index < 0 or index >= len(image_encoder.blocks) for index in self.block_indices):
            raise ValueError("block index out of range")
        self.score_mode = score_mode
        self.eps = float(eps)
        self._activations: Dict[int, torch.Tensor] = {}
        self._sum: Dict[int, float] = {i: 0.0 for i in self.block_indices}
        self._count: Dict[int, int] = {i: 0 for i in self.block_indices}
        self._handles = [
            image_encoder.blocks[idx].register_forward_hook(self._hook(idx))
            for idx in self.block_indices
        ]

    def _hook(self, idx: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("SAM block output must be a tensor")
            if not output.requires_grad:
                raise RuntimeError(
                    "block output has no gradient; do not wrap the sensitivity forward in "
                    "torch.no_grad()/inference_mode, and for a fully frozen baseline call "
                    "images.requires_grad_(True)"
                )
            output.retain_grad()
            self._activations[idx] = output

        return hook

    @torch.no_grad()
    def update_after_backward(self) -> None:
        missing = set(self.block_indices) - set(self._activations)
        if missing:
            raise RuntimeError(f"no activation captured for blocks {sorted(missing)}")
        for idx in self.block_indices:
            activation = self._activations[idx]
            if activation.grad is None:
                raise RuntimeError(f"no gradient captured for block {idx}")
            a = activation.detach().float()
            g = activation.grad.detach().float()
            raw = (a * g).abs().mean()
            if self.score_mode == "normalized":
                raw = raw / (a.abs().mean() * g.abs().mean() + self.eps)
            value = float(raw.item())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite sensitivity for block {idx}")
            self._sum[idx] += value
            self._count[idx] += 1
        self._activations.clear()

    def scores(self, normalize: bool = True, *, require_all: bool = True) -> Dict[int, float]:
        if require_all:
            missing = [idx for idx, count in self._count.items() if count == 0]
            if missing:
                raise RuntimeError(f"no sensitivity observations for blocks {missing}")
        result = {
            idx: self._sum[idx] / max(self._count[idx], 1) for idx in self.block_indices
        }
        return _normalize_scores(result) if normalize else result

    def reset(self) -> None:
        self._activations.clear()
        self._sum = {i: 0.0 for i in self.block_indices}
        self._count = {i: 0 for i in self.block_indices}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _normalize_scores(scores: Mapping[int, float]) -> Dict[int, float]:
    if not scores:
        raise ValueError("scores cannot be empty")
    values: Dict[int, float] = {}
    for key, value in scores.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"score for block {key} must be finite")
        values[int(key)] = max(numeric, 0.0)
    if len(values) != len(scores):
        raise ValueError("score keys collapse to duplicate integer block indices")
    total = sum(values.values())
    if total <= 0:
        return {key: 1.0 / len(values) for key in values}
    return {key: value / total for key, value in values.items()}


def damage_priority_scores(
    low_occupancy_mask_scores: Mapping[int, float],
    *,
    general_mask_scores: Mapping[int, float] | None = None,
    classification_scores: Mapping[int, float] | None = None,
    general_penalty: float = 0.20,
    classification_penalty: float = 0.25,
    floor: float = 0.0,
) -> Dict[int, float]:
    """Favor blocks specific to low-occupancy mask errors, not generic context."""
    for value, name in (
        (general_penalty, "general_penalty"),
        (classification_penalty, "classification_penalty"),
        (floor, "floor"),
    ):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    low = _normalize_scores(low_occupancy_mask_scores)
    blocks = set(low)
    general = _normalize_scores(general_mask_scores) if general_mask_scores else None
    cls = _normalize_scores(classification_scores) if classification_scores else None
    for other, name in ((general, "general_mask_scores"), (cls, "classification_scores")):
        if other is not None and set(other) != blocks:
            raise ValueError(f"{name} block keys must match low_occupancy_mask_scores")
    raw = {}
    for block in sorted(blocks):
        value = low[block]
        if general is not None:
            value -= general_penalty * general[block]
        if cls is not None:
            value -= classification_penalty * cls[block]
        raw[block] = max(value, floor)
    return _normalize_scores(raw)


def allocate_block_ranks(
    scores: Mapping[int, float],
    *,
    dim: int,
    adapters_per_block: int = 2,
    baseline_rank: int,
    target_total_params: int | None = None,
    budget_tolerance: float = 0.02,
    rank_candidates: Sequence[int] | None = None,
    max_search_combinations: int = 1_000_000,
) -> Dict[int, int]:
    """Allocate more coefficients (lower RandLoRA rank) to priority blocks.

    Within the declared budget tolerance, the priority reward is optimized
    first; budget error is used as a tie-breaker. If no candidate is feasible,
    the closest-budget solution is returned.
    """
    if not scores:
        raise ValueError("scores cannot be empty")
    if dim <= 0 or adapters_per_block <= 0 or baseline_rank <= 0:
        raise ValueError("dim, adapters_per_block and baseline_rank must be positive")
    if baseline_rank > dim:
        raise ValueError("baseline_rank cannot exceed dim")
    if not math.isfinite(float(budget_tolerance)) or budget_tolerance < 0:
        raise ValueError("budget_tolerance must be finite and non-negative")
    if max_search_combinations <= 0:
        raise ValueError("max_search_combinations must be positive")
    normalized = _normalize_scores(scores)
    blocks = tuple(sorted(normalized))
    if rank_candidates is None:
        ranks = sorted(
            {
                max(1, baseline_rank // 2),
                baseline_rank,
                min(dim, baseline_rank * 2),
            }
        )
    else:
        ranks = sorted({int(rank) for rank in rank_candidates})
        if not ranks or any(rank <= 0 or rank > dim for rank in ranks):
            raise ValueError("rank_candidates must be in [1, dim]")
    if target_total_params is None:
        target_total_params = len(blocks) * adapters_per_block * randlora_trainable_params(
            dim, dim, baseline_rank
        )
    if target_total_params <= 0:
        raise ValueError("target_total_params must be positive")

    combinations = len(ranks) ** len(blocks)
    if combinations > max_search_combinations:
        raise ValueError(
            f"rank allocation search would evaluate {combinations:,} combinations; "
            "reduce rank_candidates/blocks or raise max_search_combinations explicitly"
        )

    best_fallback = None
    best_feasible = None
    for choice in itertools.product(ranks, repeat=len(blocks)):
        params_per_block = [
            adapters_per_block * randlora_trainable_params(dim, dim, rank)
            for rank in choice
        ]
        total = sum(params_per_block)
        absolute_error = abs(total - target_total_params)
        rel_error = absolute_error / target_total_params
        reward = sum(
            normalized[block] * params
            for block, params in zip(blocks, params_per_block)
        )
        fallback = (rel_error, -reward, absolute_error, total, choice)
        if best_fallback is None or fallback < best_fallback:
            best_fallback = fallback
        feasible = (-reward, rel_error, absolute_error, total, choice)
        if rel_error <= budget_tolerance and (
            best_feasible is None or feasible < best_feasible
        ):
            best_feasible = feasible

    assert best_fallback is not None
    selected_choice = best_feasible[4] if best_feasible is not None else best_fallback[4]
    return {block: int(rank) for block, rank in zip(blocks, selected_choice)}


def allocate_target_ranks(
    scores: Mapping[str, float],
    *,
    target_shapes: Mapping[str, tuple[int, int]],
    baseline_rank: int,
    target_total_params: int | None = None,
    budget_tolerance: float = 0.01,
    rank_candidates: Sequence[int] = (48, 56, 64, 70, 80, 96, 112),
    max_states: int = 250_000,
) -> Dict[str, int]:
    """Allocate RandLoRA basis ranks per target under a fixed parameter budget.

    The objective rewards assigning *more trainable RandLoRA coefficients* to
    high-priority targets. Because RandLoRA trainable capacity generally grows
    as basis rank decreases, the returned pattern tends to use smaller ranks on
    high-score targets and larger ranks on low-score targets. Every candidate
    remains full-rank reachable because ``ceil(min_dim/r) * r >= min_dim``.

    Dynamic programming is used instead of a Cartesian product so 16 SAM Q/V
    targets can be optimized without an exponential search.
    """
    if not scores:
        raise ValueError("scores cannot be empty")
    if set(scores) != set(target_shapes):
        raise ValueError("scores and target_shapes must contain identical target keys")
    if baseline_rank <= 0:
        raise ValueError("baseline_rank must be positive")
    if not math.isfinite(float(budget_tolerance)) or budget_tolerance < 0:
        raise ValueError("budget_tolerance must be finite and non-negative")
    if max_states <= 0:
        raise ValueError("max_states must be positive")

    normalized = _normalize_generic_scores(scores)
    targets = tuple(sorted(normalized))
    candidates = tuple(sorted({int(rank) for rank in rank_candidates}))
    if not candidates or any(rank <= 0 for rank in candidates):
        raise ValueError("rank_candidates must contain positive integers")

    costs: Dict[str, Dict[int, int]] = {}
    for key in targets:
        shape = target_shapes[key]
        if len(shape) != 2:
            raise ValueError(f"target shape for {key} must be (in_features, out_features)")
        in_features, out_features = (int(shape[0]), int(shape[1]))
        min_dim = min(in_features, out_features)
        valid = [rank for rank in candidates if rank <= min_dim]
        if baseline_rank > min_dim:
            raise ValueError(f"baseline_rank exceeds min dimension for {key}")
        if not valid:
            raise ValueError(f"no valid rank candidates for {key}")
        costs[key] = {
            rank: randlora_trainable_params(in_features, out_features, rank) for rank in valid
        }

    if target_total_params is None:
        target_total_params = sum(
            randlora_trainable_params(*target_shapes[key], baseline_rank) for key in targets
        )
    if target_total_params <= 0:
        raise ValueError("target_total_params must be positive")

    # total_params -> (reward, tuple[ranks...]). For equal totals only the
    # highest reward can ever be optimal, so exact deduplication is safe.
    states: Dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    processed: list[str] = []
    for key in targets:
        processed.append(key)
        next_states: Dict[int, tuple[float, tuple[int, ...]]] = {}
        for total, (reward, choice) in states.items():
            for rank, cost in costs[key].items():
                new_total = total + cost
                new_reward = reward + normalized[key] * cost
                previous = next_states.get(new_total)
                candidate = (new_reward, choice + (rank,))
                if previous is None or candidate[0] > previous[0]:
                    next_states[new_total] = candidate

        if len(next_states) > max_states:
            # Retain states most likely to reach the target budget while keeping
            # high reward. This is only a safety valve; SAM Q/V searches with the
            # recommended candidates normally remain well below the cap.
            remaining = targets[len(processed) :]
            rem_min = sum(min(costs[t].values()) for t in remaining)
            rem_max = sum(max(costs[t].values()) for t in remaining)

            def state_priority(item):
                total, (reward, _choice) = item
                reachable_low = total + rem_min
                reachable_high = total + rem_max
                if reachable_low <= target_total_params <= reachable_high:
                    distance = 0
                else:
                    distance = min(
                        abs(reachable_low - target_total_params),
                        abs(reachable_high - target_total_params),
                    )
                return (distance, -reward, abs(total - target_total_params))

            kept = sorted(next_states.items(), key=state_priority)[:max_states]
            next_states = dict(kept)
        states = next_states

    feasible = []
    fallback = []
    for total, (reward, choice) in states.items():
        rel_error = abs(total - target_total_params) / target_total_params
        fallback.append((rel_error, -reward, abs(total - target_total_params), total, choice))
        if rel_error <= budget_tolerance:
            feasible.append((-reward, rel_error, abs(total - target_total_params), total, choice))
    if not fallback:
        raise RuntimeError("target rank allocation produced no states")
    selected = min(feasible)[4] if feasible else min(fallback)[4]
    return {key: int(rank) for key, rank in zip(targets, selected)}


def _normalize_generic_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in scores.items():
        if not isinstance(key, str) or not key:
            raise ValueError("target score keys must be non-empty strings")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"score for {key} must be finite")
        values[key] = max(numeric, 0.0)
    total = sum(values.values())
    if total <= 0:
        return {key: 1.0 / len(values) for key in values}
    return {key: value / total for key, value in values.items()}
