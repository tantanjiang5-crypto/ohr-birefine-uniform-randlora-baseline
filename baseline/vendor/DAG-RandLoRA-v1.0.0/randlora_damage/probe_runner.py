from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .dag import FusedQKVWeightGradientProfiler


ProbeLossFn = Callable[[nn.Module, object], Tuple[torch.Tensor | None, int]]


@dataclass(frozen=True)
class ProbeRunReport:
    batches_seen: int
    batches_used: int
    selected_instances: int
    profiler_updates: int
    mode_before: bool


def collect_qv_gradient_profile(
    model: nn.Module,
    dataloader: Iterable[object],
    loss_fn: ProbeLossFn,
    *,
    block_indices: Sequence[int] = tuple(range(4, 12)),
    targets: Sequence[str] = ("q", "v"),
    max_instances: int = 1024,
    max_batches: int = 512,
    accumulator_device: str | torch.device = "same",
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    progress_every: int = 5,
) -> tuple[Dict[str, torch.Tensor], ProbeRunReport]:
    """Run a deterministic no-update gradient calibration pass.

    ``loss_fn(model, batch)`` must return ``(scalar_loss, selected_count)``.
    The scalar loss must be the *mean over selected instances*. Return
    ``(None, 0)`` when the batch contains no requested instances. Do not call a
    GradScaler inside ``loss_fn``; this pass profiles the real loss gradient.

    All model parameter ``requires_grad`` flags and train/eval mode are restored
    afterward. Only selected fused-QKV base weights temporarily require grads.
    """
    if max_instances <= 0 or max_batches <= 0:
        raise ValueError("max_instances and max_batches must be positive")
    original_mode = bool(model.training)
    original_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    batches_seen = 0
    batches_used = 0
    selected_instances = 0

    try:
        model.eval()
        model.requires_grad_(False)
        with FusedQKVWeightGradientProfiler(
            model,
            block_indices=block_indices,
            targets=targets,
            accumulator_device=accumulator_device,
        ) as profiler:
            for batch in dataloader:
                batches_seen += 1
                if batches_seen > max_batches:
                    break
                model.zero_grad(set_to_none=True)
                profiler.zero_probe_grads()
                loss, count = loss_fn(model, batch)
                count = int(count)
                if count < 0:
                    raise ValueError("loss_fn selected_count cannot be negative")
                if count == 0:
                    if loss is not None:
                        raise ValueError("loss_fn must return loss=None when selected_count=0")
                    continue
                if loss is None or not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                    raise TypeError("loss_fn must return a scalar Tensor for non-empty selections")
                if not torch.isfinite(loss):
                    raise FloatingPointError("probe loss is non-finite")
                loss.backward()
                profiler.update_after_backward(sample_weight=float(count))
                batches_used += 1
                selected_instances += count
                if progress_callback is not None and (
                    batches_used == 1 or batches_used % max(int(progress_every), 1) == 0
                ):
                    progress_callback(batches_seen, batches_used, selected_instances)
                if selected_instances >= max_instances:
                    break
            if batches_used == 0:
                raise RuntimeError("gradient probe collected no usable batches")
            matrices = profiler.matrices()
            updates = profiler.updates
    finally:
        for name, parameter in model.named_parameters():
            if name in original_flags:
                parameter.requires_grad_(original_flags[name])
            parameter.grad = None
        model.train(original_mode)

    return matrices, ProbeRunReport(
        batches_seen=batches_seen,
        batches_used=batches_used,
        selected_instances=selected_instances,
        profiler_updates=updates,
        mode_before=original_mode,
    )
