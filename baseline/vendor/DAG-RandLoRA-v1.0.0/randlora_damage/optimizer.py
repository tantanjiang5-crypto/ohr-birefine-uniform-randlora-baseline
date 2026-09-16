from __future__ import annotations

import math
from typing import List, MutableSequence, Optional

import torch
from torch import nn


def is_randlora_parameter(name: str) -> bool:
    return "randlora_lambda" in name or "randlora_gamma" in name


def named_randlora_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad and is_randlora_parameter(name):
            yield name, param


def randlora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [param for _, param in named_randlora_parameters(model)]


def _validate_group_hparams(lr: float, weight_decay: float) -> None:
    if not math.isfinite(float(lr)) or lr <= 0:
        raise ValueError("encoder_lr must be finite and positive")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("encoder_weight_decay must be finite and non-negative")


def _materialize_existing_params(param_groups: MutableSequence[dict]) -> set[int]:
    """Turn one-shot iterables into lists before duplicate inspection.

    Optimizer groups occasionally contain generators. Iterating over those only
    for a duplicate check would otherwise consume them and silently remove the
    parameters from the optimizer construction that follows.
    """
    existing_ids: set[int] = set()
    for index, group in enumerate(param_groups):
        if not isinstance(group, dict):
            raise TypeError(f"optimizer group {index} must be a dict")
        params = group.get("params", ())
        if isinstance(params, torch.Tensor):
            raise TypeError("optimizer group 'params' must be an iterable of Parameters")
        if not isinstance(params, (list, tuple)):
            params = list(params)
            group["params"] = params
        for param in params:
            if not isinstance(param, nn.Parameter):
                raise TypeError("optimizer groups may contain only nn.Parameter objects")
            existing_ids.add(id(param))
    return existing_ids


def _new_group(
    model: nn.Module,
    *,
    encoder_lr: float,
    encoder_weight_decay: float,
    group_name: str,
    existing_ids: set[int],
) -> dict:
    _validate_group_hparams(encoder_lr, encoder_weight_decay)
    adapter_params = randlora_parameters(model)
    if not adapter_params:
        raise RuntimeError("no trainable RandLoRA parameters found")
    duplicates = [param for param in adapter_params if id(param) in existing_ids]
    if duplicates:
        raise ValueError(f"{len(duplicates)} RandLoRA parameters already exist in optimizer groups")
    return {
        "params": adapter_params,
        "lr": float(encoder_lr),
        "weight_decay": float(encoder_weight_decay),
        "group_name": str(group_name),
    }


def append_randlora_param_group(
    param_groups: MutableSequence[dict],
    model: nn.Module,
    *,
    encoder_lr: float = 1e-4,
    encoder_weight_decay: float = 0.0,
    group_name: str = "randlora_encoder",
) -> MutableSequence[dict]:
    """Append RandLoRA parameters without collapsing existing head groups.

    This is intended *before* optimizer construction. Existing one-shot
    parameter iterables are materialized so the safety check cannot consume
    them.
    """
    existing_ids = _materialize_existing_params(param_groups)
    param_groups.append(
        _new_group(
            model,
            encoder_lr=encoder_lr,
            encoder_weight_decay=encoder_weight_decay,
            group_name=group_name,
            existing_ids=existing_ids,
        )
    )
    return param_groups


def add_randlora_to_optimizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    *,
    encoder_lr: float = 1e-4,
    encoder_weight_decay: float = 0.0,
    group_name: str = "randlora_encoder",
) -> torch.optim.Optimizer:
    """Safely add adapters to an already-created optimizer.

    Uses PyTorch's public ``Optimizer.add_param_group`` API and preserves every
    pre-existing group exactly.
    """
    existing_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", ())
    }
    group = _new_group(
        model,
        encoder_lr=encoder_lr,
        encoder_weight_decay=encoder_weight_decay,
        group_name=group_name,
        existing_ids=existing_ids,
    )
    optimizer.add_param_group(group)
    return optimizer


def build_adamw_param_groups(
    model: nn.Module,
    *,
    encoder_lr: float = 1e-4,
    head_lr: Optional[float] = None,
    encoder_weight_decay: float = 0.0,
    head_weight_decay: float = 0.01,
):
    """Convenience groups for simple trainers.

    Complex existing optimizers should preserve their original head groups and
    call :func:`append_randlora_param_group` or
    :func:`add_randlora_to_optimizer`.
    """
    _validate_group_hparams(encoder_lr, encoder_weight_decay)
    if head_lr is not None and (not math.isfinite(float(head_lr)) or head_lr <= 0):
        raise ValueError("head_lr must be finite and positive")
    if not math.isfinite(float(head_weight_decay)) or head_weight_decay < 0:
        raise ValueError("head_weight_decay must be finite and non-negative")
    adapter_params = []
    other_trainable = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_randlora_parameter(name):
            adapter_params.append(param)
        else:
            other_trainable.append(param)
    if not adapter_params:
        raise RuntimeError("no trainable RandLoRA parameters found")
    groups = [
        {
            "params": adapter_params,
            "lr": float(encoder_lr),
            "weight_decay": float(encoder_weight_decay),
            "group_name": "randlora_encoder",
        }
    ]
    if other_trainable:
        groups.append(
            {
                "params": other_trainable,
                "lr": float(head_lr if head_lr is not None else encoder_lr),
                "weight_decay": float(head_weight_decay),
                "group_name": "existing_trainable_heads",
            }
        )
    return groups


def build_adamw(
    model: nn.Module,
    *,
    encoder_lr: float = 1e-4,
    head_lr: Optional[float] = None,
    encoder_weight_decay: float = 0.0,
    head_weight_decay: float = 0.01,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    groups = build_adamw_param_groups(
        model,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        encoder_weight_decay=encoder_weight_decay,
        head_weight_decay=head_weight_decay,
    )
    return torch.optim.AdamW(groups, betas=betas, eps=eps)
