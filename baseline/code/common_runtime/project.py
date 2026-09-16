from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


def add_project_paths(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    candidates = [
        root,
        root / "baseline" / "code",
        root / "baseline" / "vendor" / "segment-anything",
        root / "baseline" / "vendor" / "tq-motp",
        root / "baseline" / "vendor" / "classification-heads",
        root / "models" / "segment-anything",
        root / "external" / "tq_motp_head_code" / "tq_motp_head_code",
        root / "experiments" / "tq_motp_lora_joint_gtbox_20260724T144600Z" / "code",
        root / "experiments" / "sam1_q_series" / "code" / ".deps",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root


def import_module_from_file(path: str | Path, module_name: str) -> ModuleType:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model_class(model_file: str | Path, project_root: str | Path):
    add_project_paths(project_root)
    module = import_module_from_file(model_file, "tqmotp_qj2_unified_runtime")
    if hasattr(module, "TQMOTPQJ2SAMModel"):
        return module.TQMOTPQJ2SAMModel
    if hasattr(module, "build_project_model_class"):
        return module.build_project_model_class(project_root)
    raise AttributeError(
        f"{model_file} must expose TQMOTPQJ2SAMModel or build_project_model_class"
    )


def build_model(model_file: str | Path, project_root: str | Path, model_cfg: dict[str, Any]):
    model_class = load_model_class(model_file, project_root)
    supported = {
        "num_classes": int(model_cfg.get("num_classes", 8)),
        "lora_rank": int(model_cfg.get("lora_rank", 4)),
        "lora_alpha": float(model_cfg.get("lora_alpha", 4.0)),
        "lora_dropout": float(model_cfg.get("lora_dropout", 0.05)),
        "adapter_type": str(model_cfg.get("adapter_type", "lora")),
        "randlora_config": model_cfg.get("randlora_config"),
    }
    try:
        return model_class(**supported)
    except TypeError as error:
        raise TypeError(f"failed to construct unified model with {supported}: {error}") from error


def import_tqmotp_loss(project_root: str | Path):
    add_project_paths(project_root)
    try:
        from tq_motp import TQMOTPLoss
    except ImportError as error:
        raise ImportError(
            "TQMOTPLoss was not importable. Verify the official TQ-MOTP source under "
            f"{Path(project_root) / 'external/tq_motp_head_code/tq_motp_head_code'}"
        ) from error
    return TQMOTPLoss


def optimizer_coverage_audit(model: torch.nn.Module, groups: list[dict[str, Any]]) -> dict[str, Any]:
    trainable = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    frozen = {id(parameter): name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    memberships: dict[int, list[str]] = {}
    unknown: list[int] = []
    for group_index, group in enumerate(groups):
        name = str(group.get("name", f"group_{group_index}"))
        for parameter in group["params"]:
            identifier = id(parameter)
            memberships.setdefault(identifier, []).append(name)
            if identifier not in trainable and identifier not in frozen:
                unknown.append(identifier)
    missing = [name for identifier, name in trainable.items() if identifier not in memberships]
    duplicate = [trainable.get(identifier, str(identifier)) for identifier, names in memberships.items() if len(names) > 1]
    frozen_in_optimizer = [frozen[identifier] for identifier in memberships if identifier in frozen]
    result = {
        "pass": not missing and not duplicate and not frozen_in_optimizer and not unknown,
        "trainable_parameter_tensors": len(trainable),
        "trainable_numel": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "optimizer_unique_parameter_tensors": len(memberships),
        "missing_trainable": missing,
        "duplicate_trainable": duplicate,
        "frozen_in_optimizer": frozen_in_optimizer,
        "unknown_optimizer_parameter_ids": unknown,
        "groups": [
            {
                "name": group.get("name", f"group_{index}"),
                "lr": float(group.get("lr", 0.0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_tensors": len(group["params"]),
                "numel": int(sum(parameter.numel() for parameter in group["params"])),
            }
            for index, group in enumerate(groups)
        ],
    }
    return result
