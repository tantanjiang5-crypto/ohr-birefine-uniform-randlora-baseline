from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_jsonable(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def move_state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def state_dict_max_abs_diff(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> tuple[float, str | None]:
    if left.keys() != right.keys():
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise RuntimeError(f"state_dict keys differ: missing_left={missing_left[:5]}, missing_right={missing_right[:5]}")
    maximum = 0.0
    maximum_name: str | None = None
    for name in left:
        a = left[name].detach().cpu()
        b = right[name].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"state mismatch for {name}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        if a.numel() == 0:
            continue
        difference = float((a.float() - b.float()).abs().max().item())
        if difference > maximum:
            maximum = difference
            maximum_name = name
    return maximum, maximum_name
