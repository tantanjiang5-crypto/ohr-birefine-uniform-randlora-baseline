from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from .config import RandLoRADamageConfig
from .core import FusedQKVRandLoRA, RandLoRABasisBank, RandLoRALinear
from .sam_injector import (
    _resolve_image_encoder,
    clear_adapter_caches,
    iter_randlora_coefficients,
    iter_randlora_wrappers,
)


FORMAT_VERSION = 3
PACKAGE_VERSION = "1.3.0"


def is_randlora_state_key(name: str) -> bool:
    return (
        "_randlora_basis_banks" in name
        or "randlora_lambda" in name
        or "randlora_gamma" in name
    )


def _assert_unmerged(model: nn.Module) -> None:
    merged = [i for i, module in enumerate(iter_randlora_wrappers(model)) if module.merged]
    if merged:
        raise RuntimeError(
            "adapter checkpoint operations require an unmerged model; "
            f"found {len(merged)} merged wrapper(s)"
        )


def adapter_state_dict(model_or_encoder: nn.Module) -> Dict[str, torch.Tensor]:
    """Return an immutable snapshot with keys relative to ``image_encoder``."""
    encoder = _resolve_image_encoder(model_or_encoder)
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in encoder.state_dict().items()
        if is_randlora_state_key(name)
    }


def _manifest(encoder: nn.Module) -> Tuple[Dict[str, object], ...]:
    value = encoder.__dict__.get("_randlora_manifest", ())
    return tuple(dict(item) for item in value)


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _basis_fingerprints(encoder: nn.Module) -> Dict[str, Dict[str, str]]:
    registry = getattr(encoder, "_randlora_basis_banks", None)
    if not isinstance(registry, nn.ModuleDict):
        return {}
    return {
        key: {
            "basis_a": _tensor_digest(bank.basis_a),
            "basis_b": _tensor_digest(bank.basis_b),
        }
        for key, bank in registry.items()
        if isinstance(bank, RandLoRABasisBank)
    }


def _state_digest(
    state: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, Any],
    manifest: Tuple[Dict[str, object], ...],
    resolved_default_rank: Any,
    basis_fingerprints: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "config": config,
        "manifest": list(manifest),
        "resolved_default_rank": resolved_default_rank,
        "basis_fingerprints": basis_fingerprints,
        "metadata": dict(metadata or {}),
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(_tensor_digest(state[key]).encode())
    return digest.hexdigest()


def _validate_metadata(value: Any, path: str = "metadata") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} float values must be finite")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_metadata(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_metadata(item, f"{path}.{key}")
        return
    raise TypeError(
        f"{path} contains unsupported type {type(value).__name__}; use JSON-like primitives"
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_adapter_checkpoint(
    model_or_encoder: nn.Module,
    path: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    _assert_unmerged(model_or_encoder)
    encoder = _resolve_image_encoder(model_or_encoder)
    config = encoder.__dict__.get("_randlora_damage_config")
    if not isinstance(config, RandLoRADamageConfig):
        raise RuntimeError("model has no RandLoRA-Damage configuration")
    metadata_payload = dict(metadata or {})
    _validate_metadata(metadata_payload)
    state = adapter_state_dict(encoder)
    manifest = _manifest(encoder)
    resolved_default_rank = encoder.__dict__.get("_randlora_default_rank")
    basis_fingerprints = _basis_fingerprints(encoder)
    config_payload = config.to_dict()
    payload = {
        "format": "randlora-damage-encoder",
        "format_version": FORMAT_VERSION,
        "package_version": PACKAGE_VERSION,
        "config": config_payload,
        "resolved_default_rank": resolved_default_rank,
        "manifest": list(manifest),
        "basis_fingerprints": basis_fingerprints,
        "state_dict": state,
        "metadata": metadata_payload,
    }
    payload["state_digest"] = _state_digest(
        state,
        config=config_payload,
        manifest=manifest,
        resolved_default_rank=resolved_default_rank,
        basis_fingerprints=basis_fingerprints,
        metadata=metadata_payload,
    )
    path = Path(path)
    _atomic_torch_save(payload, path)
    return path


def _safe_torch_load(path: str | Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch <2.0 compatibility
        return torch.load(path, map_location=map_location)


def _normalize_legacy_key(key: str, encoder_keys: Mapping[str, torch.Tensor]) -> Optional[str]:
    if key in encoder_keys:
        return key
    prefix = "image_encoder."
    if key.startswith(prefix) and key[len(prefix) :] in encoder_keys:
        return key[len(prefix) :]
    matches = [current for current in encoder_keys if key.endswith(current)]
    return matches[0] if len(matches) == 1 else None


def _config_comparison_payload(config: RandLoRADamageConfig) -> Dict[str, Any]:
    # Compare every field that changes model semantics, training behavior or
    # reproducibility. budget_warning_tolerance is reporting-only.
    payload = config.to_dict()
    payload.pop("budget_warning_tolerance", None)
    return payload




def _manifest_for_comparison(
    manifest: Tuple[Dict[str, object], ...], version: int
) -> Tuple[Dict[str, object], ...]:
    if version >= 3:
        return tuple(dict(item) for item in manifest)
    # v1.1/v2 manifests predate alpha/scaling/basis-key/dtype fields. Compare
    # the structural fields they actually recorded so old adapters remain
    # portable without weakening v3 checks.
    legacy_fields = (
        "path",
        "block",
        "target",
        "in_features",
        "out_features",
        "rank",
        "num_bases",
    )
    return tuple({key: item.get(key) for key in legacy_fields} for item in manifest)


def _basis_fingerprints_from_state(
    state: Mapping[str, torch.Tensor]
) -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {}
    marker = "_randlora_basis_banks."
    for key, tensor in state.items():
        if marker not in key or not isinstance(tensor, torch.Tensor):
            continue
        suffix = key.split(marker, 1)[1]
        if suffix.endswith(".basis_a"):
            bank_key = suffix[: -len(".basis_a")]
            found.setdefault(bank_key, {})["basis_a"] = _tensor_digest(tensor)
        elif suffix.endswith(".basis_b"):
            bank_key = suffix[: -len(".basis_b")]
            found.setdefault(bank_key, {})["basis_b"] = _tensor_digest(tensor)
    return found


def load_adapter_checkpoint(
    model_or_encoder: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    check_config: bool = True,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    _assert_unmerged(model_or_encoder)
    payload = _safe_torch_load(path, map_location)
    if not isinstance(payload, Mapping) or payload.get("format") != "randlora-damage-encoder":
        raise ValueError("not a RandLoRA-Damage adapter checkpoint")
    version = int(payload.get("format_version", -1))
    if version not in (1, 2, FORMAT_VERSION):
        raise ValueError(f"unsupported adapter format version: {version}")

    encoder = _resolve_image_encoder(model_or_encoder)
    current = encoder.state_dict()
    expected_keys = {key for key in current if is_randlora_state_key(key)}
    if not expected_keys:
        raise RuntimeError("current model has no RandLoRA adapter state")
    raw_saved = payload.get("state_dict")
    if not isinstance(raw_saved, Mapping):
        raise ValueError("checkpoint state_dict is missing or invalid")

    integrity_error = None
    if version >= 3:
        required = {
            "config",
            "resolved_default_rank",
            "manifest",
            "basis_fingerprints",
            "state_digest",
        }
        missing_payload_fields = sorted(required - set(payload))
        if missing_payload_fields:
            integrity_error = f"missing required payload fields: {missing_payload_fields}"
        else:
            raw_tensor_state = {
                key: value
                for key, value in raw_saved.items()
                if isinstance(key, str) and isinstance(value, torch.Tensor)
            }
            try:
                expected_digest = _state_digest(
                    raw_tensor_state,
                    config=payload["config"],
                    manifest=tuple(dict(x) for x in payload["manifest"]),
                    resolved_default_rank=payload["resolved_default_rank"],
                    basis_fingerprints=payload["basis_fingerprints"],
                    metadata=payload.get("metadata", {}),
                )
                if expected_digest != payload["state_digest"]:
                    integrity_error = "state digest mismatch"
            except Exception as exc:
                integrity_error = f"could not verify state digest: {exc}"
        if integrity_error is not None:
            raise RuntimeError(f"adapter checkpoint integrity failure: {integrity_error}")

    normalized_saved: Dict[str, torch.Tensor] = {}
    unrecognized = []
    duplicates = []
    forbidden_non_adapter_keys = []
    for raw_key, value in raw_saved.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            unrecognized.append(str(raw_key))
            continue
        key = raw_key if version >= 2 else _normalize_legacy_key(raw_key, current)
        if key is None or key not in current:
            unrecognized.append(raw_key)
            continue
        if not is_randlora_state_key(key):
            forbidden_non_adapter_keys.append(key)
            continue
        if key in normalized_saved:
            duplicates.append(key)
            continue
        normalized_saved[key] = value

    shape_errors = []
    loadable: Dict[str, torch.Tensor] = {}
    for key, value in normalized_saved.items():
        if tuple(value.shape) != tuple(current[key].shape):
            shape_errors.append((key, tuple(value.shape), tuple(current[key].shape)))
        else:
            loadable[key] = value

    missing_from_checkpoint = sorted(expected_keys - set(loadable))
    unexpected_in_checkpoint = sorted(unrecognized)

    try:
        saved_config = RandLoRADamageConfig.from_dict(payload.get("config", {}))
    except Exception as exc:
        raise ValueError(f"invalid checkpoint config: {exc}") from exc
    current_config = encoder.__dict__.get("_randlora_damage_config")
    config_mismatch = None
    if check_config:
        if not isinstance(current_config, RandLoRADamageConfig):
            config_mismatch = {"saved": _config_comparison_payload(saved_config), "current": None}
        elif _config_comparison_payload(saved_config) != _config_comparison_payload(current_config):
            config_mismatch = {
                "saved": _config_comparison_payload(saved_config),
                "current": _config_comparison_payload(current_config),
            }

    saved_manifest_raw = payload.get("manifest", ())
    try:
        saved_manifest = tuple(dict(x) for x in saved_manifest_raw) if saved_manifest_raw else ()
    except Exception as exc:
        raise ValueError(f"invalid checkpoint manifest: {exc}") from exc
    current_manifest = _manifest(encoder)
    manifest_missing = version >= 2 and not saved_manifest
    current_manifest_missing = not current_manifest
    manifest_mismatch = bool(
        saved_manifest
        and current_manifest
        and _manifest_for_comparison(saved_manifest, version)
        != _manifest_for_comparison(current_manifest, version)
    )

    saved_resolved_rank = payload.get("resolved_default_rank")
    current_resolved_rank = encoder.__dict__.get("_randlora_default_rank")
    resolved_rank_invalid = version >= 3 and (
        not isinstance(saved_resolved_rank, int) or isinstance(saved_resolved_rank, bool)
    )
    resolved_rank_mismatch = resolved_rank_invalid or (
        saved_resolved_rank is not None
        and current_resolved_rank is not None
        and int(saved_resolved_rank) != int(current_resolved_rank)
    )

    saved_basis_fingerprints = payload.get("basis_fingerprints", {})
    basis_fingerprint_preload_mismatch = False
    basis_state_fingerprint_mismatch = False
    if version >= 3:
        if not isinstance(saved_basis_fingerprints, Mapping):
            basis_fingerprint_preload_mismatch = True
        elif saved_config.save_basis:
            basis_state_fingerprint_mismatch = (
                _basis_fingerprints_from_state(raw_saved)
                != dict(saved_basis_fingerprints)
            )
        else:
            basis_fingerprint_preload_mismatch = (
                dict(saved_basis_fingerprints) != _basis_fingerprints(encoder)
            )

    problems = {
        "missing_from_checkpoint": missing_from_checkpoint,
        "unexpected_in_checkpoint": unexpected_in_checkpoint,
        "forbidden_non_adapter_keys": sorted(set(forbidden_non_adapter_keys)),
        "duplicate_keys": sorted(set(duplicates)),
        "shape_errors": shape_errors,
        "config_mismatch": config_mismatch,
        "manifest_missing": manifest_missing,
        "current_manifest_missing": current_manifest_missing,
        "manifest_mismatch": manifest_mismatch,
        "resolved_rank_mismatch": resolved_rank_mismatch,
        "basis_fingerprint_preload_mismatch": basis_fingerprint_preload_mismatch,
        "basis_state_fingerprint_mismatch": basis_state_fingerprint_mismatch,
    }
    mismatch_values = (
        missing_from_checkpoint,
        unexpected_in_checkpoint,
        forbidden_non_adapter_keys,
        duplicates,
        shape_errors,
        config_mismatch is not None,
        manifest_missing,
        current_manifest_missing,
        manifest_mismatch,
        resolved_rank_mismatch,
        basis_fingerprint_preload_mismatch,
        basis_state_fingerprint_mismatch,
    )
    if strict and any(mismatch_values):
        raise RuntimeError(f"adapter checkpoint mismatch: {problems}")

    pre_load_snapshot = adapter_state_dict(encoder) if strict else None
    encoder.load_state_dict(loadable, strict=False)
    clear_adapter_caches(encoder)

    basis_fingerprint_postload_mismatch = False
    if version >= 3:
        declared_fingerprints = (
            dict(saved_basis_fingerprints)
            if isinstance(saved_basis_fingerprints, Mapping)
            else {}
        )
        basis_fingerprint_postload_mismatch = (
            declared_fingerprints != _basis_fingerprints(encoder)
        )
        if strict and basis_fingerprint_postload_mismatch:
            assert pre_load_snapshot is not None
            encoder.load_state_dict(pre_load_snapshot, strict=False)
            clear_adapter_caches(encoder)
            raise RuntimeError(
                "adapter checkpoint mismatch: basis fingerprints differ after load; "
                "the pre-load adapter state was restored"
            )

    return {
        "format_version": version,
        "package_version": payload.get("package_version"),
        "config": saved_config,
        "metadata": payload.get("metadata", {}),
        "loaded_keys": tuple(sorted(loadable)),
        "integrity_verified": version >= 3,
        "basis_fingerprint_postload_mismatch": basis_fingerprint_postload_mismatch,
        **problems,
    }


@torch.no_grad()
def merge_all(model: nn.Module, *, safe: bool = True) -> None:
    """Merge transactionally; rollback newly merged layers on failure."""
    merged = []
    try:
        for module in iter_randlora_wrappers(model):
            if module.merged:
                continue
            module.merge(safe=safe)
            merged.append(module)
    except Exception:
        for module in reversed(merged):
            module.unmerge()
        raise


@torch.no_grad()
def unmerge_all(model: nn.Module) -> None:
    for module in iter_randlora_wrappers(model):
        module.unmerge()


def _unload_merged_recursive(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (FusedQKVRandLoRA, RandLoRALinear)):
            if not child.merged:
                raise RuntimeError("internal error: wrapper must be merged before unload")
            setattr(module, name, child.base_layer)
        else:
            _unload_merged_recursive(child)


def merge_and_unload(model: nn.Module, *, safe: bool = True) -> nn.Module:
    """Mutate into an ordinary SAM-compatible model with merged weights."""
    merge_all(model, safe=safe)
    _unload_merged_recursive(model)
    encoder = _resolve_image_encoder(model)
    if hasattr(encoder, "_randlora_basis_banks"):
        delattr(encoder, "_randlora_basis_banks")
    for key in ("_randlora_damage_config", "_randlora_default_rank", "_randlora_target_ranks", "_randlora_manifest"):
        encoder.__dict__.pop(key, None)
    return model


def _validate_copied_basis_links(model: nn.Module) -> None:
    encoder = _resolve_image_encoder(model)
    registry = getattr(encoder, "_randlora_basis_banks", None)
    if registry is None:
        return
    registered_ids = {id(bank) for bank in registry.values()}
    for coeff in iter_randlora_coefficients(model):
        if id(coeff.basis_bank) not in registered_ids:
            raise RuntimeError("deep-copied coefficient is not linked to the copied basis registry")


def merged_copy(model: nn.Module, *, safe: bool = True) -> nn.Module:
    """Deep-copy then merge/unload. Reject an already merged source model."""
    _assert_unmerged(model)
    copied = copy.deepcopy(model)
    _validate_copied_basis_links(copied)
    return merge_and_unload(copied, safe=safe)
