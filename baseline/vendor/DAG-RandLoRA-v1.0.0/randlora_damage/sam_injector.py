from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

from .config import (
    RandLoRADamageConfig,
    canonical_target_key,
    find_rank_for_budget,
    randlora_trainable_params,
)
from .core import FusedQKVRandLoRA, RandLoRABasisBank, RandLoRACoefficients, RandLoRALinear


@dataclass(frozen=True)
class InjectionReport:
    selected_blocks: Tuple[int, ...]
    qkv_targets: Tuple[str, ...]
    extra_targets: Tuple[str, ...]
    default_rank: int
    block_ranks: Dict[int, int]
    target_ranks: Dict[str, int]
    requested_budget: Optional[int]
    actual_adapter_params: int
    budget_relative_error: Optional[float]
    budget_warning: Optional[str]
    trainable_image_encoder_params: int
    total_image_encoder_params: int
    wrapped_modules: Tuple[str, ...]
    module_manifest: Tuple[Dict[str, object], ...]

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_image_encoder_params / max(self.total_image_encoder_params, 1)


@dataclass(frozen=True)
class _TargetSpec:
    block_idx: int
    name: str
    in_features: int
    out_features: int

    @property
    def key(self) -> str:
        return canonical_target_key(self.block_idx, self.name)


def _resolve_image_encoder(model_or_encoder: nn.Module) -> nn.Module:
    encoder = getattr(model_or_encoder, "image_encoder", model_or_encoder)
    if not isinstance(encoder, nn.Module):
        raise TypeError("image_encoder must be an nn.Module")
    return encoder


def _get_linear_by_target(block: nn.Module, target: str) -> nn.Linear:
    if target == "attn.proj":
        layer = block.attn.proj
    elif target == "mlp.lin1":
        layer = block.mlp.lin1
    elif target == "mlp.lin2":
        layer = block.mlp.lin2
    else:
        raise KeyError(target)
    if isinstance(layer, RandLoRALinear):
        raise RuntimeError(f"target {target} is already RandLoRA-wrapped")
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"{target} must be nn.Linear, got {type(layer).__name__}")
    return layer


def _set_target(block: nn.Module, target: str, module: nn.Module) -> None:
    if target == "attn.proj":
        block.attn.proj = module
    elif target == "mlp.lin1":
        block.mlp.lin1 = module
    elif target == "mlp.lin2":
        block.mlp.lin2 = module
    else:
        raise KeyError(target)


def _collect_target_specs(encoder: nn.Module, config: RandLoRADamageConfig) -> List[_TargetSpec]:
    specs: List[_TargetSpec] = []
    for block_idx in config.block_indices:
        block = encoder.blocks[block_idx]
        if not hasattr(block, "attn") or not hasattr(block.attn, "qkv"):
            raise AttributeError(f"block {block_idx} lacks official SAM1 attn.qkv")
        qkv = block.attn.qkv
        if isinstance(qkv, FusedQKVRandLoRA):
            raise RuntimeError(f"block {block_idx} qkv is already RandLoRA-wrapped")
        if not isinstance(qkv, nn.Linear) or qkv.out_features != 3 * qkv.in_features:
            raise TypeError(
                f"block {block_idx} does not expose official SAM1 fused qkv nn.Linear; "
                "SAM2/SAM3 require a dedicated injector"
            )
        for target in config.qkv_targets:
            specs.append(_TargetSpec(block_idx, target, qkv.in_features, qkv.in_features))
        for target in config.extra_targets:
            layer = _get_linear_by_target(block, target)
            specs.append(_TargetSpec(block_idx, target, layer.in_features, layer.out_features))
    return specs


def _basis_key(rank: int, max_dim: int, min_dim: int, seed: int) -> str:
    return f"r{rank}_max{max_dim}_min{min_dim}_seed{seed}"


def _adapter_dtype(config: RandLoRADamageConfig, base_weight: torch.Tensor) -> torch.dtype:
    return torch.float32 if config.adapter_dtype == "float32" else base_weight.dtype


def _reference_weight(encoder: nn.Module, spec: _TargetSpec) -> torch.Tensor:
    block = encoder.blocks[spec.block_idx]
    if spec.name in {"q", "v"}:
        return block.attn.qkv.weight
    return _get_linear_by_target(block, spec.name).weight


def inject_randlora_damage_encoder(
    model_or_encoder: nn.Module,
    config: Optional[RandLoRADamageConfig] = None,
) -> InjectionReport:
    """Inject RandLoRA/DAG-RandLoRA into an official SAM1 ViT encoder.

    v1.3 adds target-specific Q/V rank allocation while preserving all v1.2
    behavior when ``target_rank_pattern`` is empty. The original SAM checkpoint
    must be loaded before injection. Prompt encoder, mask decoder and heads are
    untouched.
    """
    config = copy.deepcopy(config or RandLoRADamageConfig())
    encoder = _resolve_image_encoder(model_or_encoder)
    if not hasattr(encoder, "blocks"):
        raise AttributeError("expected official SAM1 image_encoder.blocks")
    if hasattr(encoder, "_randlora_basis_banks"):
        raise RuntimeError("encoder already contains a RandLoRA basis registry")

    config.validate(num_blocks=len(encoder.blocks))
    specs = _collect_target_specs(encoder, config)
    shapes = [(s.in_features, s.out_features) for s in specs]
    if config.auto_match_budget and config.target_trainable_params is not None:
        default_rank, _ = find_rank_for_budget(shapes, config.target_trainable_params)
    elif config.rank is not None:
        default_rank = config.rank
    else:
        raise ValueError("could not resolve RandLoRA rank")

    block_ranks = {idx: config.rank_for_block(idx, default_rank) for idx in config.block_indices}
    target_ranks = {
        spec.key: config.rank_for_target(spec.block_idx, spec.name, default_rank) for spec in specs
    }
    for spec in specs:
        rank = target_ranks[spec.key]
        min_dim = min(spec.in_features, spec.out_features)
        if rank > min_dim:
            raise ValueError(
                f"rank {rank} exceeds min dimension {min_dim} for "
                f"block {spec.block_idx} target {spec.name}"
            )

    preserve_float32 = config.adapter_dtype == "float32"
    registry = nn.ModuleDict()
    bank_by_rank: Dict[int, RandLoRABasisBank] = {}
    bank_key_by_rank: Dict[int, str] = {}
    for rank in sorted(set(target_ranks.values())):
        rank_specs = [s for s in specs if target_ranks[s.key] == rank]
        max_dim = max(max(s.in_features, s.out_features) for s in rank_specs)
        min_dim = max(min(s.in_features, s.out_features) for s in rank_specs)
        key = _basis_key(rank, max_dim, min_dim, config.seed)
        bank = RandLoRABasisBank(
            rank=rank,
            max_dim=max_dim,
            min_dim=min_dim,
            seed=config.seed,
            persistent=config.save_basis,
            sparse=config.sparse_basis,
            very_sparse=config.very_sparse_basis,
            preserve_float32=preserve_float32,
        )
        ref = _reference_weight(encoder, rank_specs[0])
        bank = bank.to(device=ref.device, dtype=_adapter_dtype(config, ref))
        registry[key] = bank
        bank_by_rank[rank] = bank
        bank_key_by_rank[rank] = key

    qkv_replacements: Dict[int, FusedQKVRandLoRA] = {}
    extra_replacements: Dict[Tuple[int, str], RandLoRALinear] = {}
    wrapped: List[str] = []
    manifest: List[Dict[str, object]] = []

    for block_idx in config.block_indices:
        block = encoder.blocks[block_idx]
        if config.qkv_targets:
            qkv = block.attn.qkv
            coeffs: Dict[str, RandLoRACoefficients] = {}
            for target in config.qkv_targets:
                key = canonical_target_key(block_idx, target)
                rank = target_ranks[key]
                alpha = config.alpha_for_rank(rank)
                bank = bank_by_rank[rank]
                coeff = RandLoRACoefficients(
                    in_features=qkv.in_features,
                    out_features=qkv.in_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=config.dropout,
                    forward_mode=config.forward_mode,
                    cache_eval_delta=config.cache_eval_delta,
                    basis_bank=bank,
                    preserve_float32=preserve_float32,
                ).to(device=qkv.weight.device, dtype=_adapter_dtype(config, qkv.weight))
                coeffs[target] = coeff
                manifest.append(
                    {
                        "path": f"blocks.{block_idx}.attn.qkv.{target}",
                        "block": block_idx,
                        "target": target,
                        "in_features": qkv.in_features,
                        "out_features": qkv.in_features,
                        "rank": rank,
                        "num_bases": coeff.num_bases,
                        "alpha": alpha,
                        "scaling": coeff.scaling,
                        "basis_key": bank_key_by_rank[rank],
                        "adapter_dtype": config.adapter_dtype,
                    }
                )
            qkv_replacements[block_idx] = FusedQKVRandLoRA(qkv, coeffs)
            wrapped.append(f"blocks.{block_idx}.attn.qkv[{','.join(config.qkv_targets)}]")

        for target in config.extra_targets:
            layer = _get_linear_by_target(block, target)
            key = canonical_target_key(block_idx, target)
            rank = target_ranks[key]
            alpha = config.alpha_for_rank(rank)
            bank = bank_by_rank[rank]
            coeff = RandLoRACoefficients(
                in_features=layer.in_features,
                out_features=layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=config.dropout,
                forward_mode=config.forward_mode,
                cache_eval_delta=config.cache_eval_delta,
                basis_bank=bank,
                preserve_float32=preserve_float32,
            ).to(device=layer.weight.device, dtype=_adapter_dtype(config, layer.weight))
            extra_replacements[(block_idx, target)] = RandLoRALinear(layer, coeff)
            wrapped.append(f"blocks.{block_idx}.{target}")
            manifest.append(
                {
                    "path": f"blocks.{block_idx}.{target}",
                    "block": block_idx,
                    "target": target,
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "rank": rank,
                    "num_bases": coeff.num_bases,
                    "alpha": alpha,
                    "scaling": coeff.scaling,
                    "basis_key": bank_key_by_rank[rank],
                    "adapter_dtype": config.adapter_dtype,
                }
            )

    original_requires_grad = {name: p.requires_grad for name, p in encoder.named_parameters()}
    original_qkv = {idx: encoder.blocks[idx].attn.qkv for idx in qkv_replacements}
    original_extra = {
        key: _get_linear_by_target(encoder.blocks[key[0]], key[1]) for key in extra_replacements
    }

    try:
        if config.freeze_image_encoder:
            encoder.requires_grad_(False)
        encoder.add_module("_randlora_basis_banks", registry)
        for block_idx, wrapper in qkv_replacements.items():
            wrapper.base_layer.requires_grad_(False)
            encoder.blocks[block_idx].attn.qkv = wrapper
        for (block_idx, target), wrapper in extra_replacements.items():
            wrapper.base_layer.requires_grad_(False)
            _set_target(encoder.blocks[block_idx], target, wrapper)
    except Exception:
        for block_idx, original in original_qkv.items():
            encoder.blocks[block_idx].attn.qkv = original
        for (block_idx, target), original in original_extra.items():
            _set_target(encoder.blocks[block_idx], target, original)
        if hasattr(encoder, "_randlora_basis_banks"):
            delattr(encoder, "_randlora_basis_banks")
        for name, parameter in encoder.named_parameters():
            if name in original_requires_grad:
                parameter.requires_grad_(original_requires_grad[name])
        raise

    encoder.__dict__["_randlora_damage_config"] = config
    encoder.__dict__["_randlora_default_rank"] = default_rank
    encoder.__dict__["_randlora_target_ranks"] = dict(target_ranks)
    encoder.__dict__["_randlora_manifest"] = tuple(manifest)

    adapter_params = sum(
        p.numel()
        for name, p in encoder.named_parameters()
        if "randlora_lambda" in name or "randlora_gamma" in name
    )
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    expected = sum(
        randlora_trainable_params(s.in_features, s.out_features, target_ranks[s.key]) for s in specs
    )
    if adapter_params != expected:
        raise RuntimeError(f"adapter parameter audit failed: actual={adapter_params}, expected={expected}")
    if config.freeze_image_encoder and trainable != adapter_params:
        raise RuntimeError(
            f"unexpected trainable image-encoder parameters: trainable={trainable}, adapter={adapter_params}"
        )

    budget_error: Optional[float] = None
    budget_warning: Optional[str] = None
    if config.target_trainable_params is not None:
        budget_error = abs(adapter_params - config.target_trainable_params) / config.target_trainable_params
        if budget_error > config.budget_warning_tolerance:
            budget_warning = (
                f"adapter budget differs by {budget_error:.2%}: requested="
                f"{config.target_trainable_params}, actual={adapter_params}. "
                "This can be caused by a block/target rank pattern or coarse candidate ranks."
            )
            warnings.warn(budget_warning, RuntimeWarning, stacklevel=2)

    return InjectionReport(
        selected_blocks=tuple(config.block_indices),
        qkv_targets=tuple(config.qkv_targets),
        extra_targets=tuple(config.extra_targets),
        default_rank=default_rank,
        block_ranks=block_ranks,
        target_ranks=target_ranks,
        requested_budget=config.target_trainable_params,
        actual_adapter_params=adapter_params,
        budget_relative_error=budget_error,
        budget_warning=budget_warning,
        trainable_image_encoder_params=trainable,
        total_image_encoder_params=total,
        wrapped_modules=tuple(wrapped),
        module_manifest=tuple(manifest),
    )


def iter_randlora_wrappers(model: nn.Module) -> Iterator[nn.Module]:
    for module in model.modules():
        if isinstance(module, (FusedQKVRandLoRA, RandLoRALinear)):
            yield module


def iter_randlora_coefficients(model: nn.Module) -> Iterator[RandLoRACoefficients]:
    for wrapper in iter_randlora_wrappers(model):
        if isinstance(wrapper, FusedQKVRandLoRA):
            yield from wrapper.coefficients.values()
        else:
            yield wrapper.coefficients


def clear_adapter_caches(model: nn.Module) -> None:
    for coeff in iter_randlora_coefficients(model):
        coeff.clear_cache()


def set_adapters_enabled(
    model: nn.Module,
    enabled: bool,
    *,
    unmerge_when_disabling: bool = True,
) -> None:
    for module in iter_randlora_wrappers(model):
        if not enabled and module.merged:
            if not unmerge_when_disabling:
                raise RuntimeError("cannot disable a merged adapter without unmerging it")
            module.unmerge()
        module.adapters_enabled = bool(enabled)
