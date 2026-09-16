from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@contextmanager
def _autocast_disabled(device_type: str) -> Iterator[None]:
    """Disable autocast on the CPU/CUDA backends supported by this package."""
    if device_type not in {"cpu", "cuda"}:
        # SAM1 experiments supported by this package run on CPU or CUDA. For an
        # unrecognized private backend, explicit FP32 tensors remain in effect
        # without invoking a backend-specific autocast API.
        yield
        return
    # Do not catch exceptions around this yield: numerical/shape failures from
    # adapter code must propagate unchanged.
    with torch.autocast(device_type=device_type, enabled=False):
        yield


class _ScaleSharedBasis(torch.autograd.Function):
    """Scale fixed bases while returning gradients only for lambda/gamma."""

    @staticmethod
    def forward(ctx, basis_a: torch.Tensor, lamb: torch.Tensor, gamma: torch.Tensor):
        # basis_a [r,1,d], lambda [r,n], gamma [n,d]
        out = basis_a * lamb[:, :, None] * gamma[None, :, :]
        ctx.save_for_backward(basis_a, lamb, gamma)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        basis_a, lamb, gamma = ctx.saved_tensors
        basis_a_2d = basis_a[:, 0, :].to(dtype=grad_output.dtype)
        lamb_f = lamb.to(dtype=grad_output.dtype)
        gamma_f = gamma.to(dtype=grad_output.dtype)
        grad_lamb = torch.einsum("rni,ri,ni->rn", grad_output, basis_a_2d, gamma_f)
        grad_gamma = torch.einsum("rni,ri,rn->ni", grad_output, basis_a_2d, lamb_f)
        return None, grad_lamb.to(lamb.dtype), grad_gamma.to(gamma.dtype)


def _safe_std(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.std(unbiased=False)


def _kaiming_uniform(shape: Tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    # torch 2.1 exposes the generator argument on torch.rand but not on
    # nn.init.kaiming_uniform_.  For a=sqrt(5), PyTorch's Kaiming initializer
    # reduces to U(-1/sqrt(fan_in), 1/sqrt(fan_in)); draw that distribution
    # explicitly so the package remains deterministic on the experiment's
    # pinned torch 2.1 runtime.
    if len(shape) < 2:
        raise ValueError("Kaiming uniform requires at least two dimensions")
    fan_in = int(shape[1])
    for size in shape[2:]:
        fan_in *= int(size)
    if fan_in <= 0:
        raise ValueError("invalid fan_in for random basis")
    bound = 1.0 / math.sqrt(float(fan_in))
    tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
    tensor = tensor.mul(2.0 * bound).sub(bound)
    std = _safe_std(tensor)
    if not torch.isfinite(std) or std <= 0:
        raise RuntimeError("invalid random basis standard deviation")
    return tensor / std


def _ternary_basis(
    shape: Tuple[int, ...], generator: torch.Generator, sparsity: float
) -> torch.Tensor:
    if not math.isfinite(sparsity) or sparsity <= 0:
        raise ValueError("sparsity must be finite and positive")
    uniform = torch.rand(shape, generator=generator, dtype=torch.float32)
    result = torch.zeros_like(uniform)
    p = 1.0 / (2.0 * sparsity)
    result[uniform < p] = -1.0
    result[uniform > 1.0 - p] = 1.0
    std = _safe_std(result)
    if not torch.isfinite(std) or std <= 0:
        raise RuntimeError("sparse basis collapsed to zero; choose lower sparsity")
    return result / std


class RandLoRABasisBank(nn.Module):
    """A shared pair of fixed random bases for compatible target matrices."""

    def __init__(
        self,
        *,
        rank: int,
        max_dim: int,
        min_dim: int,
        seed: int,
        persistent: bool = True,
        sparse: bool = False,
        very_sparse: bool = False,
        preserve_float32: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0 or min_dim <= 0 or max_dim < min_dim:
            raise ValueError("invalid basis dimensions")
        self.rank = int(rank)
        self.max_dim = int(max_dim)
        self.min_dim = int(min_dim)
        self.num_bases = math.ceil(min_dim / rank)
        self.seed = int(seed)
        self.persistent = bool(persistent)
        self.preserve_float32 = bool(preserve_float32)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        if very_sparse:
            sparsity = math.sqrt(min_dim)
            basis_a = _ternary_basis((rank, 1, min_dim), generator, sparsity)
            basis_b = _ternary_basis((max_dim, self.num_bases, rank), generator, sparsity)
        elif sparse:
            basis_a = _ternary_basis((rank, 1, min_dim), generator, 3.0)
            basis_b = _ternary_basis((max_dim, self.num_bases, rank), generator, 3.0)
        else:
            basis_a = _kaiming_uniform((rank, 1, min_dim), generator)
            basis_b = torch.cat(
                [_kaiming_uniform((max_dim, 1, rank), generator) for _ in range(self.num_bases)],
                dim=1,
            )
            basis_b = basis_b / _safe_std(basis_b)

        self.register_buffer("basis_a", basis_a, persistent=persistent)
        self.register_buffer("basis_b", basis_b, persistent=persistent)

    def _apply(self, fn):
        result = super()._apply(fn)
        if self.preserve_float32:
            # Preserve device moves while rejecting parent-level half()/bfloat16()
            # casts that would silently defeat adapter_dtype="float32".
            self._buffers["basis_a"] = self.basis_a.float()
            self._buffers["basis_b"] = self.basis_b.float()
        return result

    def slices(
        self,
        *,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        min_dim = min(in_features, out_features)
        max_dim = max(in_features, out_features)
        expected_bases = math.ceil(min_dim / self.rank)
        if min_dim > self.min_dim or max_dim > self.max_dim or expected_bases > self.num_bases:
            raise ValueError(
                "basis bank is too small for target: "
                f"target=({out_features},{in_features}), bank=({self.max_dim},{self.min_dim})"
            )
        a = self.basis_a[:, :, :min_dim].to(device=device, dtype=dtype)
        b = self.basis_b[:max_dim, :expected_bases, :].to(device=device, dtype=dtype)
        return a, b


class RandLoRACoefficients(nn.Module):
    """Per-target trainable diagonal scales over a shared fixed basis bank."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        forward_mode: str,
        cache_eval_delta: bool,
        basis_bank: RandLoRABasisBank,
        preserve_float32: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.num_bases = math.ceil(min(in_features, out_features) / rank)
        self.scaling = float(alpha) / float(rank)
        self.forward_mode = str(forward_mode)
        self.cache_eval_delta = bool(cache_eval_delta)
        self.preserve_float32 = bool(preserve_float32)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.randlora_lambda = nn.Parameter(torch.zeros(rank, self.num_bases))
        self.randlora_gamma = nn.Parameter(
            torch.full(
                (self.num_bases, min(in_features, out_features)),
                1.0 / max(in_features, out_features),
            )
        )
        # Keep a strong, non-registered reference. Unlike a weakref, this is
        # pickle/deepcopy safe; unlike normal Module assignment, it does not
        # duplicate the shared bank in every coefficient's state_dict.
        object.__setattr__(self, "_basis_bank", basis_bank)
        self._cached_delta: Optional[torch.Tensor] = None
        self._cached_key: Optional[Tuple[object, ...]] = None

    @property
    def basis_bank(self) -> RandLoRABasisBank:
        bank = self.__dict__.get("_basis_bank")
        if not isinstance(bank, RandLoRABasisBank):
            raise RuntimeError("RandLoRA basis bank is unavailable")
        return bank

    def rebind_basis_bank(self, basis_bank: RandLoRABasisBank) -> None:
        object.__setattr__(self, "_basis_bank", basis_bank)
        self.clear_cache()

    def _cache_key(self) -> Tuple[object, ...]:
        bank = self.basis_bank
        return (
            self.randlora_lambda._version,
            self.randlora_gamma._version,
            self.randlora_lambda.device,
            self.randlora_lambda.dtype,
            bank.basis_a._version,
            bank.basis_b._version,
            bank.basis_a.device,
            bank.basis_a.dtype,
            self.scaling,
        )

    def clear_cache(self) -> None:
        self._cached_delta = None
        self._cached_key = None

    def train(self, mode: bool = True):
        self.clear_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_cache()
        result = super()._apply(fn)
        if self.preserve_float32:
            for parameter in (self.randlora_lambda, self.randlora_gamma):
                parameter.data = parameter.data.float()
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.float()
        self.clear_cache()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        result = super()._load_from_state_dict(*args, **kwargs)
        if self.preserve_float32:
            self.randlora_lambda.data = self.randlora_lambda.data.float()
            self.randlora_gamma.data = self.randlora_gamma.data.float()
        self.clear_cache()
        return result

    def _factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self.randlora_lambda.dtype
        device = self.randlora_lambda.device
        basis_a, basis_b = self.basis_bank.slices(
            in_features=self.in_features,
            out_features=self.out_features,
            dtype=dtype,
            device=device,
        )
        scaled_a = _ScaleSharedBasis.apply(
            basis_a,
            self.randlora_lambda,
            self.randlora_gamma,
        )
        # [r,n,d] -> [n*r,d], aligned with B=[max,n,r] -> [max,n*r]
        factor_small = scaled_a.permute(1, 0, 2).reshape(self.num_bases * self.rank, -1)
        factor_large = basis_b.flatten(start_dim=1)
        return factor_small, factor_large

    def _compute_delta_weight(self) -> torch.Tensor:
        # Delta construction must not depend on the ambient autocast dtype,
        # otherwise an eval cache created under AMP can be reused later with an
        # incompatible dtype or reduced precision.
        with _autocast_disabled(self.randlora_lambda.device.type):
            factor_small, factor_large = self._factors()
            if self.in_features <= self.out_features:
                delta = factor_large @ factor_small
            else:
                delta = factor_small.T @ factor_large.T
            return delta * self.scaling

    def delta_weight(self) -> torch.Tensor:
        # eval() does not disable autograd. Cache only in a true inference/no-grad
        # region so sensitivity analysis and validation backward retain gradients.
        allow_cache = not self.training and self.cache_eval_delta and not torch.is_grad_enabled()
        cache_key = self._cache_key()
        if allow_cache and self._cached_delta is not None and self._cached_key == cache_key:
            return self._cached_delta
        delta = self._compute_delta_weight()
        if allow_cache:
            self._cached_delta = delta.detach().clone()
            self._cached_key = cache_key
        return delta

    def _factorized_forward(self, x: torch.Tensor) -> torch.Tensor:
        factor_small, factor_large = self._factors()
        if self.in_features <= self.out_features:
            hidden = F.linear(x, factor_small)
            out = F.linear(hidden, factor_large)
        else:
            hidden = F.linear(x, factor_large.T)
            out = F.linear(hidden, factor_small.T)
        return out * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dropped = self.dropout(x).to(dtype=self.randlora_lambda.dtype)
        mode = self.forward_mode
        if mode == "auto":
            mode = "factorized" if self.training else "materialized"

        context = (
            _autocast_disabled(x.device.type)
            if self.preserve_float32
            else _null_context()
        )
        with context:
            if mode == "materialized":
                return F.linear(x_dropped, self.delta_weight())
            if mode == "factorized":
                return self._factorized_forward(x_dropped)
        raise RuntimeError(f"unknown forward_mode: {self.forward_mode}")


@contextmanager
def _null_context() -> Iterator[None]:
    yield


class RandLoRALinear(nn.Module):
    """RandLoRA wrapper for an ordinary ``nn.Linear`` layer."""

    def __init__(self, base_layer: nn.Linear, coefficients: RandLoRACoefficients) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("RandLoRALinear supports nn.Linear only")
        self.base_layer = base_layer
        self.coefficients = coefficients
        self.adapters_enabled = True
        self.merged = False
        self._pre_merge_weight: Optional[torch.Tensor] = None

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_layer.bias

    def train(self, mode: bool = True):
        if mode and self.merged:
            self.unmerge()
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        if self.adapters_enabled and not self.merged:
            out = out + self.coefficients(x).to(dtype=out.dtype)
        return out

    @torch.no_grad()
    def merge(self, *, safe: bool = True) -> None:
        if self.merged:
            return
        original = self.base_layer.weight.detach().clone()
        delta = self.coefficients.delta_weight().to(self.base_layer.weight)
        candidate = original + delta
        if safe and not torch.isfinite(candidate).all():
            raise FloatingPointError("non-finite values detected during RandLoRA merge")
        self.base_layer.weight.copy_(candidate)
        self._pre_merge_weight = original
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        if self._pre_merge_weight is None:
            raise RuntimeError("merged RandLoRA layer lost its pre-merge weight")
        self.base_layer.weight.copy_(self._pre_merge_weight.to(self.base_layer.weight))
        self._pre_merge_weight = None
        self.merged = False


class FusedQKVRandLoRA(nn.Module):
    """Official SAM1 fused-QKV wrapper that changes Q/V slices only."""

    _SLICE_INDEX = {"q": 0, "k": 1, "v": 2}

    def __init__(self, base_layer: nn.Linear, coefficients: Dict[str, RandLoRACoefficients]) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("SAM qkv must be nn.Linear")
        if base_layer.out_features != 3 * base_layer.in_features:
            raise ValueError(
                f"expected fused qkv shape [3d,d], got "
                f"[{base_layer.out_features},{base_layer.in_features}]"
            )
        if not coefficients or not set(coefficients).issubset({"q", "v"}):
            raise ValueError("coefficients must target q and/or v only")
        self.base_layer = base_layer
        self.coefficients = nn.ModuleDict(coefficients)
        self.embed_dim = base_layer.in_features
        self.adapters_enabled = True
        self.merged = False
        self._pre_merge_slices: Dict[str, torch.Tensor] = {}

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_layer.bias

    def train(self, mode: bool = True):
        if mode and self.merged:
            self.unmerge()
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        if not self.adapters_enabled or self.merged:
            return out
        chunks = list(out.split(self.embed_dim, dim=-1))
        for target, coeff in self.coefficients.items():
            idx = self._SLICE_INDEX[target]
            chunks[idx] = chunks[idx] + coeff(x).to(dtype=chunks[idx].dtype)
        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def merge(self, *, safe: bool = True) -> None:
        if self.merged:
            return
        candidate = self.base_layer.weight.detach().clone()
        originals: Dict[str, torch.Tensor] = {}
        for target, coeff in self.coefficients.items():
            idx = self._SLICE_INDEX[target]
            row = slice(idx * self.embed_dim, (idx + 1) * self.embed_dim)
            originals[target] = candidate[row].clone()
            candidate[row].add_(coeff.delta_weight().to(candidate))
        if safe and not torch.isfinite(candidate).all():
            raise FloatingPointError("non-finite values detected during fused QKV merge")
        self.base_layer.weight.copy_(candidate)
        self._pre_merge_slices = originals
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        if set(self._pre_merge_slices) != set(self.coefficients):
            raise RuntimeError("merged fused-QKV layer lost its pre-merge slices")
        for target, original in self._pre_merge_slices.items():
            idx = self._SLICE_INDEX[target]
            row = slice(idx * self.embed_dim, (idx + 1) * self.embed_dim)
            self.base_layer.weight[row].copy_(original.to(self.base_layer.weight))
        self._pre_merge_slices = {}
        self.merged = False
