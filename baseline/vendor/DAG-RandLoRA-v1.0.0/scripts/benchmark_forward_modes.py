"""Benchmark factorized/materialized/auto without assuming a universal winner."""
from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from randlora_damage.core import RandLoRABasisBank, RandLoRACoefficients


def amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(mode, device, steps, warmup, tokens, dim, rank, amp):
    bank = RandLoRABasisBank(
        rank=rank,
        max_dim=dim,
        min_dim=dim,
        seed=42,
        preserve_float32=True,
    ).to(device)
    coeff = RandLoRACoefficients(
        in_features=dim,
        out_features=dim,
        rank=rank,
        alpha=2 * rank,
        dropout=0.0,
        forward_mode=mode,
        cache_eval_delta=True,
        basis_bank=bank,
        preserve_float32=True,
    ).to(device)
    with torch.no_grad():
        coeff.randlora_lambda.normal_(std=0.01)
    x = torch.randn(1, tokens, dim, device=device)
    coeff.train()

    for _ in range(warmup):
        with amp_context(device, amp):
            coeff(x).square().mean().backward()
        coeff.zero_grad(set_to_none=True)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        with amp_context(device, amp):
            loss = coeff(x).square().mean()
        loss.backward()
        coeff.zero_grad(set_to_none=True)
        synchronize(device)
        timings.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "mode": mode,
        "mean_seconds_per_step": statistics.mean(timings),
        "median_seconds_per_step": statistics.median(timings),
        "peak_bytes": peak,
        "finite": bool(torch.isfinite(loss)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--rank", type=int, default=70)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.tokens, args.dim, args.rank) <= 0:
        raise SystemExit("steps, warmup, tokens, dim and rank must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    for mode in ("factorized", "materialized", "auto"):
        print(bench(mode, device, args.steps, args.warmup, args.tokens, args.dim, args.rank, args.amp))


if __name__ == "__main__":
    main()
