"""Official SAM1 ViT-B structural/full-encoder smoke.

Run this inside the real project environment where Meta's ``segment_anything``
is installed. A checkpoint is optional for structural validation, but pass the
actual checkpoint before a formal experiment.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from randlora_damage import (  # noqa: E402
    RandLoRADamageConfig,
    audit_trainable_image_encoder,
    inject_randlora_damage_encoder,
)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--skip-full-encoder", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "segment_anything is not installed. Run inside the existing Meta SAM1 project environment."
        ) from exc
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    sam = sam_model_registry["vit_b"](
        checkpoint=str(args.checkpoint) if args.checkpoint is not None else None
    ).to(device)
    sam.eval()

    # Full-encoder zero-update equivalence before any optimizer step.
    image = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    baseline = None
    if not args.skip_full_encoder:
        with torch.no_grad(), autocast_context(device, args.amp):
            baseline = sam.image_encoder(image).float().cpu()

    report = inject_randlora_damage_encoder(sam, RandLoRADamageConfig())
    assert report.default_rank == 70, report
    assert report.actual_adapter_params == 147_488, report
    audit = audit_trainable_image_encoder(sam)
    assert audit["pass"], audit

    wrapper = sam.image_encoder.blocks[4].attn.qkv
    token = torch.randn(1, 2, 2, 768, device=device)
    with torch.no_grad():
        original_qkv = wrapper.base_layer(token)
        adapted_qkv = wrapper(token)
    torch.testing.assert_close(adapted_qkv, original_qkv, rtol=0, atol=0)

    if not args.skip_full_encoder:
        with torch.no_grad(), autocast_context(device, args.amp):
            injected = sam.image_encoder(image).float().cpu()
        torch.testing.assert_close(injected, baseline, rtol=0, atol=0)

        # First step: lambda must receive a finite gradient. Gamma is expected
        # to be zero initially because lambda is initialized to zero for no-op.
        sam.train()
        with autocast_context(device, args.amp):
            loss = sam.image_encoder(image).float().square().mean()
        loss.backward()
        lambda_grads = []
        gamma_grads = []
        for name, parameter in sam.image_encoder.named_parameters():
            if "randlora_lambda" in name:
                lambda_grads.append(parameter.grad)
            elif "randlora_gamma" in name:
                gamma_grads.append(parameter.grad)
        assert lambda_grads and all(g is not None and torch.isfinite(g).all() for g in lambda_grads)
        assert sum(float(g.abs().sum()) for g in lambda_grads) > 0
        assert gamma_grads and all(g is not None and torch.isfinite(g).all() for g in gamma_grads)

    print(
        {
            "status": "PASS",
            "checkpoint_loaded": args.checkpoint is not None,
            "device": str(device),
            "full_encoder_checked": not args.skip_full_encoder,
            "default_rank": report.default_rank,
            "actual_adapter_params": report.actual_adapter_params,
            "trainable_audit": audit,
        }
    )


if __name__ == "__main__":
    main()
