from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from randlora_damage import (  # noqa: E402
    RandLoRADamageConfig,
    audit_trainable_image_encoder,
    inject_randlora_damage_encoder,
    lora_trainable_params,
)
from tests.mock_sam import MockSAM  # noqa: E402


def main() -> None:
    # Real SAM ViT-B budget calculation.
    lora_r4_all12 = 12 * 2 * lora_trainable_params(768, 768, 4)
    config = RandLoRADamageConfig()
    mock = MockSAM(dim=16)
    mock_config = RandLoRADamageConfig(
        block_indices=tuple(range(4, 12)),
        target_trainable_params=1536,
        rank=None,
        auto_match_budget=True,
    )
    report = inject_randlora_damage_encoder(mock, mock_config)
    audit = audit_trainable_image_encoder(mock)
    x = torch.randn(2, 8, 16)
    y = mock(x)
    loss = y[0].square().mean() + y[1].square().mean()
    loss.backward()
    payload = {
        "status": "PASS" if audit["pass"] else "FAIL",
        "real_sam_vit_b_reference_lora_r4_qv_all12": lora_r4_all12,
        "default_requested_randlora_budget": config.target_trainable_params,
        "mock_report": {
            "default_rank": report.default_rank,
            "actual_adapter_params": report.actual_adapter_params,
            "wrapped_modules": list(report.wrapped_modules),
        },
        "trainable_audit": audit,
        "finite_loss": bool(torch.isfinite(loss)),
    }
    print(json.dumps(payload, indent=2, default=str))
    if payload["status"] != "PASS" or not payload["finite_loss"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
