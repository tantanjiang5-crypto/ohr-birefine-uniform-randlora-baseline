from __future__ import annotations

import json

import torch

from randlora_damage import (
    FusedQKVWeightGradientProfiler,
    RandLoRADamageConfig,
    build_r2_dag_plan,
    inject_randlora_damage_encoder,
    randlora_trainable_params,
)
from tests.mock_sam import MockSAM


def main() -> None:
    torch.manual_seed(7)
    model = MockSAM(dim=16, depth=12)
    model.image_encoder.requires_grad_(False)
    with FusedQKVWeightGradientProfiler(model, block_indices=(4, 5)) as profiler:
        x = torch.randn(2, 6, 16)
        y = model.image_encoder(x)
        loss = y.square().mean()
        loss.backward()
        profiler.update_after_backward(sample_weight=2)
        hard = profiler.matrices()
    general = {k: v * (0.5 if k == "4:q" else 1.0) for k, v in hard.items()}
    baseline_rank = 8
    target_params = len(hard) * randlora_trainable_params(16, 16, baseline_rank)
    plan = build_r2_dag_plan(
        hard,
        general_gradients=general,
        baseline_rank=baseline_rank,
        target_total_params=target_params,
        rank_candidates=(4, 8, 16),
        budget_tolerance=0.20,
    )
    cfg = RandLoRADamageConfig(
        block_indices=(4, 5),
        qkv_targets=("q", "v"),
        auto_match_budget=False,
        rank=baseline_rank,
        target_trainable_params=None,
        target_rank_pattern=plan.target_rank_pattern,
    )
    fresh = MockSAM(dim=16, depth=12)
    report = inject_randlora_damage_encoder(fresh, cfg)
    output = {
        "status": "PASS",
        "profile_targets": sorted(hard),
        "r2_priority": plan.scores,
        "r2_target_rank_pattern": plan.target_rank_pattern,
        "r2_budget_error": plan.budget_relative_error,
        "injected_target_ranks": report.target_ranks,
        "actual_adapter_params": report.actual_adapter_params,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
