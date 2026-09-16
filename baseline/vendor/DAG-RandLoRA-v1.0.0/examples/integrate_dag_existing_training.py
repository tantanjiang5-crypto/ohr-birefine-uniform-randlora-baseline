"""Minimal integration for an existing SAM1 training project.

This file deliberately does not recreate the user's trainer. It shows the exact
DAG-RandLoRA API boundary: Codex should reuse the already validated SAM resize,
PiCOPlus DETACH, QJ-2 DETACH, losses and evaluator, and replace only the encoder
PEFT injection/configuration shown here.
"""

from randlora_damage import (
    DAGAllocationPlan,
    RandLoRADamageConfig,
    config_from_allocation_plan,
    inject_randlora_damage_encoder,
)


def inject_from_plan(sam, allocation_json: str, *, baseline_params: int):
    plan = DAGAllocationPlan.load_json(allocation_json)
    if plan.target_total_params != baseline_params:
        raise RuntimeError(
            f"allocation budget {plan.target_total_params} != measured R0 budget {baseline_params}"
        )

    base = RandLoRADamageConfig(
        block_indices=tuple(range(4, 12)),
        qkv_targets=("q", "v"),
        extra_targets=(),
        rank=None,
        target_trainable_params=baseline_params,
        auto_match_budget=True,
        budget_warning_tolerance=0.01,
        alpha_multiplier=2.0,
        dropout=0.0,
        forward_mode="auto",
        adapter_dtype="float32",
        seed=42,
        save_basis=True,
        freeze_image_encoder=True,
    )
    cfg = config_from_allocation_plan(base, plan)
    report = inject_randlora_damage_encoder(sam, cfg)
    if report.budget_relative_error is not None and report.budget_relative_error > 0.01:
        raise RuntimeError(f"DAG allocation budget mismatch: {report.budget_relative_error:.2%}")
    return report
