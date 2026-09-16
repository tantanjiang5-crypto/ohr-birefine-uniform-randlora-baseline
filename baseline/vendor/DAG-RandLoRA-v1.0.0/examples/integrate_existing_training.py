"""Integration template for SAM1 + TQ-MOTP + QJ-2.

Insert after loading the original SAM checkpoint and before constructing the
optimizer. Keep the baseline losses, sampler, evaluator and head settings.
"""

from randlora_damage import (
    RandLoRADamageConfig,
    append_randlora_param_group,
    inject_randlora_damage_encoder,
    save_adapter_checkpoint,
)


def attach_randlora(model, baseline_head_param_groups, actual_lora_budget: int):
    config = RandLoRADamageConfig(
        block_indices=tuple(range(4, 12)),
        qkv_targets=("q", "v"),
        extra_targets=(),
        target_trainable_params=actual_lora_budget,
        auto_match_budget=True,
        rank=None,
        alpha_multiplier=2.0,
        dropout=0.0,
        forward_mode="auto",       # factorized train, cached materialized eval
        adapter_dtype="float32",  # stable under AMP/fp16 base weights
        seed=42,
        save_basis=True,
        freeze_image_encoder=True,
    )
    report = inject_randlora_damage_encoder(model, config)
    print(report)

    # Preserve all existing TQ-MOTP/QJ-2 no-decay and LR groups exactly.
    param_groups = [dict(group) for group in baseline_head_param_groups]
    append_randlora_param_group(
        param_groups,
        model,
        encoder_lr=1e-4,
        encoder_weight_decay=0.0,
    )
    return param_groups, report


# Baseline loss remains unchanged:
# loss = 1.0*loss_cls + 5.0*loss_mask_bce + 5.0*loss_dice + loss_qj2
#
# save_adapter_checkpoint(
#     model,
#     output_dir / "randlora_adapter_best.pt",
#     metadata={"epoch": epoch, "d_ap75": d_ap75},
# )
