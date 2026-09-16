# DAG-RandLoRA / RandLoRA-Damage Encoder v1.3.0

**Damage-Aware Gradient RandLoRA for official SAM1 ViT image encoders.**

This package extends the audited RandLoRA-Damage Encoder v1.2.0 with a fixed-budget, damage-aware capacity allocation pipeline. It keeps the original RandLoRA random bases frozen and full-rank reachable, while allowing different basis ranks across SAM blocks and, in R2, independently across fused-QKV Q and V slices.

## What is new in v1.3

- Exact fused-QKV **full weight gradient** profiling for Q/V without updating the encoder.
- Training-only gradient profile save/load utilities.
- R1 **OC-RandLoRA** block-level low-occupancy allocation.
- R2 **DAG-RandLoRA** Q/V-level GID + hard-specificity allocation.
- Target-specific rank config keys such as `"8:q"` and `"8:v"`.
- Dynamic-programming fixed-budget allocator for 16 SAM Q/V targets.
- Existing v1.2 cache, merge/unmerge, checkpoint, AMP, optimizer and diagnostics behavior retained.

## Intended experiment

```text
R0  Uniform RandLoRA (already validated)
 ↓
P   training-only hard/general Q/V gradient calibration
 ↓
R1  occupancy-conditioned block allocation, Q/V share rank
 ↓
R2  Q/V-specific GID + hard-specificity allocation
```

Do not use validation/test data to generate the allocation.

## Core API

```python
from randlora_damage import (
    collect_qv_gradient_profile,
    save_gradient_profile,
    build_r1_oc_plan,
    build_r2_dag_plan,
    DAGAllocationPlan,
    config_from_allocation_plan,
    inject_randlora_damage_encoder,
)
```

### Gradient profiling

The provided generic runner expects a project callback that returns the segmentation probe loss and selected instance count:

```python
def hard_loss_fn(model, batch):
    # Use the existing SAM44 forward/loss code.
    # Select GT instances with occupancy <= 0.10.
    # Return the SAME encoder mask loss used in formal training, reduced by mean.
    return scalar_loss, selected_instance_count

hard_gradients, report = collect_qv_gradient_profile(
    reference_model,
    calibration_loader,
    hard_loss_fn,
    block_indices=tuple(range(4, 12)),
    max_instances=1024,
)
save_gradient_profile("profiles/hard_occ_le_010.pt", hard_gradients)
```

The profiler temporarily enables gradients on the frozen fused-QKV base weights only. It does not update them and restores every original `requires_grad` flag afterward.

### R1 plan

```python
plan = build_r1_oc_plan(
    hard_gradients,
    general_gradients=general_gradients,
    baseline_rank=R0_RESOLVED_RANK,
    target_total_params=R0_ACTUAL_ADAPTER_PARAMS,
    rank_candidates=(64, 70, 80),  # when R0 rank is ~70
    budget_tolerance=0.01,
)
plan.save_json("allocations/R1_OC_allocation.json")
```

### R2 plan

```python
plan = build_r2_dag_plan(
    hard_gradients,
    general_gradients=general_gradients,
    baseline_rank=R0_RESOLVED_RANK,
    target_total_params=R0_ACTUAL_ADAPTER_PARAMS,
    rank_candidates=(56, 64, 70, 80, 96),
    budget_tolerance=0.01,
    gid_weight=1.0,
    specificity_weight=0.5,
    classification_penalty=0.0,
)
plan.save_json("allocations/R2_DAG_allocation.json")
```

`classification_penalty=0` is intentional for the first formal experiment because the current classification head is DETACH. Classification-context profiling should be a later ablation if needed.

### Inject an allocation into a fresh formal run

```python
plan = DAGAllocationPlan.load_json("allocations/R2_DAG_allocation.json")
base_cfg = RandLoRADamageConfig(
    block_indices=tuple(range(4, 12)),
    qkv_targets=("q", "v"),
    extra_targets=(),
    rank=None,
    target_trainable_params=R0_ACTUAL_ADAPTER_PARAMS,
    auto_match_budget=True,
    budget_warning_tolerance=0.01,
    alpha_multiplier=2.0,  # must match R0 effective scaling protocol
    forward_mode="auto",
    adapter_dtype="float32",
    seed=R0_RANDLORA_SEED,
    save_basis=True,
    freeze_image_encoder=True,
)
report = inject_randlora_damage_encoder(
    sam,
    config_from_allocation_plan(base_cfg, plan),
)
```

The formal R1/R2 model must start from the same initialization as R0. Do not continue training from R0 best; that checkpoint is only a read-only gradient reference.

## CLI allocation builder

```bash
PYTHONPATH=. python scripts/build_dag_allocation.py \
  --stage r2 \
  --hard-profile profiles/hard_occ_le_010.pt \
  --general-profile profiles/general_same_calibration.pt \
  --baseline-rank 70 \
  --target-total-params 147488 \
  --budget-tolerance 0.01 \
  --output allocations/R2_DAG_allocation.json
```

## Validation

```bash
PYTHONPATH=. pytest -q
python -m compileall -q randlora_damage tests scripts examples
```

The package uses only PyTorch. The current build environment does not include the user's Meta SAM checkpoint/trainer/CUDA setup, so full `/workspace/sam44` integration must still be smoke-tested there before a 20-epoch run.

## Research-positioning warning

Do **not** claim generic adaptive rank or generic gradient-informed basis construction as novel. GaRA-SAM already performs input-aware gated-rank adaptation in SAM, and GiVA (AISTATS 2026) derives vector-adaptation bases from downstream gradients. The intended contribution here is specifically **damage-hardness-conditioned, fixed-budget capacity allocation inside full-rank-reachable RandLoRA**, while keeping RandLoRA's random bases frozen.
