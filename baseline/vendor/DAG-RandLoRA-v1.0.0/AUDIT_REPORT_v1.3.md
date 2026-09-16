# DAG-RandLoRA v1.0.0 / RandLoRA-Damage Encoder v1.3.0 Audit

## Result

Code audit and synthetic integration checks: **PASS**.

This release is based on the previously audited v1.2.0 implementation and adds only the mechanisms required for the proposed damage-aware allocation experiment. Existing v1.2 behavior is preserved when `target_rank_pattern={}`.

## New implementation audited

1. **Target-specific Q/V rank injection**
   - `target_rank_pattern` supports keys such as `8:q` and `8:v`.
   - Q and V may use different RandLoRA basis ranks in the same fused SAM1 QKV layer.
   - K remains unchanged by the adapter.
   - Per-target rank overrides block rank, which overrides the uniform resolved rank.

2. **Exact gradient profiler**
   - Temporarily enables gradients on selected frozen fused-QKV base weights only.
   - Uses PyTorch's exact `weight.grad`, sliced into Q and V, rather than an activation-gradient proxy.
   - Never performs an optimizer step and restores all original `requires_grad` states.
   - Supports weighted aggregation when batch loss is a mean over a varying number of selected instances.

3. **GID estimator**
   - Uses entropy effective rank of the Q/V full gradient matrix.
   - Rank-1 and full-rank mathematical regression tests pass.

4. **R1 fixed-budget allocator**
   - Occupancy-conditioned block allocation.
   - Q/V share one rank per block.
   - Designed for a conservative first experiment.

5. **R2 fixed-budget allocator**
   - Q/V-specific priority from GID and hard/general gradient specificity.
   - Dynamic programming replaces exponential Cartesian search for 16 SAM Q/V targets.
   - The objective reallocates trainable RandLoRA scaling capacity while keeping total parameters within the configured tolerance when a feasible candidate combination exists.

6. **Scientific protocol safeguards**
   - Profiles can be generated without updating model parameters.
   - The execution plan explicitly forbids validation/test-based rank selection.
   - R1/R2 are required to restart from the same formal initialization as R0.
   - Classification penalty is disabled in the first R2 experiment because the formal classification head is DETACH.

## Novelty boundary after 2026 literature check

The package intentionally does **not** implement generic input-gated rank adaptation or claim generic gradient-informed basis construction as new. GaRA-SAM already applies input-aware gated-rank adaptation to SAM, and GiVA (AISTATS 2026) derives vector-adaptation bases from downstream gradients. The proposed experimental contribution is narrower: training-hardness-conditioned allocation of capacity *inside full-rank-reachable RandLoRA* while retaining frozen random bases.

## Remaining external validation

The build environment does not contain the user's `/workspace/sam44` trainer, Meta SAM ViT-B checkpoint or CUDA RTX3090 runtime. Therefore the following must still be checked by Codex in the real project before formal training:

- actual Uniform RandLoRA R0 parameter count and resolved rank;
- real PiCOPlus/QJ-2 DETACH implementation and optimizer groups;
- exact per-instance mask loss access for hard/general calibration;
- official SAM resize/box/postprocess chain;
- 100–200 CUDA/AMP optimizer-step smoke;
- peak VRAM, step time and CPU/GPU utilization;
- full validation metrics after 20 epochs.
