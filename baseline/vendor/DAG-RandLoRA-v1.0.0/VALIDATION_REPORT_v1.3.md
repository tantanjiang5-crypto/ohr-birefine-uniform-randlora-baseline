# Validation Report — DAG-RandLoRA v1.0.0 / package v1.3.0

Validation environment:

```text
Python 3.13.5
PyTorch 2.10.0+cpu
```

## Passed

- Source `compileall`: PASS.
- Pytest: **45/45 PASS**.
- All inherited v1.2 regression tests: PASS.
- Uniform v1.2-compatible injection path: PASS.
- Q/V target-specific rank injection: PASS.
- Constant `alpha/r` scaling with different Q/V ranks: PASS.
- Exact fused-QKV gradient extraction vs PyTorch `weight.grad`: PASS.
- Profiler restores model mode and all `requires_grad` flags: PASS.
- Entropy effective-rank/GID rank-1 case: PASS.
- Entropy effective-rank/GID full-rank identity case: PASS.
- Dynamic-programming target allocation: PASS.
- High-priority target receives at least as much RandLoRA trainable capacity in regression test: PASS.
- R2 GID + hard-specificity score ordering: PASS.
- Allocation plan JSON serialization: covered by API smoke.
- Target-specific config serialization: PASS.
- Target-specific strict adapter checkpoint round trip: PASS.
- Existing cache/merge/unmerge/AMP/checkpoint/optimizer tests: PASS.
- Synthetic DAG package audit: PASS.
- Wheel build with local no-isolation toolchain: PASS.
- Wheel isolated target install/import: PASS (`randlora_damage.__version__ == 1.3.0`).
- Real ViT-B parameter-cost function smoke for 16 Q/V targets and rank candidates: PASS; fixed-budget allocator found a pattern within 1% in synthetic priority testing.

## Environment limitation

No claim is made that the real `/workspace/sam44` integration has already passed. This environment has no Meta SAM checkpoint, no user trainer and no CUDA GPU. The included `CODEX_EXECUTION_PLAN.md` requires a real-project smoke before a formal run.
