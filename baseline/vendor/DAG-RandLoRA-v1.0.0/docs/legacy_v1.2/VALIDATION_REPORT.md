# Validation report — RandLoRA-Damage Encoder v1.2.0

Validation environment: Python 3.13.5, PyTorch 2.10.0+cpu.

## Input integrity

- Uploaded v1.1.0 ZIP SHA-256 matched supplied checksum: PASS.
- Uploaded v1.1.0 source compileall: PASS.
- Uploaded v1.1.0 original tests: 21/21 PASS.

## v1.2.0 validation

- Source compileall: PASS.
- Pytest: **37/37 PASS**.
- Paper formula vs explicit per-basis sum: PASS.
- Custom factor-scaling backward vs ordinary autograd: PASS.
- Zero-lambda initial no-op: PASS.
- First-step lambda gradient finite/nonzero and initial gamma gradient zero as mathematically expected: PASS.
- Fused QKV Q/V update with K bitwise unchanged: PASS.
- Factorized/materialized/auto output equivalence: PASS.
- Eval no-grad cache invalidation after parameter/state/device/dtype changes: PASS.
- Eval no-grad cache does not detach later eval-mode backward: PASS.
- Base image encoder frozen and adapter gradients finite: PASS.
- FP32 adapter survives parent `.half()` and produces finite output: PASS.
- CPU bfloat16 autocast smoke: PASS.
- Exact historical pre-merge restoration, including low-precision bitwise round trip: PASS.
- `train(True)` automatically unmerges: PASS.
- Merge transaction rollback and merge-and-unload: PASS.
- Deepcopy shared-basis links: PASS.
- Whole-model pickle no longer fails from weakref: PASS (state_dict remains recommended).
- Adapter checkpoint v3 atomic save and immutable state snapshot: PASS.
- State digest corruption detection: PASS.
- Strict loader rejects missing keys after digest recomputation: PASS.
- Strict loader rejects ordinary SAM base weights after digest recomputation: PASS.
- Full-SAM ↔ Image-Encoder key portability: PASS.
- Full config, manifest, resolved-rank and shape mismatch checks: PASS.
- Basis fingerprint checks: PASS.
- Actual v1.0 checkpoint generated with the v1.0 wheel and loaded by v1.2 strict mode: PASS.
- Actual v1.1 checkpoint generated with uploaded v1.1 source and loaded by v1.2 strict mode: PASS.
- Optimizer generator preservation: PASS.
- Existing optimizer public `add_param_group` integration: PASS.
- Occupancy rank allocation selects non-uniform priority solution within budget tolerance: PASS.
- Allocation combination guard and score validation: PASS.
- Float-logit mask diagnostics and threshold/shape validation: PASS.
- Experiment gate NaN/range validation: PASS.
- Transactional injection on invalid later target: PASS.
- Duplicate targets and ambiguous rank/budget rejection: PASS.
- SAM1 ViT-B fused-QKV shape (`768→2304`): PASS.
- Parameter budget target 147,456; rank 70; actual 147,488; difference +32 (+0.0217%): PASS.
- Scaling semantics audited: default `alpha_multiplier=2.0` resolves to constant `alpha/r=2`; paper-style visual `10/r` remains an explicit configuration (`alpha=10`) rather than an undocumented default change: PASS.
- `torch.compile(..., backend="eager")` forward/backward smoke: PASS.
- Package audit script: PASS.

## Build validation

- Wheel build with local installed build backend: PASS.
- Wheel filename/version: `randlora_damage_encoder-1.2.0-py3-none-any.whl`.
- Isolated target-directory wheel installation: PASS.
- Isolated import reports `__version__ == "1.2.0"`: PASS.
- Installed-wheel injection and finite forward smoke: PASS.
- Wheel compressed-data integrity (`unzip -t`): PASS.
- Source ZIP internal SHA-256 manifest: PASS.
- Final ZIP/wheel checksums are supplied alongside the artifacts.

## Environment limitation

Meta's official `segment_anything` package, a SAM ViT-B checkpoint and CUDA GPU were not available. The official source layout was checked and the package includes a stronger real-environment script, but this report does **not** claim an official 1024×1024 checkpoint forward/backward, RTX3090 AMP profile, TQ-MOTP/QJ-2 trainer integration or 20-epoch vehicle-damage result.
