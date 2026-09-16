# DAG-RandLoRA Method Design

## Core idea

DAG-RandLoRA keeps RandLoRA's frozen random full-rank-reachable basis construction, but removes the uniform-capacity assumption. A training-only calibration pass estimates exact fused-QKV weight gradients for low-occupancy damage instances and the same calibration distribution. Capacity is then redistributed under the same total number of trainable RandLoRA scaling coefficients.

### R1 — OC-RandLoRA

Block-level allocation only. Q and V share a basis rank within a block. Blocks with stronger low-occupancy-specific segmentation gradients receive smaller RandLoRA basis rank (more scaling coefficients) and low-priority blocks receive larger basis rank. Total trainable adapter parameters remain within the configured tolerance of Uniform RandLoRA.

### R2 — DAG-RandLoRA

For every `block:q` and `block:v`, compute the entropy effective rank of the hard-instance full weight gradient as Gradient Intrinsic Dimensionality (GID), and combine it with hard/general gradient-norm specificity:

```text
priority = z(log(GID)) + 0.5 * z(log(||G_hard||F / (||G_general||F + eps)))
```

A fixed-budget dynamic-programming allocator assigns target-specific RandLoRA basis ranks. Q and V may differ within one block. `alpha/r` is held constant so rank allocation does not silently change adapter scaling.

## Novelty boundary

The contribution is not generic adaptive rank: AdaLoRA-style allocation already exists and GaRA-SAM dynamically gates rank components in SAM. The contribution is also not generic gradient-informed basis initialization: GiVA (AISTATS 2026) already derives vector-adaptation bases from downstream gradients. DAG-RandLoRA instead keeps RandLoRA's random bases frozen and changes only *where full-rank adaptation capacity is spent*, using vehicle-damage hard-instance gradient structure under a fixed budget.

## Experimental hygiene

Profiles must be derived from the training split only. The validated Uniform RandLoRA checkpoint may be used as a read-only reference model for profiling, but R1/R2 formal runs must restart from the same formal initialization used by R0 rather than continuing from the R0 best checkpoint.
