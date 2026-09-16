# OHR-BiRefine / Uniform RandLoRA baseline transfer

This repository transfers the validated **coco_bj823 Uniform RandLoRA baseline** to another host so OHR-BiRefine can be added under the same protocol. It contains the training code, vendored lightweight dependencies, exact source configuration, validation references, path configurator, host verifier, and launchers.

It does not contain OHR-BiRefine source because the original OHR-BiRefine ZIP was not present on the source host. Add that module only after reviewing the checklist in `reference/OHR_BIREFINE_PRELIMINARY_REVIEW.md`.

## Validated baseline

- Model: SAM1 ViT-B, oracle GT-box prompts, one mask per GT instance, 59 classes.
- Adapter: Uniform RandLoRA on image-encoder blocks 4–11, fused Q/V only, matched to 147456 trainable LoRA parameters, seed 2026.
- Classification: TQ-MOTP with gradient scale 0.25.
- Quality: QJ-2 with detached quality path.
- Optimizer: AdamW; RandLoRA `1e-4`, mask decoder `5e-5`, TQ/prototypes `3e-4`, QJ-2 `1e-4`, weight decay `1e-4` where applicable.
- Training: 20 epochs, batch 2, accumulation 4 (effective batch 8), warmup 5%, gradient clip 1.0, AMP, validation every epoch.
- Loss: mask BCE 5.0 + Dice 5.0 + QJ-2 0.1 + TQ-MOTP 1.0; MQ-RS disabled.
- Primary checkpoint metric: `D_legacy_AP`.
- Historical result: best epoch 18, `D_legacy_AP=0.6761107389269907`, mask mIoU `0.9133142872843776`, Boundary F1 `0.7478917743228316`.

The original and resolved settings are in `configs/baseline_original_host.json` and `reference/RESOLVED_CONFIG.json`.

## Second-host setup

Use Python 3.10. Install PyTorch for the target CUDA driver first. The source environment used torch 2.1.0 and torchvision 0.16.0.

```bash
git clone https://github.com/tantanjiang5-crypto/ohr-birefine-uniform-randlora-baseline.git
cd ohr-birefine-uniform-randlora-baseline

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Choose the correct CUDA wheel index for the new host:
python -m pip install torch==2.1.0 torchvision==0.16.0
python -m pip install -r requirements.txt
```

Prepare coco_bj823 with this layout:

```text
coco_bj823/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

Install Git LFS before cloning, or run `git lfs pull` after installing it. Materialize the exact common initialization from its LFS parts, copy the SAM checkpoint listed in `artifacts/README.md`, then generate the host-local configuration:

```bash
git lfs pull
python scripts/materialize_artifacts.py

python scripts/configure.py \
  --dataset-root /data/coco_bj823 \
  --sam-checkpoint /models/sam_vit_b_01ec64.pth \
  --common-init "$PWD/artifacts/seed2026_59cls.pth" \
  --output configs/uniform_randlora_seed2026.local.json

python scripts/verify_host.py --config configs/uniform_randlora_seed2026.local.json
```

`verify_host.py` checks the exact SAM checkpoint, common initialization and both annotation files by SHA256. A mismatch means the run is not the same baseline protocol.

## Run

Run a two-step, 16-validation-image smoke test first:

```bash
bash scripts/run_smoke.sh \
  configs/uniform_randlora_seed2026.local.json \
  outputs/uniform_smoke \
  0
```

After checking the smoke output, run the complete baseline:

```bash
bash scripts/run_formal.sh \
  configs/uniform_randlora_seed2026.local.json \
  outputs/uniform_formal_seed2026 \
  0
```

Both scripts expose one physical GPU and present it to the trainer as `cuda:0`, which is required by the audited runner.

## Add OHR-BiRefine fairly

Keep the dataset, split, common initialization, prompt protocol, RandLoRA selection, seed, batch schedule, optimizer budget and evaluation unchanged. Add the module as an explicit model component and optimizer group. Perform a real-data smoke test and verify:

- no GT mask or GT occupancy enters inference-time routing;
- the 8-pixel far-FP definition remains in 256×256 mask coordinates;
- ROI ordering matches the flattened SAM instance ordering;
- the refined mask is the mask consumed by segmentation loss and evaluation;
- all trainable OHR parameters occur exactly once in the optimizer;
- AMP gradients are finite and checkpoint strict round-trip succeeds;
- Boundary F1, far-FP, foreground completeness/recall, empty-mask rate and AP are reported alongside the baseline.

Train the primary OHR comparison from the same `seed2026_59cls.pth`. Fine-tuning from the historical baseline `best.pth` is a separate experiment and cannot be reported as an equal-budget comparison.

## Repository contents

```text
baseline/code/          portable baseline training and model code
baseline/vendor/        SAM, classification heads, TQ-MOTP and RandLoRA sources
configs/                exact source-host config
scripts/                configure, verify, smoke and formal launchers
reference/              audits, resolved config and best validation metrics
artifacts/README.md      large-file checksums and transfer requirements
```

No dataset, credential, image, or machine secret is stored in this repository. The exact common initialization is stored as Git LFS parts and deterministically reconstructed; the SAM checkpoint and historical trained checkpoint are not included.
