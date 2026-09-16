#!/usr/bin/env python3
"""Create a host-local config while preserving the validated experiment protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--common-init", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/uniform_randlora_seed2026.local.json"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source = repo / "configs" / "baseline_original_host.json"
    config = json.loads(source.read_text())
    dataset = args.dataset_root.resolve()
    config["paths"] = {
        "project_root": str(repo),
        "model_file": str(repo / "baseline" / "code" / "TQMOTP_QJ2_GRAD_SCALE_DAG.py"),
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "common_init": str(args.common_init.resolve()),
        "train_json": str(dataset / "annotations" / "instances_train2017.json"),
        "train_images": str(dataset / "train2017"),
        "val_json": str(dataset / "annotations" / "instances_val2017.json"),
        "val_images": str(dataset / "val2017"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
