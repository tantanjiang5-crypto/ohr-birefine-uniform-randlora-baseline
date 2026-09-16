#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

EXPECTED = {
    "sam_checkpoint": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "common_init": "40467b766d2a98d6937a17029b73507a7264fc54fa9b80bf28442b29707f03e5",
    "train_json": "b6801334b74f56280b5099b0fadb1a469672b1042a184502ae7b1c333cdf2187",
    "val_json": "3da039ee7d4d604b27dc75ffd889b500edf07b827e43eb1cec7eca68b6a8fa38",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    failures: list[str] = []
    for key, expected in EXPECTED.items():
        path = Path(config["paths"][key])
        if not path.is_file():
            failures.append(f"missing {key}: {path}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"SHA256 mismatch {key}: {actual} != {expected}")
    for key in ("train_images", "val_images"):
        path = Path(config["paths"][key])
        if not path.is_dir():
            failures.append(f"missing directory {key}: {path}")
    for module in ("torch", "torchvision", "numpy", "PIL", "pycocotools"):
        try:
            importlib.import_module(module)
        except Exception as error:
            failures.append(f"cannot import {module}: {error}")
    if failures:
        print("HOST VERIFICATION FAILED", file=sys.stderr)
        print("\n".join(f"- {item}" for item in failures), file=sys.stderr)
        raise SystemExit(1)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("HOST VERIFICATION FAILED: CUDA unavailable")
    print(json.dumps({"status": "PASS", "torch": torch.__version__,
                      "cuda_devices": torch.cuda.device_count()}, indent=2))


if __name__ == "__main__":
    main()
