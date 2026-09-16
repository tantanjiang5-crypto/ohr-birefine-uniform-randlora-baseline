from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from randlora_damage import (  # noqa: E402
    find_rank_for_budget,
    lora_trainable_params,
    randlora_trainable_params,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--branches", type=int, default=2, help="Q+V = 2")
    parser.add_argument("--target", type=int, default=147456)
    parser.add_argument("--lora-rank", type=int, default=4)
    args = parser.parse_args()

    shapes = [(args.dim, args.dim)] * (args.blocks * args.branches)
    rank, actual = find_rank_for_budget(shapes, args.target)
    payload = {
        "randlora": {
            "matched_basis_rank": rank,
            "num_bases": math.ceil(args.dim / rank),
            "actual_trainable_params": actual,
            "target_trainable_params": args.target,
            "absolute_difference": actual - args.target,
        },
        "reference_lora": {
            "rank": args.lora_rank,
            "same_selected_layers_params": args.blocks
            * args.branches
            * lora_trainable_params(args.dim, args.dim, args.lora_rank),
        },
        "paper_style_candidates": {
            str(r): args.blocks
            * args.branches
            * randlora_trainable_params(args.dim, args.dim, r)
            for r in (8, 16)
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
