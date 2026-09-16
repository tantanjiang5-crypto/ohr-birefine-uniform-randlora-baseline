from __future__ import annotations

import argparse
from pathlib import Path

from randlora_damage import (
    build_r1_oc_plan,
    build_r2_dag_plan,
    load_gradient_profile,
)


def parse_ranks(text: str):
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-budget R1/R2 DAG-RandLoRA allocation")
    parser.add_argument("--stage", choices=("r1", "r2"), required=True)
    parser.add_argument("--hard-profile", required=True)
    parser.add_argument("--general-profile", required=True)
    parser.add_argument("--classification-profile")
    parser.add_argument("--baseline-rank", type=int, required=True)
    parser.add_argument("--target-total-params", type=int, required=True)
    parser.add_argument("--budget-tolerance", type=float, default=0.01)
    parser.add_argument("--rank-candidates", default=None)
    parser.add_argument("--gid-weight", type=float, default=1.0)
    parser.add_argument("--specificity-weight", type=float, default=0.5)
    parser.add_argument("--classification-penalty", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hard, hard_meta = load_gradient_profile(args.hard_profile)
    general, general_meta = load_gradient_profile(args.general_profile)
    classification = None
    if args.classification_profile:
        classification, _ = load_gradient_profile(args.classification_profile)
    ranks = parse_ranks(args.rank_candidates or ("64,70,80" if args.stage == "r1" else "56,64,70,80,96"))
    metadata = {
        "hard_profile": str(Path(args.hard_profile).resolve()),
        "general_profile": str(Path(args.general_profile).resolve()),
        "hard_profile_metadata": hard_meta,
        "general_profile_metadata": general_meta,
    }

    if args.stage == "r1":
        plan = build_r1_oc_plan(
            hard,
            general_gradients=general,
            baseline_rank=args.baseline_rank,
            target_total_params=args.target_total_params,
            rank_candidates=ranks,
            budget_tolerance=args.budget_tolerance,
            metadata=metadata,
        )
    else:
        plan = build_r2_dag_plan(
            hard,
            general_gradients=general,
            baseline_rank=args.baseline_rank,
            target_total_params=args.target_total_params,
            rank_candidates=ranks,
            budget_tolerance=args.budget_tolerance,
            gid_weight=args.gid_weight,
            specificity_weight=args.specificity_weight,
            classification_gradients=classification,
            classification_penalty=args.classification_penalty,
            temperature=args.temperature,
            metadata=metadata,
        )

    out = plan.save_json(args.output)
    print(f"stage={plan.stage}")
    print(f"target_total_params={plan.target_total_params}")
    print(f"actual_total_params={plan.actual_total_params}")
    print(f"budget_relative_error={plan.budget_relative_error:.6f}")
    print(f"block_rank_pattern={plan.block_rank_pattern}")
    print(f"target_rank_pattern={plan.target_rank_pattern}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
