#!/usr/bin/env python3
"""
CLI for offline evaluation and threshold tuning.

Example:
  python3 evaluate_model.py \
    --dataset-path /data/treesat_dataset \
    --ground-truth-csv /data/labels/ground_truth.csv \
    --species-threshold 0.5
"""

import argparse
import json
import sys

from backend.evaluation import evaluate_offline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline model evaluation and threshold tuning")
    parser.add_argument("--dataset-path", required=True, help="Path containing s1/ and s2/ tile folders")
    parser.add_argument("--ground-truth-csv", required=True, help="Ground-truth CSV path")
    parser.add_argument("--species-threshold", type=float, default=0.5, help="Default global species threshold")
    parser.add_argument(
        "--threshold-grid",
        default="",
        help="Comma-separated threshold candidates (e.g., 0.1,0.2,0.3,0.4,0.5)",
    )
    parser.add_argument("--output-dir", default="", help="Optional output directory")
    return parser.parse_args()


def parse_threshold_grid(raw: str):
    if not raw:
        return None
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values or None


def main() -> int:
    args = parse_args()

    threshold_grid = parse_threshold_grid(args.threshold_grid)

    result = evaluate_offline(
        dataset_path=args.dataset_path,
        ground_truth_csv=args.ground_truth_csv,
        species_threshold=args.species_threshold,
        threshold_grid=threshold_grid,
        output_dir=args.output_dir or None,
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
