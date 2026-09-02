#!/usr/bin/env python
"""Select a trajectory checkpoint using normalized validation proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def safe_ratio(value: float, reference: float) -> float:
    return float(value) / max(abs(float(reference)), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.evaluation_dir).glob("epoch_*_val20.json")):
        payload = json.loads(path.read_text())
        mean = payload["aggregate"]["T1"]["mean"]
        persistence = payload["baselines"]["persistence"]["aggregate"]["T1"]["mean"]
        components = {
            "coordinate": safe_ratio(mean["neuralmd_rmse"], persistence["neuralmd_rmse"]),
            "matching": safe_ratio(mean["neuralmd_matching"], persistence["neuralmd_matching"]),
            "instability": safe_ratio(
                100.0 - mean["neuralmd_stability"],
                100.0 - persistence["neuralmd_stability"],
            ),
            "rmsf": safe_ratio(mean["dyn_rmsf_mae"], persistence["dyn_rmsf_mae"]),
        }
        epoch = int(path.name.split("_")[1])
        checkpoint = Path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pt"
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "evaluation": str(path),
                "score": sum(components.values()) / len(components),
                "components": components,
                "metrics": {
                    key: mean[key]
                    for key in (
                        "neuralmd_rmse",
                        "neuralmd_matching",
                        "neuralmd_stability",
                        "dyn_rmsf_mae",
                    )
                },
            }
        )
    if not rows:
        raise RuntimeError("no checkpoint validation results found")
    best = min(rows, key=lambda row: row["score"])
    result = {
        "selection_rule": (
            "mean of model/persistence ratios for coordinate error, matching, "
            "instability (100-stability), and RMSF MAE; lower is better"
        ),
        "best": best,
        "candidates": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
