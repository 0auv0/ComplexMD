#!/usr/bin/env python
"""Select rigid-projection hyperparameters using validation results only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCENARIOS = ("T1", "T2", "T3")


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("split") != "val":
        raise ValueError(f"selection input must use the val split: {path}")
    for scenario in SCENARIOS:
        if payload["aggregate"][scenario]["num_complexes"] <= 0:
            raise ValueError(f"empty {scenario} result: {path}")
    return payload


def safe_ratio(value: float, reference: float) -> float:
    denominator = max(abs(float(reference)), 1e-8)
    ratio = float(value) / denominator
    if not math.isfinite(ratio):
        raise ValueError("non-finite validation score")
    return ratio


def structural_score(payload: dict) -> float:
    """Matching and instability relative to the same-set persistence baseline."""
    terms = []
    baseline = payload["baselines"]["persistence"]["aggregate"]
    for scenario in SCENARIOS:
        mean = payload["aggregate"][scenario]["mean"]
        persistence = baseline[scenario]["mean"]
        terms.append(
            safe_ratio(mean["neuralmd_matching"], persistence["neuralmd_matching"])
        )
        terms.append(
            safe_ratio(
                100.0 - mean["neuralmd_stability"],
                100.0 - persistence["neuralmd_stability"],
            )
        )
    return sum(terms) / len(terms)


def motion_score(payload: dict) -> float:
    """Coordinate/dynamics errors relative to same-set persistence."""
    terms = []
    baseline = payload["baselines"]["persistence"]["aggregate"]
    keys = ("neuralmd_rmse", "dyn_rmsf_mae", "stab_error_growth_slope")
    for scenario in SCENARIOS:
        mean = payload["aggregate"][scenario]["mean"]
        persistence = baseline[scenario]["mean"]
        terms.extend(safe_ratio(mean[key], persistence[key]) for key in keys)
    return sum(terms) / len(terms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="outputs/rigid_validation_128")
    parser.add_argument(
        "--output", default="outputs/rigid_validation_128/selection.json"
    )
    args = parser.parse_args()
    directory = Path(args.directory)

    alpha_files = {
        "0": directory / "alpha_0_scale_1.json",
        "0.25": directory / "alpha_025_scale_1.json",
        "1": directory / "alpha_1_scale_1.json",
    }
    scale_files = {
        "0": directory / "alpha_0_scale_0.json",
        "0.25": directory / "alpha_0_scale_025.json",
        "0.5": directory / "alpha_0_scale_05.json",
        "1": directory / "alpha_0_scale_1.json",
    }
    alpha_payloads = {key: load(path) for key, path in alpha_files.items()}
    alpha_scores = {
        key: structural_score(payload) for key, payload in alpha_payloads.items()
    }
    best_alpha = min(alpha_scores, key=alpha_scores.get)
    if best_alpha != "0":
        raise ValueError(
            "scale grid was evaluated with alpha=0, but validation selected "
            f"alpha={best_alpha}; run a scale grid at the selected alpha"
        )

    scale_payloads = {key: load(path) for key, path in scale_files.items()}
    scale_scores = {
        key: motion_score(payload) for key, payload in scale_payloads.items()
    }
    best_scale = min(scale_scores, key=scale_scores.get)
    result = {
        "selection_split": "val",
        "num_complexes": alpha_payloads[best_alpha]["aggregate"]["T1"][
            "num_complexes"
        ],
        "alpha_objective": (
            "mean of Matching/persistence-Matching and "
            "(100-Stability)/(100-persistence-Stability) over T1/T2/T3; lower"
        ),
        "alpha_scores": alpha_scores,
        "selected_internal_deformation_scale": float(best_alpha),
        "scale_objective": (
            "mean of RMSE, RMSF-MAE and error-growth-slope ratios to persistence "
            "over T1/T2/T3; lower"
        ),
        "scale_scores": scale_scores,
        "selected_flow_base_scale": float(best_scale),
        "inputs": {
            "alpha": {key: str(path.resolve()) for key, path in alpha_files.items()},
            "scale": {key: str(path.resolve()) for key, path in scale_files.items()},
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
