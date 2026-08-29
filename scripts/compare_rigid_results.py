#!/usr/bin/env python
"""Compare rigid-projected Flow BindMD with all retained baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


KEYS = (
    "neuralmd_rmse",
    "neuralmd_matching",
    "neuralmd_stability",
    "geo_ligand_rmsd",
    "geo_internal_distance_rmse",
    "phys_inferred_bond_length_rmse",
    "phys_ligand_clash_rate",
    "dyn_rmsf_mae",
    "dyn_rg_mae",
    "dyn_contact_occupancy_mae",
    "stab_error_growth_slope",
    "stab_step_displacement_p95",
)


def bindmd_mean(payload: dict, scenario: str) -> dict:
    return payload["aggregate"][scenario]["mean"]


def neuralmd_mean(payload: dict, scenario: str) -> dict:
    return payload["summary"][scenario]["mean"]


def finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def assert_same_test_set(reference: dict, candidate: dict, label: str) -> None:
    for scenario in ("T1", "T2", "T3"):
        expected = {
            (row["complex_index"], row["identifier"])
            for row in reference["records"]
            if row["scenario"] == scenario
        }
        observed = {
            (row["complex_index"], row["identifier"])
            for row in candidate["records"]
            if row["scenario"] == scenario
        }
        if observed != expected:
            raise ValueError(f"{label} {scenario} evaluates a different test set")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rigid", required=True)
    parser.add_argument(
        "--flow", default="outputs/misato_aligned_full_flow/bindmd_flow_test.json"
    )
    parser.add_argument(
        "--ddim", default="outputs/misato_aligned_full/bindmd_test.json"
    )
    parser.add_argument(
        "--neuralmd",
        default=(
            "/data/shared/zwr/GOAI/NeuralMD/outputs/"
            "neuralmd_baseline_rerun_20260811/test_sde_seed42_all.json"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "rigid_flow_bindmd": Path(args.rigid),
        "flow_bindmd": Path(args.flow),
        "ddim_bindmd": Path(args.ddim),
        "neuralmd": Path(args.neuralmd),
    }
    payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
    rigid = payloads["rigid_flow_bindmd"]
    assert_same_test_set(rigid, payloads["flow_bindmd"], "flow_bindmd")
    assert_same_test_set(rigid, payloads["ddim_bindmd"], "ddim_bindmd")

    result = {
        "inputs": {name: str(path.resolve()) for name, path in paths.items()},
        "rigid_projection": {
            "internal_deformation_scale": rigid.get("internal_deformation_scale"),
            "flow_base_scale": rigid.get("flow_base_scale"),
            "sampling_steps": rigid.get("sampling_steps"),
        },
        "coordinate_note": (
            "Both BindMD variants and persistence use protein-aligned canonical "
            "coordinates. NeuralMD uses its released evaluator coordinates, so "
            "rigid-invariant internal metrics are the safest direct comparison; "
            "absolute-coordinate metrics should be interpreted with caution."
        ),
        "metric_direction": {
            key: ("higher" if key == "neuralmd_stability" else "lower")
            for key in KEYS
        },
        "scenarios": {},
    }

    for scenario in ("T1", "T2", "T3"):
        sources = {
            "rigid_flow_bindmd": bindmd_mean(rigid, scenario),
            "flow_bindmd": bindmd_mean(payloads["flow_bindmd"], scenario),
            "ddim_bindmd": bindmd_mean(payloads["ddim_bindmd"], scenario),
            "neuralmd": neuralmd_mean(payloads["neuralmd"], scenario),
            "persistence": rigid["baselines"]["persistence"]["aggregate"][scenario][
                "mean"
            ],
        }
        metrics = {}
        for key in KEYS:
            values = {name: finite(source.get(key)) for name, source in sources.items()}
            rigid_value = values["rigid_flow_bindmd"]
            flow_value = values["flow_bindmd"]
            delta = None
            relative = None
            if rigid_value is not None and flow_value is not None:
                delta = rigid_value - flow_value
                if flow_value != 0:
                    relative = delta / abs(flow_value)
            metrics[key] = {
                "values": values,
                "rigid_minus_flow": delta,
                "rigid_relative_change_from_flow": relative,
            }
        result["scenarios"][scenario] = {
            "num_complexes": rigid["aggregate"][scenario]["num_complexes"],
            "metrics": metrics,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
