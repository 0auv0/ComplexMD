#!/usr/bin/env python
"""Compare full BindMD, persistence, and released NeuralMD seed-42 metrics."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bindmd", default="outputs/misato_aligned_full/bindmd_test.json"
    )
    parser.add_argument(
        "--neuralmd",
        default=(
            "/data/shared/zwr/GOAI/NeuralMD/outputs/"
            "neuralmd_baseline_rerun_20260811/test_sde_seed42_all.json"
        ),
    )
    parser.add_argument(
        "--output", default="outputs/misato_aligned_full/comparison.json"
    )
    args = parser.parse_args()
    bindmd = json.loads(Path(args.bindmd).read_text())
    neuralmd = json.loads(Path(args.neuralmd).read_text())
    result = {
        "bindmd_result": str(Path(args.bindmd).resolve()),
        "neuralmd_result": str(Path(args.neuralmd).resolve()),
        "coordinate_note": (
            "BindMD and persistence use per-frame protein-aligned canonical coordinates; "
            "NeuralMD uses its released evaluator coordinates. Rigid-invariant internal "
            "metrics are directly comparable; absolute-coordinate metrics require caution."
        ),
        "scenarios": {},
    }
    for scenario in ("T1", "T2", "T3"):
        sources = {
            "bindmd": bindmd["aggregate"][scenario]["mean"],
            "persistence": bindmd["baselines"]["persistence"]["aggregate"][scenario]["mean"],
            "neuralmd": neuralmd["summary"][scenario]["mean"],
        }
        result["scenarios"][scenario] = {
            "num_complexes": bindmd["aggregate"][scenario]["num_complexes"],
            "metrics": {
                key: {name: values.get(key) for name, values in sources.items()}
                for key in KEYS
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
