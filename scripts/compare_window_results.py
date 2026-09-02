#!/usr/bin/env python
"""Compare a new ComplexMD window experiment with retained baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = (
    "neuralmd_rmse",
    "neuralmd_matching",
    "neuralmd_stability",
    "geo_ligand_rmsd",
    "geo_internal_distance_rmse",
    "phys_inferred_bond_length_rmse",
    "phys_ligand_clash_rate",
    "dyn_rmsf_mae",
    "dyn_rmsf_correlation",
    "dyn_rg_mae",
    "dyn_contact_occupancy_mae",
    "stab_error_growth_slope",
    "fragment_torsion_angle_mae_deg",
    "fragment_reference_drift_rmse",
    "pose_translation_mean_angstrom",
    "pose_rotation_mean_deg",
    "world_ligand_frame_rmsd",
    "world_pocket_frame_rmsd",
    "world_complex_frame_rmsd",
)


def bindmd(directory: Path, tier: str) -> dict:
    payload = json.loads((directory / f"{tier}.json").read_text())
    return payload["aggregate"][tier]["mean"]


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--v2-dir", required=True)
    parser.add_argument("--v3-dir")
    parser.add_argument("--neuralmd", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    v2_dir = Path(args.v2_dir)
    v3_dir = Path(args.v3_dir) if args.v3_dir else None
    neuralmd_payload = json.loads(Path(args.neuralmd).read_text())
    include_v3 = bool(
        v3_dir is not None
        and all((v3_dir / f"{tier}.json").is_file() for tier in ("T1", "T2", "T3"))
    )
    result = {
        "metric_direction": {
            metric: ("higher" if metric in {"neuralmd_stability", "dyn_rmsf_correlation"} else "lower")
            for metric in METRICS
        },
        "v3_included": include_v3,
        "scenarios": {},
    }
    for tier in ("T1", "T2", "T3"):
        sources = {
            "v4_current8_history4_scratch": bindmd(candidate_dir, tier),
            "v2_rigid_fragment": bindmd(v2_dir, tier),
            "neuralmd": neuralmd_payload["summary"][tier]["mean"],
        }
        if include_v3:
            sources["v3_current6_history6"] = bindmd(v3_dir, tier)
        metrics = {}
        for metric in METRICS:
            values = {name: finite(mean.get(metric)) for name, mean in sources.items()}
            candidate = values["v4_current8_history4_scratch"]
            comparisons = {}
            for name, baseline in values.items():
                if name == "v4_current8_history4_scratch" or candidate is None or baseline is None:
                    continue
                delta = candidate - baseline
                comparisons[name] = {
                    "absolute_delta": delta,
                    "relative_delta": delta / abs(baseline) if baseline != 0 else None,
                }
            metrics[metric] = {"values": values, "candidate_minus_baseline": comparisons}
        result["scenarios"][tier] = {"metrics": metrics}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
