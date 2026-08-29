#!/usr/bin/env python
"""Validate the competition material-A XTC contract for T1--T3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_xtc(path: Path, pdb_path: Path) -> np.ndarray:
    try:
        import mdtraj as md
    except ImportError:
        md = None
    if md is not None:
        return np.asarray(md.load(str(path), top=str(pdb_path)).xyz) * 10.0
    import MDAnalysis as mda

    universe = mda.Universe(str(pdb_path), str(path))
    return np.stack(
        [time_step.positions.copy() for time_step in universe.trajectory]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    prediction_root = Path(args.prediction_root)
    evaluation_root = Path(args.evaluation_root)
    records, errors = [], []
    for tier in ("T1", "T2", "T3"):
        expected_ids = [
            line.strip()
            for line in (evaluation_root / tier / "ids.txt").read_text().splitlines()
            if line.strip()
        ]
        expected_names = {f"{identifier}_pred.xtc" for identifier in expected_ids}
        tier_dir = prediction_root / tier
        actual_names = {path.name for path in tier_dir.glob("*_pred.xtc")}
        for missing in sorted(expected_names - actual_names):
            errors.append(f"{tier}: missing {missing}")
        for extra in sorted(actual_names - expected_names):
            errors.append(f"{tier}: unexpected {extra}")
        nested = list(tier_dir.glob("*/*_pred.xtc"))
        if nested:
            errors.append(f"{tier}: nested XTC paths are forbidden ({len(nested)})")
        for identifier in expected_ids:
            path = tier_dir / f"{identifier}_pred.xtc"
            if not path.exists():
                continue
            sample = evaluation_root / tier / identifier
            meta = json.loads((sample / "meta.json").read_text())
            coordinates = read_xtc(path, sample / f"{identifier}.pdb")
            record = {
                "id": identifier,
                "tier": tier,
                "frames": int(coordinates.shape[0]),
                "atoms": int(coordinates.shape[1]),
                "finite": bool(np.isfinite(coordinates).all()),
            }
            records.append(record)
            if record["frames"] != int(meta["n_pred"]):
                errors.append(
                    f"{identifier}: frames {record['frames']} != {meta['n_pred']}"
                )
            if record["atoms"] != int(meta["n_atoms"]):
                errors.append(
                    f"{identifier}: atoms {record['atoms']} != {meta['n_atoms']}"
                )
            if not record["finite"]:
                errors.append(f"{identifier}: contains NaN or infinite coordinates")

    result = {
        "valid": not errors,
        "expected_xtc": 90,
        "validated_xtc": len(records),
        "errors": errors,
        "records": records,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in result if key != "records"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
