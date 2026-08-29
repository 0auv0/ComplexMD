#!/usr/bin/env python
"""Validate a competition-format prediction and rigid geometry contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bindmd.data import apply_rigid_transform, kabsch_transform, load_goai_system


def rigid_rmsd(mobile: torch.Tensor, reference: torch.Tensor) -> float:
    rotation, mobile_center, reference_center = kabsch_transform(mobile, reference)
    fitted = apply_rigid_transform(
        mobile, rotation, mobile_center, reference_center
    )
    return float(torch.sqrt((fitted - reference).square().mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()

    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise ImportError("validation requires MDAnalysis") from exc

    system = load_goai_system(args.input_root, args.tier, args.id)
    sample = Path(args.input_root) / args.tier / args.id
    prediction = (
        Path(args.prediction_root) / args.tier / args.id / f"{args.id}_pred.xtc"
    )
    universe = mda.Universe(str(sample / f"{args.id}.pdb"), str(prediction))
    coordinates = torch.as_tensor(
        np.stack([step.positions.copy() for step in universe.trajectory]),
        dtype=torch.float32,
    )
    times = [float(step.time) for step in universe.trajectory]

    meta = system.meta
    protein_reference = system.observed_angstrom[0, system.protein_indices]
    ligand_reference = system.observed_angstrom[-1, system.ligand_indices]
    protein_rmsd = [
        rigid_rmsd(frame[system.protein_indices], protein_reference)
        for frame in coordinates
    ]
    ligand_rmsd = [
        rigid_rmsd(frame[system.ligand_indices], ligand_reference)
        for frame in coordinates
    ]
    report = {
        "id": args.id,
        "frames": int(coordinates.shape[0]),
        "expected_frames": int(meta["n_pred"]),
        "atoms": int(coordinates.shape[1]),
        "expected_atoms": int(meta["n_atoms"]),
        "finite": bool(torch.isfinite(coordinates).all()),
        "first_time_ps": times[0],
        "dt_ps": (times[1] - times[0]) if len(times) > 1 else None,
        "expected_dt_ps": float(meta["dt_ps"]),
        "protein_frame0_rigid_rmsd_max_angstrom": max(protein_rmsd),
        "ligand_template_rigid_rmsd_max_angstrom": max(ligand_rmsd),
    }
    report["valid"] = bool(
        report["frames"] == report["expected_frames"]
        and report["atoms"] == report["expected_atoms"]
        and report["finite"]
        and abs(report["dt_ps"] - report["expected_dt_ps"]) < 1e-4
        and report["protein_frame0_rigid_rmsd_max_angstrom"] < 0.02
        and report["ligand_template_rigid_rmsd_max_angstrom"] < 0.02
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
