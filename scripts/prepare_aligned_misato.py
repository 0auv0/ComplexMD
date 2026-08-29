#!/usr/bin/env python
"""Precompute full MISATO splits in fixed frame-0 protein-pocket coordinates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bindmd_full_aligned.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-complexes", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data_config = config["data"]
    root = Path(data_config["root"])
    neuralmd_repo = Path(data_config["neuralmd_repo"])
    output_dir = Path(data_config["aligned_cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = ["/data", str(neuralmd_repo)]
    from bindmd.data import alignment_summary, load_aligned_misato_complex

    peptide_file = (
        neuralmd_repo / "NeuralMD" / "datasets" / "MISATO" / "utils" / "peptides.txt"
    )
    peptides = {
        line.strip().upper()
        for line in peptide_file.read_text().splitlines()
        if line.strip()
    }
    hdf5_path = root / "raw" / "MD.hdf5"
    coordinate_system = {
        "pocket": "frame_0_ligand_distance_crop",
        "alignment": "per_frame_pocket_backbone_N_CA_C_Kabsch_to_frame_0",
        "origin": "first_frame_first_pocket_residue_N",
        "orientation": "frame_0_N_to_CA_and_CA_to_C_right_handed",
    }
    with h5py.File(hdf5_path, "r") as handle:
        for split in args.splits:
            output = output_dir / f"aligned_{split}.pt"
            if output.exists() and not args.force:
                print(f"cache already exists: {output}", flush=True)
                continue
            identifiers = [
                line.strip()
                for line in (root / "raw" / f"{split}_MD.txt").read_text().splitlines()
                if line.strip() and line.strip().upper() not in peptides
            ]
            if args.max_complexes:
                identifiers = identifiers[: args.max_complexes]
            cases, diagnostics = [], {}
            start = time.time()
            for index, identifier in enumerate(identifiers):
                data = load_aligned_misato_complex(
                    handle,
                    identifier,
                    neuralmd_repo=neuralmd_repo,
                    pocket_cutoff=data_config["pocket_cutoff"],
                    max_pocket_residues=data_config["max_pocket_residues"],
                )
                cases.append(data)
                diagnostics[identifier] = alignment_summary(data)
                if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(identifiers):
                    elapsed = time.time() - start
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "completed": index + 1,
                                "total": len(identifiers),
                                "elapsed_seconds": elapsed,
                                "complexes_per_second": (index + 1) / elapsed,
                            }
                        ),
                        flush=True,
                    )
            payload = {
                "source_split": split,
                "source_split_file": str(root / "raw" / f"{split}_MD.txt"),
                "identifiers": identifiers,
                "coordinate_system": coordinate_system,
                "cases": cases,
            }
            torch.save(payload, output)
            rows = list(diagnostics.values())
            summary = {
                "source_split": split,
                "num_complexes": len(identifiers),
                "coordinate_system": coordinate_system,
                "elapsed_seconds": time.time() - start,
                "mean_raw_ligand_step": sum(r["raw_ligand_step_mean"] for r in rows)
                / max(len(rows), 1),
                "mean_aligned_ligand_step": sum(
                    r["aligned_ligand_step_mean"] for r in rows
                )
                / max(len(rows), 1),
                "diagnostics": diagnostics,
            }
            output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
            print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
