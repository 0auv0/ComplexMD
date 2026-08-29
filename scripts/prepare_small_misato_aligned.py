#!/usr/bin/env python
"""Build the same tiny MISATO split in a fixed frame-0 protein-pocket frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small_sample_aligned.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data_config = config["data"]
    output = Path(data_config["cache"])
    if output.exists() and not args.force:
        print(f"cache already exists: {output}")
        return

    neuralmd_repo = Path(data_config["neuralmd_repo"])
    sys.path[:0] = ["/data", str(neuralmd_repo)]
    from bindmd.data.alignment import alignment_summary, load_aligned_misato_complex

    source = torch.load(data_config["source_cache"])
    train_ids = source["train_ids"]
    holdout_ids = source["holdout_ids"]
    identifiers = train_ids + holdout_ids
    hdf5_path = Path(data_config["root"]) / "raw" / "MD.hdf5"
    cases, diagnostics = {}, {}
    for index, identifier in enumerate(identifiers):
        data = load_aligned_misato_complex(
            hdf5_path,
            identifier,
            neuralmd_repo=neuralmd_repo,
            pocket_cutoff=data_config["pocket_cutoff"],
            max_pocket_residues=data_config["max_pocket_residues"],
        )
        cases[identifier] = data
        diagnostics[identifier] = alignment_summary(data)
        row = diagnostics[identifier]
        print(
            f"{index + 1:02d}/{len(identifiers):02d} {identifier} "
            f"pocket={row['pocket_residues']} "
            f"step={row['raw_ligand_step_mean']:.3f}->"
            f"{row['aligned_ligand_step_mean']:.3f} A",
            flush=True,
        )

    payload = {
        "source_split": "train",
        "source_split_file": source["source_split_file"],
        "seed": source["seed"],
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "coordinate_system": {
            "pocket": "frame_0_ligand_distance_crop",
            "alignment": "per_frame_pocket_backbone_N_CA_C_Kabsch_to_frame_0",
            "origin": "first_frame_first_pocket_residue_N",
            "orientation": "frame_0_N_to_CA_and_CA_to_C_right_handed",
        },
        "alignment_diagnostics": diagnostics,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    metadata = {key: value for key, value in payload.items() if key != "cases"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
