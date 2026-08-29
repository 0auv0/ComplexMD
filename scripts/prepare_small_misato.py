#!/usr/bin/env python
"""Cache a reproducible subset drawn strictly from MISATO's train split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small_sample.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data_config = config["data"]
    output = Path(data_config["cache"])
    if output.exists() and not args.force:
        print(f"cache already exists: {output}")
        return

    neuralmd_repo = Path(data_config["neuralmd_repo"])
    root = Path(data_config["root"])
    sys.path[:0] = ["/data", str(neuralmd_repo)]
    from NeuralMD.datasets.MISATO.dataset_MISATO_semi_flexible import (
        DatasetMISATOSemiFlexibleSingleTrajectory,
    )

    identifiers = [
        line.strip()
        for line in (root / "raw" / "train_MD.txt").read_text().splitlines()
        if line.strip()
    ]
    peptide_file = (
        neuralmd_repo
        / "NeuralMD"
        / "datasets"
        / "MISATO"
        / "utils"
        / "peptides.txt"
    )
    peptides = {
        line.strip().upper()
        for line in peptide_file.read_text().splitlines()
        if line.strip()
    }
    identifiers = [identifier for identifier in identifiers if identifier.upper() not in peptides]
    random.Random(config["seed"]).shuffle(identifiers)
    count = data_config["train_complexes"] + data_config["holdout_complexes"]
    selected = identifiers[:count]
    train_ids = selected[: data_config["train_complexes"]]
    holdout_ids = selected[data_config["train_complexes"] :]

    cases = {}
    for index, identifier in enumerate(selected):
        data = DatasetMISATOSemiFlexibleSingleTrajectory(
            str(root), identifier
        )[0]
        cases[identifier] = data
        print(
            f"{index + 1:02d}/{count:02d} {identifier} "
            f"ligand={data.ligand_x.shape[0]} residues={data.protein_backbone_residue.shape[0]}",
            flush=True,
        )
    payload = {
        "source_split": "train",
        "source_split_file": str(root / "raw" / "train_MD.txt"),
        "seed": config["seed"],
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    metadata = {key: value for key, value in payload.items() if key != "cases"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()

