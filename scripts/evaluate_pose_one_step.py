#!/usr/bin/env python
"""Select hierarchical checkpoints with causal one-step pocket-pose validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from bindmd.data import MISATOAlignedDataset, MISATOFrameDataset, collate_bindmd
from bindmd.models import build_model
from bindmd.models.geometry import axis_angle_to_matrix, rotation_geodesic_angle


def move(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def evaluate(checkpoint: str, config: dict, loader, device) -> dict:
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(payload.get("config", config)["model"]).to(device)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    model.eval()
    totals = {
        "count": 0.0,
        "model_translation_l2": 0.0,
        "model_rotation_deg": 0.0,
        "zero_translation_l2": 0.0,
        "zero_rotation_deg": 0.0,
        "last_translation_l2": 0.0,
        "last_rotation_deg": 0.0,
    }
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            prediction = model.predict_pose_delta(batch)
            target = batch["pocket_pose_target"]
            valid = batch["pocket_pose_target_valid"]
            last_valid = batch["pocket_pose_history_valid"][:, -1]
            last = batch["pocket_pose_history"][:, -1]
            last = torch.where(last_valid[:, None], last, torch.zeros_like(last))
            count = float(valid.sum())
            totals["count"] += count
            for label, value in (
                ("model", prediction),
                ("zero", torch.zeros_like(target)),
                ("last", last),
            ):
                translation = (value[:, :3] - target[:, :3]).norm(dim=-1)
                rotation = torch.rad2deg(
                    rotation_geodesic_angle(
                        axis_angle_to_matrix(value[:, 3:]),
                        axis_angle_to_matrix(target[:, 3:]),
                    )
                )
                totals[f"{label}_translation_l2"] += float(translation[valid].sum())
                totals[f"{label}_rotation_deg"] += float(rotation[valid].sum())
    count = max(totals.pop("count"), 1.0)
    return {
        "checkpoint": checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "valid_examples": int(count),
        **{name: value / count for name, value in totals.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data = config["data"]
    base = MISATOAlignedDataset(data["aligned_cache_dir"], args.split)
    dataset = MISATOFrameDataset(
        base,
        history_frames=data["history_frames"],
        pocket_cutoff=data["pocket_cutoff"],
        max_pocket_residues=data["max_pocket_residues"],
        random_target=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=data["batch_size"],
        shuffle=False,
        num_workers=data["num_workers"],
        collate_fn=collate_bindmd,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for checkpoint in args.checkpoints:
        row = evaluate(checkpoint, config, loader, device)
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {"split": args.split, "selection_sample": "one causal target per complex", "results": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
