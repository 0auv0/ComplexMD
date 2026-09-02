#!/usr/bin/env python
"""Train BindMD on NeuralMD-compatible processed MISATO trajectories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from bindmd.data import (
    MISATOAlignedDataset,
    MISATOFrameDataset,
    MISATOProcessedDataset,
    collate_bindmd,
)
from bindmd.models import build_model


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bindmd_base.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--resume")
    parser.add_argument("--initialize")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = config["data"]
    if "aligned_cache_dir" in data_config:
        base = MISATOAlignedDataset(
            data_config["aligned_cache_dir"],
            args.split,
            data_config.get("topology_cache_dir"),
            data_config.get("qm_hdf5"),
        )
    else:
        base = MISATOProcessedDataset(data_config["root"], args.split)
    dataset = MISATOFrameDataset(
        base,
        history_frames=data_config["history_frames"],
        pocket_cutoff=data_config["pocket_cutoff"],
        max_pocket_residues=data_config["max_pocket_residues"],
    )
    loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        collate_fn=collate_bindmd,
        pin_memory=True,
    )
    model = build_model(config["model"]).to(device)
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    start_epoch, global_step = 0, 0
    if args.resume and args.initialize:
        parser.error("--resume and --initialize are mutually exclusive")
    if args.initialize:
        payload = torch.load(args.initialize, map_location="cpu")
        state = payload["model"] if "model" in payload else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        permitted_missing = all(
            name.startswith(("pose_head.", "fragment_torsion_head."))
            for name in missing
        )
        if unexpected or (missing and not permitted_missing):
            raise RuntimeError(
                f"initial checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        print(
            f"initialized model weights from {args.initialize}; "
            f"new_parameters={missing}",
            flush=True,
        )
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload.get("global_step", 0))

    amp_enabled = bool(train_config["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint_dir = Path(train_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    stop = False
    for epoch in range(start_epoch, train_config["epochs"]):
        objective_name = (
            "flow_loss"
            if config["model"].get("generation_method") in {
                "flow", "rectified_flow", "flow_matching", "se3_torsion",
                "se3_torsion_flow", "hierarchical_pose", "hierarchical_pose_flow",
                "hierarchical_pose_se3_torsion",
                "hierarchical_pose_se3_torsion_flow",
                "rigid_fragment", "rigid_fragment_flow",
                "hierarchical_pose_rigid_fragment",
                "hierarchical_pose_rigid_fragment_flow",
            }
            else "diffusion_loss"
        )
        running = {"loss": 0.0, objective_name: 0.0, "pair_loss": 0.0}
        if config["model"].get("generation_method") in {
            "hierarchical_pose", "hierarchical_pose_flow",
            "hierarchical_pose_se3_torsion",
            "hierarchical_pose_se3_torsion_flow",
            "hierarchical_pose_rigid_fragment",
            "hierarchical_pose_rigid_fragment_flow",
        }:
            running.update(
                {
                    "pose_loss": 0.0,
                    "pose_translation_loss": 0.0,
                    "pose_rotation_loss": 0.0,
                }
            )
        if config["model"].get("generation_method") in {
            "rigid_fragment", "rigid_fragment_flow",
            "hierarchical_pose_rigid_fragment",
            "hierarchical_pose_rigid_fragment_flow",
        }:
            running.update(
                {
                    "torsion_confidence_loss": 0.0,
                    "torsion_active_rate": 0.0,
                    "torsion_confidence_mean": 0.0,
                    "torsion_predicted_active_rate": 0.0,
                }
            )
        for batch_index, batch in enumerate(loader):
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model.training_loss(
                    batch,
                    history_noise_max=train_config["history_noise_max"],
                    pair_loss_weight=train_config["pair_loss_weight"],
                )
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for key in running:
                running[key] += float(output[key])
            if global_step % 20 == 0:
                seen = batch_index + 1
                summary = {key: value / seen for key, value in running.items()}
                print(json.dumps({"epoch": epoch, "step": global_step, **summary}), flush=True)
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        }
        torch.save(payload, checkpoint_dir / "last.pt")
        torch.save(payload, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if stop:
            break


if __name__ == "__main__":
    main()
