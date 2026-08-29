#!/usr/bin/env python
"""Causal backtest on the observed prefix of the anonymous GOAI systems.

The public package has no official future targets.  This script withholds the
tail of each observed trajectory, gives only the prefix to the model, and
reports ligand-relative, protein-pose, and reconstructed world-coordinate
metrics.  These are backtest metrics, not official leaderboard metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from bindmd.data import (
    build_goai_model_batch,
    canonicalize_goai_system,
    future_reference_poses,
    load_goai_system,
    restore_full_complex,
    rigid_project_ligand,
)
from bindmd.evaluation.metrics import compute_all_metrics, finite_mean
from bindmd.models import build_model
from bindmd.models.geometry import integrate_pose_deltas


BACKTEST = {
    "T1": {"observed": 8, "predicted": 2},
    "T2": {"observed": 60, "predicted": 20},
    "T3": {"observed": 10, "predicted": 10},
}


def load_model(checkpoint: Path, config: dict, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    model_config = dict(payload.get("config", config)["model"])
    evaluation = config.get("evaluation", {})
    generation_method = model_config.get("generation_method", "diffusion")
    if generation_method in {
        "flow", "rectified_flow", "flow_matching",
        "hierarchical_pose", "hierarchical_pose_flow",
    }:
        model_config["internal_deformation_scale"] = float(
            evaluation.get("internal_deformation_scale", 0.0)
        )
        model_config["flow_base_scale"] = float(
            evaluation.get("flow_base_scale", model_config.get("flow_base_scale", 1.0))
        )
    model = build_model(model_config).to(device)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    model.eval()
    return model, model_config


def rotation_error_deg(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = predicted.transpose(-1, -2) @ target
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))


def coordinate_errors(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    atom_error = (predicted - target).norm(dim=-1)
    frame_rmsd = (predicted - target).square().sum(dim=-1).mean(dim=-1).sqrt()
    return {
        "neuralmd_mae": float((predicted - target).abs().sum() / (target.shape[0] * target.shape[1])),
        "neuralmd_rmse": float(atom_error.mean()),
        "frame_rmsd_mean": float(frame_rmsd.mean()),
        "frame_rmsd_last": float(frame_rmsd[-1]),
    }


def aggregate(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["tier"]].append(record)
    metadata = {"id", "tier", "observed_frames", "predicted_frames"}
    result = {}
    for tier, rows in grouped.items():
        names = [name for name in rows[0] if name not in metadata]
        result[tier] = {
            "num_systems": len(rows),
            "mean": {
                name: finite_mean([float(row[name]) for row in rows])
                for name in names
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="GOAI_eval_public")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tiers", nargs="+", default=["T1", "T2", "T3"])
    parser.add_argument("--max-systems", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--pose-translation-scale", type=float, default=1.0)
    parser.add_argument("--pose-rotation-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.input_root)
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, model_config = load_model(Path(args.checkpoint), config, device)
    history_frames = int(config["data"]["history_frames"])
    cutoff = float(config["data"].get("pocket_cutoff", 12.0))
    max_pocket = int(config["data"].get("max_pocket_residues", 128))
    seed = int(config.get("seed", 42))

    model_records, persistence_records = [], []
    started = time.time()
    ordinal = 0
    for tier in args.tiers:
        definition = BACKTEST[tier]
        identifiers = [
            line.strip() for line in (root / tier / "ids.txt").read_text().splitlines()
            if line.strip()
        ]
        if args.max_systems:
            identifiers = identifiers[: args.max_systems]
        for index, identifier in enumerate(identifiers):
            full_system = load_goai_system(root, tier, identifier)
            full = canonicalize_goai_system(
                full_system, pocket_cutoff=cutoff, max_pocket_residues=max_pocket
            )
            observed = definition["observed"]
            predicted_frames = definition["predicted"]
            if full_system.observed_angstrom.shape[0] < observed + predicted_frames:
                raise ValueError(f"{identifier}: insufficient observed frames for backtest")
            prefix_meta = dict(full_system.meta)
            prefix_meta["n_obs"] = observed
            prefix_system = replace(
                full_system,
                meta=prefix_meta,
                observed_angstrom=full_system.observed_angstrom[:observed],
            )
            prefix = canonicalize_goai_system(
                prefix_system, pocket_cutoff=cutoff, max_pocket_residues=max_pocket
            )
            batch = build_goai_model_batch(prefix, history_frames)
            batch_device = {name: value.to(device) for name, value in batch.items()}
            generator = torch.Generator(device=device).manual_seed(seed + ordinal)
            with torch.no_grad():
                if hasattr(model, "rollout_complex"):
                    predicted_heavy_b, predicted_pose_delta_b = model.rollout_complex(
                        batch_device,
                        frames=predicted_frames,
                        ddim_steps=args.sampling_steps,
                        generator=generator,
                    )
                    predicted_heavy = predicted_heavy_b[0].cpu()
                    predicted_pose_delta = predicted_pose_delta_b[0].cpu()
                    predicted_pose_delta[:, :3] *= args.pose_translation_scale
                    predicted_pose_delta[:, 3:] *= args.pose_rotation_scale
                else:
                    predicted_heavy = model.rollout(
                        batch_device,
                        frames=predicted_frames,
                        ddim_steps=args.sampling_steps,
                        generator=generator,
                    )[0].cpu()
                    predicted_pose_delta = None
            predicted_ligand = rigid_project_ligand(prefix, predicted_heavy)
            predicted_heavy = predicted_ligand[:, prefix.ligand_heavy_local_indices]
            target_relative = full.observed_canonical[
                observed:observed + predicted_frames,
                full_system.ligand_heavy_indices,
            ]
            persistence_heavy = batch["history"][0, -1:].expand_as(predicted_heavy)

            protein_pos = torch.stack(
                [prefix.pocket_n, prefix.pocket_ca, prefix.pocket_c], dim=1
            ).reshape(-1, 3)
            protein_z = torch.tensor(
                [7, 6, 6], dtype=torch.long
            ).repeat(prefix.pocket_n.shape[0])
            common = {
                "target": target_relative,
                "ligand_z": full_system.atom_numbers[full_system.ligand_heavy_indices],
                "ligand_mass": full_system.atom_masses[full_system.ligand_heavy_indices],
                "protein_pos": protein_pos,
                "protein_z": protein_z,
            }
            model_metrics = compute_all_metrics(pred=predicted_heavy, **common)
            persistence_metrics = compute_all_metrics(pred=persistence_heavy, **common)

            ground_truth_rotation = (
                full.canonical_basis.T
                @ full.alignment_rotation[
                    observed:observed + predicted_frames
                ].transpose(-1, -2)
            )
            ground_truth_center = full.alignment_mobile_center[
                observed:observed + predicted_frames
            ]
            model_record = {
                "id": identifier,
                "tier": tier,
                "observed_frames": observed,
                "predicted_frames": predicted_frames,
                **{f"relative_{key}": value for key, value in model_metrics.items()},
            }
            persistence_record = {
                "id": identifier,
                "tier": tier,
                "observed_frames": observed,
                "predicted_frames": predicted_frames,
                **{f"relative_{key}": value for key, value in persistence_metrics.items()},
            }
            target_world = full_system.observed_angstrom[
                observed:observed + predicted_frames,
                full_system.ligand_heavy_indices,
            ]
            if predicted_pose_delta is not None:
                initial_rotation = (
                    prefix.canonical_basis.T
                    @ prefix.alignment_rotation[-1].T
                )
                model_rotation, model_center = integrate_pose_deltas(
                    initial_rotation,
                    prefix.alignment_mobile_center[-1],
                    predicted_pose_delta,
                )
                translation_error = (model_center - ground_truth_center).norm(dim=-1)
                angular_error = rotation_error_deg(
                    model_rotation, ground_truth_rotation
                )
                model_record.update(
                    {
                        "pose_model_translation_mean_angstrom": float(translation_error.mean()),
                        "pose_model_translation_last_angstrom": float(translation_error[-1]),
                        "pose_model_rotation_mean_deg": float(angular_error.mean()),
                        "pose_model_rotation_last_deg": float(angular_error[-1]),
                    }
                )
                restored_model_pose = restore_full_complex(
                    prefix,
                    predicted_ligand,
                    world_rotation=model_rotation,
                    world_center=model_center,
                )[:, full_system.ligand_heavy_indices]
                model_record.update(
                    {
                        f"world_model_{key}": value
                        for key, value in coordinate_errors(
                            restored_model_pose, target_world
                        ).items()
                    }
                )
            for pose_mode, pose_label in (
                ("hold_last", "hold_last"),
                ("constant_velocity", "constant_velocity"),
            ):
                predicted_rotation, predicted_center = future_reference_poses(
                    prefix, predicted_frames, mode=pose_mode
                )
                translation_error = (predicted_center - ground_truth_center).norm(dim=-1)
                angular_error = rotation_error_deg(
                    predicted_rotation, ground_truth_rotation
                )
                model_record.update(
                    {
                        f"pose_{pose_label}_translation_mean_angstrom": float(translation_error.mean()),
                        f"pose_{pose_label}_translation_last_angstrom": float(translation_error[-1]),
                        f"pose_{pose_label}_rotation_mean_deg": float(angular_error.mean()),
                        f"pose_{pose_label}_rotation_last_deg": float(angular_error[-1]),
                    }
                )
                restored_model = restore_full_complex(
                    prefix, predicted_ligand, pose_mode=pose_mode
                )[:, full_system.ligand_heavy_indices]
                restored_persistence = restore_full_complex(
                    prefix,
                    prefix.ligand_template_canonical.unsqueeze(0).expand(
                        predicted_frames, -1, -1
                    ),
                    pose_mode=pose_mode,
                )[:, full_system.ligand_heavy_indices]
                model_record.update(
                    {
                        f"world_{pose_label}_{key}": value
                        for key, value in coordinate_errors(restored_model, target_world).items()
                    }
                )
                persistence_record.update(
                    {
                        f"world_{pose_label}_{key}": value
                        for key, value in coordinate_errors(restored_persistence, target_world).items()
                    }
                )
            model_records.append(model_record)
            persistence_records.append(persistence_record)
            ordinal += 1
            print(
                f"[{tier} {index + 1}/{len(identifiers)}] {identifier}", flush=True
            )

    result = {
        "status": "causal backtest on withheld observed frames; not official hidden-future score",
        "backtest_definition": BACKTEST,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_config": model_config,
        "sampling_steps": args.sampling_steps,
        "pose_translation_scale": args.pose_translation_scale,
        "pose_rotation_scale": args.pose_rotation_scale,
        "elapsed_seconds": time.time() - started,
        "aggregate": aggregate(model_records),
        "records": model_records,
        "baselines": {
            "persistence": {
                "aggregate": aggregate(persistence_records),
                "records": persistence_records,
            }
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(json.dumps(result["aggregate"], indent=2, allow_nan=True))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
