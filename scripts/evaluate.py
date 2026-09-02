#!/usr/bin/env python
"""Evaluate BindMD with NeuralMD metrics and GOAI trajectory proxies."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from bindmd.data import (
    MISATOAlignedDataset,
    MISATOProcessedDataset,
    collate_bindmd,
    prepare_complex,
)
from bindmd.evaluation.metrics import compute_all_metrics, finite_mean
from bindmd.models import build_model
from bindmd.models.geometry import (
    dihedral,
    integrate_pose_deltas,
    rotation_geodesic_angle,
)


SCENARIOS = {
    "T1": {"observed": 10, "predicted": 10},
    "T2": {"observed": 80, "predicted": 20},
    "T3": {"observed": 20, "predicted": 80},
}


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluation_item(
    raw,
    observed_frames: int,
    predicted_frames: int,
    history_frames: int,
    pocket_cutoff: float,
    max_pocket_residues: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    raw_trajectory = raw.ligand_trajectory_pos.transpose(0, 1)
    item = prepare_complex(
        raw,
        reference_ligand=raw_trajectory[observed_frames - 1],
        pocket_cutoff=pocket_cutoff,
        max_pocket_residues=max_pocket_residues,
    )
    trajectory = item.pop("trajectory")
    pose_delta = item.pop("pocket_pose_delta", None)
    pose_valid = item.pop("pocket_pose_valid", None)
    history = trajectory[max(0, observed_frames - history_frames):observed_frames]
    if history.shape[0] < history_frames:
        history = torch.cat(
            [history[:1].expand(history_frames - history.shape[0], -1, -1), history],
            dim=0,
        )
    target = trajectory[observed_frames:observed_frames + predicted_frames]
    item.update(
        {
            "history": history,
            "target": target[0],
            "target_index": torch.tensor(observed_frames),
        }
    )
    if pose_delta is not None:
        first_pose = max(0, observed_frames - 1 - history_frames)
        pose_history = pose_delta[first_pose:observed_frames - 1]
        pose_history_valid = pose_valid[first_pose:observed_frames - 1]
        if pose_history.shape[0] < history_frames:
            padding = history_frames - pose_history.shape[0]
            pose_history = torch.cat(
                [torch.zeros(padding, 6, dtype=pose_delta.dtype), pose_history]
            )
            pose_history_valid = torch.cat(
                [torch.zeros(padding, dtype=torch.bool), pose_history_valid]
            )
        pose_history = pose_history.clone()
        pose_history[~pose_history_valid] = 0.0
        item.update(
            {
                "pocket_pose_history": pose_history,
                "pocket_pose_history_valid": pose_history_valid,
                # Required by the shared collator, though rollout does not use it.
                "pocket_pose_target": pose_delta[observed_frames - 1],
                "pocket_pose_target_valid": pose_valid[observed_frames - 1],
            }
        )
    return item, target


def coordinate_summary(
    predicted: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    displacement = predicted - target
    atom_error = displacement.norm(dim=-1)
    frame_rmsd = displacement.square().sum(dim=-1).mean(dim=-1).sqrt()
    return {
        "rmse": float(atom_error.mean()),
        "frame_rmsd": float(frame_rmsd.mean()),
        "last_frame_rmsd": float(frame_rmsd[-1]),
    }


def fragment_motion_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    raw,
) -> dict[str, float]:
    """Metrics specific to the explicit rigid-fragment representation."""

    result = {
        "fragment_torsion_angle_mae_deg": float("nan"),
        "fragment_torsion_delta_mae_deg": float("nan"),
        "fragment_internal_distance_rmse": float("nan"),
        "fragment_reference_drift_rmse": float("nan"),
    }
    if not hasattr(raw, "rigid_fragment"):
        return result
    fragment = raw.rigid_fragment.to(predicted.device)
    pair_mask = fragment[:, None] == fragment[None, :]
    pair_mask &= torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    if bool(pair_mask.any()):
        predicted_distance = torch.cdist(predicted, predicted)
        target_distance = torch.cdist(target, target)
        reference_distance = torch.cdist(reference[None], reference[None])[0]
        target_error = predicted_distance[:, pair_mask] - target_distance[:, pair_mask]
        reference_error = (
            predicted_distance[:, pair_mask] - reference_distance[pair_mask]
        )
        result["fragment_internal_distance_rmse"] = float(
            target_error.square().mean().sqrt()
        )
        result["fragment_reference_drift_rmse"] = float(
            reference_error.square().mean().sqrt()
        )

    if hasattr(raw, "torsion_quad") and raw.torsion_quad.shape[0]:
        quad = raw.torsion_quad.to(predicted.device)
        frame_quad = quad[None].expand(predicted.shape[0], -1, -1)
        predicted_angle = dihedral(predicted, frame_quad)
        target_angle = dihedral(target, frame_quad)
        reference_angle = dihedral(reference[None], quad[None])[0]

        def wrapped(value: torch.Tensor) -> torch.Tensor:
            return torch.atan2(torch.sin(value), torch.cos(value))

        angle_error = wrapped(predicted_angle - target_angle).abs()
        predicted_delta = wrapped(predicted_angle - reference_angle[None])
        target_delta = wrapped(target_angle - reference_angle[None])
        delta_error = wrapped(predicted_delta - target_delta).abs()
        result["fragment_torsion_angle_mae_deg"] = float(
            torch.rad2deg(angle_error).mean()
        )
        result["fragment_torsion_delta_mae_deg"] = float(
            torch.rad2deg(delta_error).mean()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bindmd_base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split")
    parser.add_argument("--scenario", choices=["T1", "T2", "T3", "all"], default="all")
    parser.add_argument("--max-complexes", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--ddim-steps", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--internal-deformation-scale", type=float)
    parser.add_argument("--flow-base-scale", type=float)
    parser.add_argument("--torsion-base-scale", type=float)
    parser.add_argument("--torsion-step-limit-deg", type=float)
    parser.add_argument("--torsion-confidence-threshold", type=float)
    parser.add_argument("--translation-step-limit", type=float)
    parser.add_argument("--rotation-step-limit-deg", type=float)
    parser.add_argument("--pose-translation-scale", type=float, default=1.0)
    parser.add_argument("--pose-rotation-scale", type=float, default=1.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    evaluation = config["evaluation"]
    split = args.split or evaluation["split"]
    max_complexes = args.max_complexes
    if max_complexes is None:
        max_complexes = int(evaluation["max_complexes"])
    output_path = Path(args.output or evaluation["output"])
    sampling_steps = (
        args.sampling_steps
        or args.ddim_steps
        or int(evaluation.get("sampling_steps", evaluation.get("ddim_steps", 10)))
    )
    batch_size = int(evaluation.get("batch_size", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location="cpu")
    model_config = dict(payload.get("config", config)["model"])
    generation_method = model_config.get("generation_method", "diffusion")
    is_flow = generation_method in {
        "flow", "rectified_flow", "flow_matching", "se3_torsion",
        "se3_torsion_flow", "hierarchical_pose", "hierarchical_pose_flow",
        "hierarchical_pose_se3_torsion",
        "hierarchical_pose_se3_torsion_flow",
        "rigid_fragment", "rigid_fragment_flow",
        "hierarchical_pose_rigid_fragment",
        "hierarchical_pose_rigid_fragment_flow",
    }
    is_cartesian_flow = generation_method in {
        "flow", "rectified_flow", "flow_matching",
        "hierarchical_pose", "hierarchical_pose_flow",
    }
    if is_cartesian_flow:
        internal_deformation_scale = (
            args.internal_deformation_scale
            if args.internal_deformation_scale is not None
            else float(
                evaluation.get(
                    "internal_deformation_scale",
                    model_config.get("internal_deformation_scale", 1.0),
                )
            )
        )
        flow_base_scale = (
            args.flow_base_scale
            if args.flow_base_scale is not None
            else float(
                evaluation.get(
                    "flow_base_scale", model_config.get("flow_base_scale", 1.0)
                )
            )
        )
        model_config["internal_deformation_scale"] = internal_deformation_scale
        model_config["flow_base_scale"] = flow_base_scale
    elif not is_flow:
        if args.internal_deformation_scale is not None or args.flow_base_scale is not None:
            parser.error("Flow projection overrides require a Flow checkpoint")
        internal_deformation_scale = None
        flow_base_scale = None
    else:
        if args.internal_deformation_scale is not None or args.flow_base_scale is not None:
            parser.error("Cartesian Flow projection overrides do not apply to SE(3)+torsion Flow")
        internal_deformation_scale = None
        flow_base_scale = None
    model = build_model(model_config).to(device)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    if args.torsion_base_scale is not None:
        if not hasattr(model, "torsion_base_scale"):
            parser.error("--torsion-base-scale requires an SE(3)+torsion checkpoint")
        model.torsion_base_scale = float(args.torsion_base_scale)
    if args.torsion_step_limit_deg is not None:
        if not hasattr(model, "torsion_step_limit"):
            parser.error("--torsion-step-limit-deg requires an SE(3)+torsion checkpoint")
        model.torsion_step_limit = math.radians(float(args.torsion_step_limit_deg))
    if args.torsion_confidence_threshold is not None:
        if not hasattr(model, "fragment_torsion_head"):
            parser.error(
                "--torsion-confidence-threshold requires a rigid-fragment checkpoint"
            )
        threshold = float(args.torsion_confidence_threshold)
        if not 0.0 < threshold < 1.0:
            parser.error("--torsion-confidence-threshold must be between zero and one")
        model.fragment_torsion_head.confidence_threshold = threshold
    if args.translation_step_limit is not None:
        if not hasattr(model, "translation_step_limit"):
            parser.error("--translation-step-limit requires an SE(3)+torsion checkpoint")
        model.translation_step_limit = float(args.translation_step_limit)
    if args.rotation_step_limit_deg is not None:
        if not hasattr(model, "rotation_step_limit"):
            parser.error("--rotation-step-limit-deg requires an SE(3)+torsion checkpoint")
        model.rotation_step_limit = math.radians(float(args.rotation_step_limit_deg))
    model.eval()
    if "aligned_cache_dir" in config["data"]:
        dataset = MISATOAlignedDataset(
            config["data"]["aligned_cache_dir"],
            split,
            config["data"].get("topology_cache_dir"),
            config["data"].get("qm_hdf5"),
        )
    else:
        dataset = MISATOProcessedDataset(config["data"]["root"], split)
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    records: list[dict[str, float | int | str]] = []
    persistence_records: list[dict[str, float | int | str]] = []

    first_index = args.start_index
    if first_index < 0 or first_index >= len(dataset):
        parser.error("--start-index is outside the selected split")
    total = (
        min(len(dataset), first_index + max_complexes)
        if max_complexes
        else len(dataset)
    )
    for start in range(first_index, total, batch_size):
        indices = list(range(start, min(start + batch_size, total)))
        raws = [dataset[index] for index in indices]
        for scenario in scenarios:
            observed = SCENARIOS[scenario]["observed"]
            predicted_frames = SCENARIOS[scenario]["predicted"]
            built = [
                evaluation_item(
                    raw,
                    observed,
                    predicted_frames,
                    config["data"]["history_frames"],
                    config["data"]["pocket_cutoff"],
                    config["data"]["max_pocket_residues"],
                )
                for raw in raws
            ]
            items = [item for item, _ in built]
            targets = [target for _, target in built]
            batch = collate_bindmd(items)
            batch = move(batch, device)
            generators = [
                torch.Generator(device=device).manual_seed(int(config["seed"]) + index)
                for index in indices
            ]
            if hasattr(model, "rollout_complex"):
                pred, predicted_pose_delta = model.rollout_complex(
                    batch,
                    frames=targets[0].shape[0],
                    ddim_steps=sampling_steps,
                    generators=generators,
                )
                predicted_pose_delta = predicted_pose_delta.clone()
                predicted_pose_delta[..., :3] *= args.pose_translation_scale
                predicted_pose_delta[..., 3:] *= args.pose_rotation_scale
            else:
                pred = model.rollout(
                    batch,
                    frames=targets[0].shape[0],
                    ddim_steps=sampling_steps,
                    generators=generators,
                )
                predicted_pose_delta = None
            for offset, (index, raw, target) in enumerate(zip(indices, raws, targets)):
                n_ligand = raw.ligand_x.shape[0]
                prediction = pred[offset, :, :n_ligand]
                canonical = bool(
                    getattr(raw, "bindmd_canonical", torch.tensor(False)).item()
                )
                if canonical:
                    protein_pos = raw.protein_pos.to(device)
                else:
                    center = raw.protein_pos[raw.mask_ca].mean(dim=0)
                    protein_pos = (raw.protein_pos - center).to(device)
                protein_z = torch.full(
                    (protein_pos.shape[0],), 6, dtype=torch.long, device=device
                )
                protein_z[raw.mask_n.to(device)] = 7
                common = {
                    "target": target.to(device),
                    "ligand_z": raw.ligand_x.to(device) + 1,
                    "ligand_mass": raw.ligand_mass.to(device),
                    "protein_pos": protein_pos,
                    "protein_z": protein_z,
                }
                metrics = compute_all_metrics(pred=prediction, **common)
                persistence = batch["history"][offset, -1, :n_ligand].unsqueeze(0)
                persistence = persistence.expand_as(prediction)
                metrics.update(
                    fragment_motion_metrics(
                        prediction,
                        target.to(device),
                        batch["history"][offset, -1, :n_ligand],
                        raw,
                    )
                )
                persistence_world_metrics = {}
                if predicted_pose_delta is not None:
                    future_steps = target.shape[0]
                    predicted_rotation, predicted_center = integrate_pose_deltas(
                        raw.pocket_world_rotation[observed - 1].to(device),
                        raw.pocket_world_center[observed - 1].to(device),
                        predicted_pose_delta[offset, :future_steps],
                    )
                    target_rotation = raw.pocket_world_rotation[
                        observed:observed + future_steps
                    ].to(device)
                    target_center = raw.pocket_world_center[
                        observed:observed + future_steps
                    ].to(device)
                    translation_error = (predicted_center - target_center).norm(dim=-1)
                    rotation_error = torch.rad2deg(
                        rotation_geodesic_angle(predicted_rotation, target_rotation)
                    )
                    hold_rotation = raw.pocket_world_rotation[
                        observed - 1
                    ].to(device).unsqueeze(0).expand(future_steps, -1, -1)
                    hold_center = raw.pocket_world_center[
                        observed - 1
                    ].to(device).unsqueeze(0).expand(future_steps, -1)
                    hold_translation_error = (hold_center - target_center).norm(dim=-1)
                    hold_rotation_error = torch.rad2deg(
                        rotation_geodesic_angle(hold_rotation, target_rotation)
                    )
                    center_canonical = (
                        (raw.alignment_reference_center - raw.canonical_origin)
                        @ raw.canonical_basis
                    ).to(device)

                    def to_world(values, rotation, center):
                        return (
                            (values - center_canonical) @ rotation
                            + center[:, None]
                        )

                    predicted_ligand_world = to_world(
                        prediction, predicted_rotation, predicted_center
                    )
                    target_ligand_world = to_world(
                        target.to(device), target_rotation, target_center
                    )
                    hold_ligand_world = to_world(
                        prediction, hold_rotation, hold_center
                    )
                    hold_persistence_world = to_world(
                        persistence, hold_rotation, hold_center
                    )
                    pocket_template = torch.stack(
                        [raw.fixed_pocket_n, raw.fixed_pocket_ca, raw.fixed_pocket_c],
                        dim=1,
                    ).reshape(-1, 3).to(device)
                    pocket_frames = pocket_template.unsqueeze(0).expand(
                        future_steps, -1, -1
                    )
                    predicted_pocket_world = to_world(
                        pocket_frames, predicted_rotation, predicted_center
                    )
                    target_pocket_world = to_world(
                        pocket_frames, target_rotation, target_center
                    )
                    hold_pocket_world = to_world(
                        pocket_frames, hold_rotation, hold_center
                    )
                    complex_prediction = torch.cat(
                        [predicted_pocket_world, predicted_ligand_world], dim=1
                    )
                    complex_target = torch.cat(
                        [target_pocket_world, target_ligand_world], dim=1
                    )
                    hold_complex_prediction = torch.cat(
                        [hold_pocket_world, hold_ligand_world], dim=1
                    )
                    persistence_world_metrics = {
                        f"world_hold_last_{key}": value
                        for key, value in coordinate_summary(
                            hold_persistence_world, target_ligand_world
                        ).items()
                    }
                    metrics.update(
                        {
                            "pose_translation_mean_angstrom": float(translation_error.mean()),
                            "pose_translation_last_angstrom": float(translation_error[-1]),
                            "pose_rotation_mean_deg": float(rotation_error.mean()),
                            "pose_rotation_last_deg": float(rotation_error[-1]),
                            "pose_hold_last_translation_mean_angstrom": float(hold_translation_error.mean()),
                            "pose_hold_last_translation_last_angstrom": float(hold_translation_error[-1]),
                            "pose_hold_last_rotation_mean_deg": float(hold_rotation_error.mean()),
                            "pose_hold_last_rotation_last_deg": float(hold_rotation_error[-1]),
                            **{
                                f"world_ligand_{key}": value
                                for key, value in coordinate_summary(
                                    predicted_ligand_world, target_ligand_world
                                ).items()
                            },
                            **{
                                f"world_pocket_{key}": value
                                for key, value in coordinate_summary(
                                    predicted_pocket_world, target_pocket_world
                                ).items()
                            },
                            **{
                                f"world_complex_{key}": value
                                for key, value in coordinate_summary(
                                    complex_prediction, complex_target
                                ).items()
                            },
                            **{
                                f"world_hold_last_ligand_{key}": value
                                for key, value in coordinate_summary(
                                    hold_ligand_world, target_ligand_world
                                ).items()
                            },
                            **{
                                f"world_hold_last_pocket_{key}": value
                                for key, value in coordinate_summary(
                                    hold_pocket_world, target_pocket_world
                                ).items()
                            },
                            **{
                                f"world_hold_last_complex_{key}": value
                                for key, value in coordinate_summary(
                                    hold_complex_prediction, complex_target
                                ).items()
                            },
                        }
                    )
                persistence_metrics = compute_all_metrics(pred=persistence, **common)
                persistence_metrics.update(
                    fragment_motion_metrics(
                        persistence,
                        target.to(device),
                        batch["history"][offset, -1, :n_ligand],
                        raw,
                    )
                )
                persistence_metrics.update(persistence_world_metrics)
                identifier = (
                    dataset.identifiers[index]
                    if hasattr(dataset, "identifiers")
                    else str(index)
                )
                metadata = {
                    "complex_index": index,
                    "identifier": identifier,
                    "scenario": scenario,
                    "observed_frames": observed,
                    "predicted_frames": int(target.shape[0]),
                }
                records.append({**metadata, **metrics})
                persistence_records.append({**metadata, **persistence_metrics})
        print(f"evaluated {indices[-1] + 1}/{total}", flush=True)

    def aggregate_records(source: list[dict]) -> dict:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in source:
            grouped[str(record["scenario"])].append(record)
        aggregate = {}
        metadata = {
            "complex_index", "identifier", "scenario", "observed_frames",
            "predicted_frames",
        }
        for scenario, rows in grouped.items():
            metric_names = [key for key in rows[0] if key not in metadata]
            aggregate[scenario] = {
                "num_complexes": len(rows),
                "mean": {
                    key: finite_mean([float(row[key]) for row in rows])
                    for key in metric_names
                },
            }
        return aggregate

    aggregate = aggregate_records(records)
    persistence_aggregate = aggregate_records(persistence_records)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "generation_method": generation_method,
        "sampling_steps": sampling_steps,
        "internal_deformation_scale": internal_deformation_scale,
        "flow_base_scale": flow_base_scale,
        "torsion_base_scale": getattr(model, "torsion_base_scale", None),
        "torsion_step_limit_deg": (
            math.degrees(model.torsion_step_limit)
            if hasattr(model, "torsion_step_limit") else None
        ),
        "torsion_confidence_threshold": (
            model.fragment_torsion_head.confidence_threshold
            if hasattr(model, "fragment_torsion_head") else None
        ),
        "translation_step_limit": getattr(model, "translation_step_limit", None),
        "rotation_step_limit_deg": (
            math.degrees(model.rotation_step_limit)
            if hasattr(model, "rotation_step_limit") else None
        ),
        "pose_translation_scale": args.pose_translation_scale,
        "pose_rotation_scale": args.pose_rotation_scale,
        "rigid_internal_projection": bool(
            is_cartesian_flow and internal_deformation_scale is not None
            and internal_deformation_scale < 1.0
        ),
        "split": split,
        "start_index": first_index,
        "end_index_exclusive": total,
        "scenario_definition": {
            "T1": "observe frames 0:9, predict frames 10:19 (10 -> 10)",
            "T2": "observe frames 0:79, predict frames 80:99 (80 -> 20)",
            "T3": "observe frames 0:19, predict frames 20:99 (20 -> 80)",
        },
        "metric_status": "raw metrics; official competition normalization unavailable",
        "aggregate": aggregate,
        "records": records,
        "baselines": {
            "persistence": {
                "aggregate": persistence_aggregate,
                "records": persistence_records,
            }
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(json.dumps(aggregate, indent=2, allow_nan=True))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
