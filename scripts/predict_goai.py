#!/usr/bin/env python
"""Generate competition-format all-atom XTC files with ComplexMD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from bindmd.data import (
    build_goai_model_batch,
    canonicalize_goai_system,
    fragment_project_ligand,
    load_goai_system,
    restore_full_complex,
    rigid_project_ligand,
    write_predicted_xtc,
)
from bindmd.models import build_model
from bindmd.models.geometry import integrate_pose_deltas


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def load_model(
    checkpoint: str,
    config_path: str | None,
    device: torch.device,
    flow_base_scale: float | None,
) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu")
    config = (
        yaml.safe_load(Path(config_path).read_text())
        if config_path
        else payload.get("config")
    )
    if config is None or "model" not in config:
        raise ValueError("checkpoint has no config; pass --config")
    model_config = dict(config["model"])
    evaluation = config.get("evaluation", {})
    if "internal_deformation_scale" in evaluation:
        model_config["internal_deformation_scale"] = float(
            evaluation["internal_deformation_scale"]
        )
    if flow_base_scale is not None:
        model_config["flow_base_scale"] = flow_base_scale
    elif "flow_base_scale" in evaluation:
        model_config["flow_base_scale"] = float(evaluation["flow_base_scale"])
    model = build_model(model_config).to(device)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    model.eval()
    return model, config


def identifiers(root: Path, tier: str, selected: list[str] | None) -> list[str]:
    available = [
        line.strip() for line in (root / tier / "ids.txt").read_text().splitlines()
        if line.strip()
    ]
    if not selected:
        return available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"unknown {tier} identifiers: {missing}")
    return [identifier for identifier in available if identifier in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--manifest-dir",
        help=(
            "Optional directory for diagnostic manifests. By default each manifest "
            "is written beside the tier predictions."
        ),
    )
    parser.add_argument("--tier", choices=["T1", "T2", "T3", "T4"], required=True)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--max-systems", type=int)
    parser.add_argument("--ligand-mode", choices=["model", "persistence"], default="model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--history-frames", type=int)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--flow-base-scale", type=float)
    parser.add_argument(
        "--topology-source",
        choices=["none", "auto", "conect", "smiles"],
        default="auto",
        help="Use supplied SMILES when available, otherwise public PDB CONECT.",
    )
    parser.add_argument(
        "--smiles-json",
        help="Optional local {system_id: smiles or {smiles, atom_order}} mapping.",
    )
    parser.add_argument(
        "--ligand-projection",
        choices=["auto", "whole", "fragments", "none"],
        default="auto",
    )
    parser.add_argument("--pose-translation-scale", type=float, default=1.0)
    parser.add_argument("--pose-rotation-scale", type=float, default=1.0)
    parser.add_argument(
        "--pose-mode",
        choices=["model", "hold_last", "constant_velocity"],
        default="hold_last",
        help="Estimate future global protein pose from observed frames only.",
    )
    parser.add_argument("--pocket-cutoff", type=float, default=12.0)
    parser.add_argument("--max-pocket-residues", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, config = None, None
    if args.ligand_mode == "model":
        if not args.checkpoint:
            parser.error("--checkpoint is required with --ligand-mode model")
        model, config = load_model(
            args.checkpoint, args.config, device, args.flow_base_scale
        )
    configured_history = (
        int(config["data"]["history_frames"])
        if config is not None and "data" in config
        else 8
    )
    history_frames = args.history_frames or configured_history

    root = Path(args.input_root)
    output_root = Path(args.output_dir)
    selected = identifiers(root, args.tier, args.ids)
    if args.max_systems:
        selected = selected[: args.max_systems]
    records = []
    smiles_catalog = (
        json.loads(Path(args.smiles_json).read_text()) if args.smiles_json else {}
    )

    for index, identifier in enumerate(selected):
        system = load_goai_system(root, args.tier, identifier)
        canonical = canonicalize_goai_system(
            system,
            pocket_cutoff=args.pocket_cutoff,
            max_pocket_residues=args.max_pocket_residues,
        )
        smiles_record = smiles_catalog.get(identifier)
        if isinstance(smiles_record, str):
            smiles, smiles_atom_order = smiles_record, None
        elif isinstance(smiles_record, dict):
            smiles = smiles_record.get("smiles")
            smiles_atom_order = smiles_record.get("atom_order")
        else:
            smiles, smiles_atom_order = None, None
        batch = build_goai_model_batch(
            canonical,
            history_frames,
            topology_source=args.topology_source,
            smiles=smiles,
            smiles_atom_order=smiles_atom_order,
        )
        frames = int(system.meta["n_pred"])
        if model is None:
            predicted_heavy = batch["history"][0, -1:].expand(frames, -1, -1)
            predicted_pose_delta = None
        else:
            model_batch = move(batch, device)
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            with torch.no_grad():
                if args.pose_mode == "model":
                    if not hasattr(model, "rollout_complex"):
                        raise ValueError(
                            "--pose-mode model requires a hierarchical pose checkpoint"
                        )
                    predicted_heavy_b, predicted_pose_delta_b = model.rollout_complex(
                        model_batch,
                        frames=frames,
                        ddim_steps=args.sampling_steps,
                        generator=generator,
                    )
                    predicted_heavy = predicted_heavy_b[0].cpu()
                    predicted_pose_delta = predicted_pose_delta_b[0].cpu()
                    predicted_pose_delta[:, :3] *= args.pose_translation_scale
                    predicted_pose_delta[:, 3:] *= args.pose_rotation_scale
                else:
                    predicted_heavy = model.rollout(
                        model_batch,
                        frames=frames,
                        ddim_steps=args.sampling_steps,
                        generator=generator,
                    )[0].cpu()
                    predicted_pose_delta = None

        projection = args.ligand_projection
        if projection == "auto":
            projection = (
                "fragments"
                if model is not None and hasattr(model, "torsion_target_scale")
                else "whole"
            )
        if projection == "whole":
            predicted_ligand = rigid_project_ligand(canonical, predicted_heavy)
        elif projection == "fragments":
            if "rigid_fragment" not in batch:
                raise ValueError("fragment projection requires ligand topology")
            predicted_ligand = fragment_project_ligand(
                canonical, predicted_heavy, batch["rigid_fragment"][0]
            )
        elif projection == "none":
            predicted_ligand = rigid_project_ligand(canonical, predicted_heavy)
            predicted_ligand[:, canonical.ligand_heavy_local_indices] = predicted_heavy
        else:
            raise AssertionError(projection)
        if predicted_pose_delta is not None:
            initial_rotation = (
                canonical.canonical_basis.T
                @ canonical.alignment_rotation[-1].T
            )
            predicted_rotation, predicted_center = integrate_pose_deltas(
                initial_rotation,
                canonical.alignment_mobile_center[-1],
                predicted_pose_delta,
            )
            full_prediction = restore_full_complex(
                canonical,
                predicted_ligand,
                world_rotation=predicted_rotation,
                world_center=predicted_center,
            )
        else:
            full_prediction = restore_full_complex(
                canonical, predicted_ligand, pose_mode=args.pose_mode
            )
        # Competition material A requires a flat directory per tier:
        # T1/T1-1_pred.xtc (no additional per-system directory).
        output_path = output_root / args.tier / f"{identifier}_pred.xtc"
        write_predicted_xtc(output_path, full_prediction, canonical)
        records.append(
            {
                "id": identifier,
                "tier": args.tier,
                "n_atoms": int(full_prediction.shape[1]),
                "n_pred": int(full_prediction.shape[0]),
                "ligand_mode": args.ligand_mode,
                "pose_mode": args.pose_mode,
                "pose_translation_scale": args.pose_translation_scale,
                "pose_rotation_scale": args.pose_rotation_scale,
                "topology_source": args.topology_source,
                "ligand_projection": projection,
                "rigid_fragments": int(
                    batch.get("rigid_fragment_count", torch.tensor([1]))[0]
                ),
                "output": str(output_path),
            }
        )
        print(f"[{index + 1}/{len(selected)}] wrote {output_path}", flush=True)

    manifest_root = Path(args.manifest_dir) if args.manifest_dir else output_root
    manifest = manifest_root / args.tier / "prediction_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2))
    print(f"saved {manifest}")


if __name__ == "__main__":
    main()
