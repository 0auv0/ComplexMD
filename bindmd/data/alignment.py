"""Protein-frame canonicalization for raw MISATO trajectories."""

from __future__ import annotations

import csv
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch_geometric.data import Data


def kabsch_transform(
    mobile: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return row-vector rotation and centroids mapping mobile to reference."""

    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[-1] != 3:
        raise ValueError("Kabsch inputs must have equal [atoms, 3] shapes")
    mobile_center = mobile.mean(dim=0)
    reference_center = reference.mean(dim=0)
    covariance = (mobile - mobile_center).transpose(0, 1) @ (
        reference - reference_center
    )
    u, _, vh = torch.linalg.svd(covariance)
    if torch.det(u @ vh) < 0:
        u = u.clone()
        u[:, -1] *= -1
    rotation = u @ vh
    return rotation, mobile_center, reference_center


def apply_rigid_transform(
    coordinates: torch.Tensor,
    rotation: torch.Tensor,
    mobile_center: torch.Tensor,
    reference_center: torch.Tensor,
) -> torch.Tensor:
    return (coordinates - mobile_center) @ rotation + reference_center


def first_residue_frame(
    pocket_n: torch.Tensor, pocket_ca: torch.Tensor, pocket_c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed right-handed frame with the first pocket N at the origin."""

    if pocket_n.shape[0] == 0:
        raise ValueError("cannot define a canonical frame from an empty pocket")
    origin = pocket_n[0]
    x_axis = torch.nn.functional.normalize(pocket_ca[0] - origin, dim=0)
    ca_to_c = pocket_c[0] - pocket_ca[0]
    y_seed = ca_to_c - torch.dot(ca_to_c, x_axis) * x_axis
    if y_seed.norm() < 1e-6:
        raise ValueError("first pocket N/CA/C atoms are collinear")
    y_axis = torch.nn.functional.normalize(y_seed, dim=0)
    z_axis = torch.linalg.cross(x_axis, y_axis)
    basis = torch.stack([x_axis, y_axis, z_axis], dim=1)
    return origin, basis


def rotation_matrix_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    """Convert batched proper rotations to stable axis-angle vectors."""

    cosine = (
        (rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
    ).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vector = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    scale = angle / (2.0 * torch.sin(angle).abs().clamp_min(1e-7))
    scale = torch.where(angle < 1e-4, 0.5 + angle.square() / 12.0, scale)
    return vector * scale.unsqueeze(-1)


def pose_deltas_from_alignment(
    alignment_rotation: torch.Tensor,
    mobile_center: torch.Tensor,
    canonical_basis: torch.Tensor | None = None,
    *,
    max_translation_step: float = 5.0,
    max_rotation_step_deg: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return body-frame SE(3) increments and a discontinuity validity mask."""

    # canonical = (aligned - origin) @ basis and
    # aligned = (world - center) @ alignment_rotation + reference_center.
    # Consequently canonical-to-world row rotations are
    # basis.T @ alignment_rotation.T.  Expressing increments in that frame
    # removes every complex's arbitrary input-PDB orientation.
    world_rotation = alignment_rotation.transpose(-1, -2)
    if canonical_basis is not None:
        world_rotation = canonical_basis.transpose(-1, -2) @ world_rotation
    world_translation = mobile_center[1:] - mobile_center[:-1]
    local_translation = torch.bmm(
        world_translation[:, None], world_rotation[:-1].transpose(-1, -2)
    ).squeeze(1)
    local_rotation = world_rotation[:-1].transpose(-1, -2) @ world_rotation[1:]
    axis_angle = rotation_matrix_to_axis_angle(local_rotation)
    valid = (
        torch.isfinite(local_translation).all(dim=-1)
        & torch.isfinite(axis_angle).all(dim=-1)
        & (world_translation.norm(dim=-1) <= max_translation_step)
        & (torch.rad2deg(axis_angle.norm(dim=-1)) <= max_rotation_step_deg)
    )
    return torch.cat([local_translation, axis_angle], dim=-1), valid


def _load_backbone_masks(
    group: Any, ligand_begin: int, neuralmd_repo: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from NeuralMD.datasets.MISATO.common import extract_backbone

    residue_map, atom_type_map, atom_name_map = _misato_maps(str(neuralmd_repo))
    element_map = {
        1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na",
        12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
        19: "K", 20: "Ca", 34: "Se", 35: "Br", 53: "I",
    }
    return extract_backbone(
        group["atoms_type"][:ligand_begin],
        group["atoms_residue"][:ligand_begin],
        group["atoms_number"][:ligand_begin],
        atom_type_map,
        residue_map,
        atom_name_map,
        element_map,
        group["molecules_begin_atom_index"][:][-1:],
    )


@lru_cache(maxsize=4)
def _misato_maps(neuralmd_repo: str) -> tuple[dict, dict, dict]:
    utils = Path(neuralmd_repo) / "NeuralMD" / "datasets" / "MISATO" / "utils"
    with (utils / "atoms_residue_map.pickle").open("rb") as handle:
        residue_map = pickle.load(handle)
    with (utils / "atoms_type_map.pickle").open("rb") as handle:
        atom_type_map = pickle.load(handle)
    with (utils / "atoms_name_map_for_pdb.pickle").open("rb") as handle:
        atom_name_map = pickle.load(handle)
    return residue_map, atom_type_map, atom_name_map


@lru_cache(maxsize=4)
def _atomic_masses(neuralmd_repo: str) -> dict[int, float]:
    path = Path(neuralmd_repo) / "NeuralMD" / "datasets" / "periodic_table.csv"
    with path.open(newline="") as handle:
        return {
            int(row["AtomicNumber"]): float(row["AtomicMass"])
            for row in csv.DictReader(handle)
        }


def load_aligned_misato_complex(
    hdf5_path: str | Path | h5py.File,
    identifier: str,
    *,
    neuralmd_repo: str | Path,
    pocket_cutoff: float = 12.0,
    max_pocket_residues: int = 128,
) -> Data:
    """Load one raw trajectory into the fixed frame-0 protein-pocket frame.

    The pocket is selected once from frame 0. Every later protein pocket is
    fitted to that reference, and the identical rigid transform is applied to
    the ligand. A second, fixed frame-0 transform places the first pocket N at
    the origin and defines orientation from its N/CA/C atoms.
    """

    neuralmd_repo = Path(neuralmd_repo)
    owns_handle = isinstance(hdf5_path, (str, Path))
    handle = h5py.File(hdf5_path, "r") if owns_handle else hdf5_path
    try:
        group = handle[identifier]
        coordinates = torch.as_tensor(
            group["trajectory_coordinates"][:], dtype=torch.float32
        )
        atom_number = group["atoms_number"][:]
        residue_index = group["atoms_residue"][:]
        ligand_begin = int(group["molecules_begin_atom_index"][:][-1])
        energy = torch.as_tensor(
            group["frames_interaction_energy"][:], dtype=torch.float32
        ).unsqueeze(0)
        mask_backbone, mask_ca, mask_c, mask_n = _load_backbone_masks(
            group, ligand_begin, neuralmd_repo
        )
    finally:
        if owns_handle:
            handle.close()

    backbone = coordinates[:, :ligand_begin, :][:, mask_backbone, :]
    mask_n_t = torch.as_tensor(mask_n, dtype=torch.bool)
    mask_ca_t = torch.as_tensor(mask_ca, dtype=torch.bool)
    mask_c_t = torch.as_tensor(mask_c, dtype=torch.bool)
    backbone_n = backbone[:, mask_n_t]
    backbone_ca = backbone[:, mask_ca_t]
    backbone_c = backbone[:, mask_c_t]

    heavy_mask = atom_number[ligand_begin:] != 1
    ligand = coordinates[:, ligand_begin:, :][:, heavy_mask, :]
    ligand_atomic_number = atom_number[ligand_begin:][heavy_mask]

    frame0_distance = torch.cdist(backbone_ca[0], ligand[0]).amin(dim=-1)
    pocket_indices = torch.nonzero(
        frame0_distance <= pocket_cutoff, as_tuple=False
    ).flatten()
    if pocket_indices.numel() == 0:
        pocket_indices = torch.argsort(frame0_distance)[: min(16, backbone_ca.shape[1])]
    if pocket_indices.numel() > max_pocket_residues:
        nearest = torch.argsort(frame0_distance[pocket_indices])[:max_pocket_residues]
        pocket_indices = pocket_indices[nearest]
    pocket_indices = pocket_indices.sort().values

    reference_alignment = torch.stack(
        [
            backbone_n[0, pocket_indices],
            backbone_ca[0, pocket_indices],
            backbone_c[0, pocket_indices],
        ],
        dim=1,
    ).reshape(-1, 3)
    mobile_alignment = torch.stack(
        [
            backbone_n[:, pocket_indices],
            backbone_ca[:, pocket_indices],
            backbone_c[:, pocket_indices],
        ],
        dim=2,
    ).reshape(coordinates.shape[0], -1, 3)
    mobile_center = mobile_alignment.mean(dim=1)
    reference_center = reference_alignment.mean(dim=0)
    mobile_zero = mobile_alignment - mobile_center[:, None]
    reference_zero = reference_alignment - reference_center
    covariance = mobile_zero.transpose(1, 2) @ reference_zero.expand(
        coordinates.shape[0], -1, -1
    )
    u, _, vh = torch.linalg.svd(covariance)
    improper = torch.det(u @ vh) < 0
    u = u.clone()
    u[improper, :, -1] *= -1
    rotation = u @ vh
    fitted = mobile_zero @ rotation + reference_center
    aligned_ligand = (
        (ligand - mobile_center[:, None]) @ rotation + reference_center
    )
    alignment_rmsd = torch.sqrt(((fitted - reference_alignment) ** 2).mean(dim=(1, 2)))
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation_angle = torch.rad2deg(
        torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    )
    translation_norm = (
        reference_center - torch.bmm(mobile_center[:, None], rotation).squeeze(1)
    ).norm(dim=-1)
    pocket_n0 = backbone_n[0, pocket_indices]
    pocket_ca0 = backbone_ca[0, pocket_indices]
    pocket_c0 = backbone_c[0, pocket_indices]
    origin, basis = first_residue_frame(pocket_n0, pocket_ca0, pocket_c0)
    world_rotation = basis.transpose(-1, -2) @ rotation.transpose(-1, -2)
    pose_delta, pose_valid = pose_deltas_from_alignment(
        rotation, mobile_center, canonical_basis=basis
    )
    canonical = lambda value: (value - origin) @ basis

    canonical_backbone = canonical(backbone[0])
    canonical_ligand = canonical(aligned_ligand)
    residue = torch.as_tensor(
        residue_index[:ligand_begin][mask_backbone][mask_ca] - 1,
        dtype=torch.long,
    )
    masses = _atomic_masses(str(neuralmd_repo))
    ligand_mass = torch.tensor(
        [masses[int(number)] for number in ligand_atomic_number], dtype=torch.float32
    )

    before_step = torch.linalg.vector_norm(ligand[1:] - ligand[:-1], dim=-1).mean(-1)
    after_step = torch.linalg.vector_norm(
        canonical_ligand[1:] - canonical_ligand[:-1], dim=-1
    ).mean(-1)
    return Data(
        protein_pos=canonical_backbone,
        protein_backbone_residue=residue,
        mask_ca=mask_ca_t,
        mask_c=mask_c_t,
        mask_n=mask_n_t,
        ligand_x=torch.as_tensor(ligand_atomic_number - 1, dtype=torch.long),
        ligand_mass=ligand_mass,
        ligand_trajectory_pos=canonical_ligand.transpose(0, 1).contiguous(),
        energy=energy,
        fixed_pocket_n=canonical(pocket_n0),
        fixed_pocket_ca=canonical(pocket_ca0),
        fixed_pocket_c=canonical(pocket_c0),
        fixed_pocket_residue=residue[pocket_indices],
        fixed_pocket_indices=pocket_indices,
        bindmd_canonical=torch.tensor(True),
        canonical_origin_atom=torch.tensor(7),
        alignment_pocket_rmsd=alignment_rmsd,
        alignment_rotation_angle_deg=rotation_angle,
        alignment_translation_norm=translation_norm,
        pocket_alignment_rotation=rotation,
        pocket_world_rotation=world_rotation,
        pocket_world_center=mobile_center,
        pocket_pose_delta=pose_delta,
        pocket_pose_valid=pose_valid,
        alignment_reference_center=reference_center,
        canonical_origin=origin,
        canonical_basis=basis,
        raw_ligand_step_mean=before_step,
        aligned_ligand_step_mean=after_step,
    )


def alignment_summary(data: Data) -> dict[str, float | int]:
    """Compact diagnostics suitable for cache metadata and experiment reports."""

    return {
        "pocket_residues": int(data.fixed_pocket_ca.shape[0]),
        "alignment_rmsd_mean": float(data.alignment_pocket_rmsd.mean()),
        "alignment_rmsd_max": float(data.alignment_pocket_rmsd.max()),
        "rotation_angle_deg_max": float(data.alignment_rotation_angle_deg.max()),
        "translation_norm_max": float(data.alignment_translation_norm.max()),
        "pose_invalid_steps": int((~data.pocket_pose_valid).sum()),
        "raw_ligand_step_mean": float(data.raw_ligand_step_mean.mean()),
        "raw_ligand_step_max": float(data.raw_ligand_step_mean.max()),
        "aligned_ligand_step_mean": float(data.aligned_ligand_step_mean.mean()),
        "aligned_ligand_step_max": float(data.aligned_ligand_step_mean.max()),
    }
