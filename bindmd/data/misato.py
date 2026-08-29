"""Causal adapter for NeuralMD's processed semi-flexible MISATO tensors."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset


class MISATOProcessedDataset(InMemoryDataset):
    """Read NeuralMD's processed files without copying or reprocessing them."""

    def __init__(self, root: str | Path, split: str):
        self.split = split
        super().__init__(str(root), transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["MD.hdf5", f"{self.split}_MD.txt"]

    @property
    def processed_file_names(self) -> str:
        return f"geometric_data_processed_{self.split}.pt"

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / "processed_semi_flexible")

    def process(self) -> None:
        raise FileNotFoundError(
            "BindMD consumes NeuralMD's processed MISATO files. "
            f"Expected {self.processed_paths[0]}"
        )


class MISATOAlignedDataset(Dataset):
    """Read a full split pre-aligned to each complex's frame-0 pocket."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        topology_cache_dir: str | Path | None = None,
    ):
        self.split = split
        self.path = Path(cache_dir) / f"aligned_{split}.pt"
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing aligned cache {self.path}; run prepare_aligned_misato.py"
            )
        payload = torch.load(self.path)
        self.identifiers = payload["identifiers"]
        self.cases = payload["cases"]
        if len(self.identifiers) != len(self.cases):
            raise ValueError(f"corrupt aligned cache: {self.path}")
        if topology_cache_dir is not None:
            topology_path = Path(topology_cache_dir) / f"topology_{split}.pt"
            topology = torch.load(topology_path)
            if topology["identifiers"] != self.identifiers:
                raise ValueError(
                    f"topology identifiers do not match aligned cache: {topology_path}"
                )
            if len(topology["cases"]) != len(self.cases):
                raise ValueError(f"corrupt topology cache: {topology_path}")
            for case, fields in zip(self.cases, topology["cases"]):
                for name, value in fields.items():
                    setattr(case, name, value)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        return self.cases[index]


def _backbone_triplet(data: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = data.protein_pos[data.mask_n]
    ca = data.protein_pos[data.mask_ca]
    c = data.protein_pos[data.mask_c]
    if not (n.shape == ca.shape == c.shape):
        raise ValueError("N/CA/C backbone masks do not describe equal residue counts")
    return n, ca, c


def prepare_complex(
    data: Any,
    *,
    reference_ligand: torch.Tensor | None = None,
    pocket_cutoff: float = 12.0,
    max_pocket_residues: int = 128,
) -> dict[str, torch.Tensor]:
    """Convert one PyG sample to causal, protein-centred model tensors.

    The source NeuralMD cache was centred with all frames. Recentring both
    molecules on the fixed protein CA centroid exactly removes that global
    offset, so future ligand coordinates do not leak into this representation.
    """

    if hasattr(data, "fixed_pocket_ca"):
        trajectory = data.ligand_trajectory_pos.transpose(0, 1)
        result = {
            "ligand_z": data.ligand_x.long() + 1,
            "ligand_mass": data.ligand_mass.float(),
            "trajectory": trajectory.float(),
            "pocket_n": data.fixed_pocket_n.float(),
            "pocket_ca": data.fixed_pocket_ca.float(),
            "pocket_c": data.fixed_pocket_c.float(),
            "pocket_residue": data.fixed_pocket_residue.long(),
            "energy": data.energy.squeeze(0).float(),
            "center": torch.zeros(3, dtype=trajectory.dtype),
        }
        for name in (
            "bond_index",
            "torsion_bond",
            "torsion_quad",
            "torsion_rotate_mask",
            "torsion_root",
            "pocket_pose_delta",
            "pocket_pose_valid",
            "pocket_world_rotation",
            "pocket_world_center",
            "alignment_reference_center",
            "canonical_origin",
            "canonical_basis",
        ):
            if hasattr(data, name):
                result[name] = getattr(data, name)
        return result

    n, ca, c = _backbone_triplet(data)
    center = ca.mean(dim=0)
    n, ca, c = n - center, ca - center, c - center
    trajectory = data.ligand_trajectory_pos.transpose(0, 1) - center
    reference = trajectory[0] if reference_ligand is None else reference_ligand - center

    distance = torch.cdist(ca, reference).amin(dim=-1)
    within = torch.nonzero(distance <= pocket_cutoff, as_tuple=False).flatten()
    if within.numel() == 0:
        within = torch.argsort(distance)[: min(16, ca.shape[0])]
    if within.numel() > max_pocket_residues:
        nearest = torch.argsort(distance[within])[:max_pocket_residues]
        within = within[nearest]
    within = within.sort().values

    return {
        "ligand_z": data.ligand_x.long() + 1,
        "ligand_mass": data.ligand_mass.float(),
        "trajectory": trajectory.float(),
        "pocket_n": n[within].float(),
        "pocket_ca": ca[within].float(),
        "pocket_c": c[within].float(),
        "pocket_residue": data.protein_backbone_residue[within].long(),
        "energy": data.energy.squeeze(0).float(),
        "center": center.float(),
    }


class MISATOFrameDataset(Dataset):
    """Sample next-frame autoregressive training examples from full trajectories."""

    def __init__(
        self,
        base: Dataset,
        *,
        history_frames: int = 8,
        pocket_cutoff: float = 12.0,
        max_pocket_residues: int = 128,
        random_target: bool = True,
    ):
        self.base = base
        self.history_frames = history_frames
        self.pocket_cutoff = pocket_cutoff
        self.max_pocket_residues = max_pocket_residues
        self.random_target = random_target

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw = self.base[index]
        total_frames = raw.ligand_trajectory_pos.shape[1]
        target_index = (
            random.randint(1, total_frames - 1)
            if self.random_target
            else 1 + (index % (total_frames - 1))
        )
        raw_trajectory = raw.ligand_trajectory_pos.transpose(0, 1)
        complex_data = prepare_complex(
            raw,
            reference_ligand=raw_trajectory[target_index - 1],
            pocket_cutoff=self.pocket_cutoff,
            max_pocket_residues=self.max_pocket_residues,
        )
        trajectory = complex_data.pop("trajectory")
        pose_delta = complex_data.pop("pocket_pose_delta", None)
        pose_valid = complex_data.pop("pocket_pose_valid", None)
        first = max(0, target_index - self.history_frames)
        history = trajectory[first:target_index]
        if history.shape[0] < self.history_frames:
            pad = history[:1].expand(self.history_frames - history.shape[0], -1, -1)
            history = torch.cat([pad, history], dim=0)
        complex_data.update(
            {
                "history": history,
                "target": trajectory[target_index],
                "target_index": torch.tensor(target_index),
            }
        )
        if pose_delta is not None:
            pose_target_index = target_index - 1
            first_pose = max(0, pose_target_index - self.history_frames)
            pose_history = pose_delta[first_pose:pose_target_index]
            valid_history = pose_valid[first_pose:pose_target_index]
            if pose_history.shape[0] < self.history_frames:
                padding = self.history_frames - pose_history.shape[0]
                pose_history = torch.cat(
                    [torch.zeros(padding, 6, dtype=pose_delta.dtype), pose_history]
                )
                valid_history = torch.cat(
                    [torch.zeros(padding, dtype=torch.bool), valid_history]
                )
            pose_history = pose_history.clone()
            pose_history[~valid_history] = 0.0
            complex_data.update(
                {
                    "pocket_pose_history": pose_history,
                    "pocket_pose_history_valid": valid_history,
                    "pocket_pose_target": pose_delta[pose_target_index],
                    "pocket_pose_target_valid": pose_valid[pose_target_index],
                }
            )
        return complex_data


def collate_bindmd(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable ligand and pocket sizes into a dense batch."""

    batch_size = len(items)
    history_frames = items[0]["history"].shape[0]
    max_ligand = max(item["ligand_z"].shape[0] for item in items)
    max_pocket = max(item["pocket_ca"].shape[0] for item in items)

    history = torch.zeros(batch_size, history_frames, max_ligand, 3)
    target = torch.zeros(batch_size, max_ligand, 3)
    ligand_z = torch.zeros(batch_size, max_ligand, dtype=torch.long)
    ligand_mass = torch.zeros(batch_size, max_ligand)
    ligand_mask = torch.zeros(batch_size, max_ligand, dtype=torch.bool)
    pocket_n = torch.zeros(batch_size, max_pocket, 3)
    pocket_ca = torch.zeros(batch_size, max_pocket, 3)
    pocket_c = torch.zeros(batch_size, max_pocket, 3)
    pocket_residue = torch.zeros(batch_size, max_pocket, dtype=torch.long)
    pocket_mask = torch.zeros(batch_size, max_pocket, dtype=torch.bool)
    target_index = torch.zeros(batch_size, dtype=torch.long)
    has_topology = "torsion_bond" in items[0]
    has_pose = "pocket_pose_history" in items[0]
    if has_pose:
        pocket_pose_history = torch.zeros(batch_size, history_frames, 6)
        pocket_pose_history_valid = torch.zeros(
            batch_size, history_frames, dtype=torch.bool
        )
        pocket_pose_target = torch.zeros(batch_size, 6)
        pocket_pose_target_valid = torch.zeros(batch_size, dtype=torch.bool)
    if has_topology:
        max_bonds = max(item["bond_index"].shape[0] for item in items)
        max_torsions = max(item["torsion_bond"].shape[0] for item in items)
        bond_index = torch.zeros(batch_size, max_bonds, 2, dtype=torch.long)
        bond_mask = torch.zeros(batch_size, max_bonds, dtype=torch.bool)
        torsion_bond = torch.zeros(batch_size, max_torsions, 2, dtype=torch.long)
        torsion_quad = torch.zeros(batch_size, max_torsions, 4, dtype=torch.long)
        torsion_rotate_mask = torch.zeros(
            batch_size, max_torsions, max_ligand, dtype=torch.bool
        )
        torsion_mask = torch.zeros(batch_size, max_torsions, dtype=torch.bool)
        torsion_root = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(items):
        n_ligand = item["ligand_z"].shape[0]
        n_pocket = item["pocket_ca"].shape[0]
        history[i, :, :n_ligand] = item["history"]
        target[i, :n_ligand] = item["target"]
        ligand_z[i, :n_ligand] = item["ligand_z"]
        ligand_mass[i, :n_ligand] = item["ligand_mass"]
        ligand_mask[i, :n_ligand] = True
        pocket_n[i, :n_pocket] = item["pocket_n"]
        pocket_ca[i, :n_pocket] = item["pocket_ca"]
        pocket_c[i, :n_pocket] = item["pocket_c"]
        pocket_residue[i, :n_pocket] = item["pocket_residue"]
        pocket_mask[i, :n_pocket] = True
        target_index[i] = item["target_index"]
        if has_pose:
            pocket_pose_history[i] = item["pocket_pose_history"]
            pocket_pose_history_valid[i] = item["pocket_pose_history_valid"]
            pocket_pose_target[i] = item["pocket_pose_target"]
            pocket_pose_target_valid[i] = item["pocket_pose_target_valid"]
        if has_topology:
            n_bonds = item["bond_index"].shape[0]
            n_torsions = item["torsion_bond"].shape[0]
            bond_index[i, :n_bonds] = item["bond_index"]
            bond_mask[i, :n_bonds] = True
            torsion_bond[i, :n_torsions] = item["torsion_bond"]
            torsion_quad[i, :n_torsions] = item["torsion_quad"]
            torsion_rotate_mask[
                i, :n_torsions, :n_ligand
            ] = item["torsion_rotate_mask"]
            torsion_mask[i, :n_torsions] = True
            torsion_root[i] = item["torsion_root"]

    result = {
        "history": history,
        "target": target,
        "ligand_z": ligand_z,
        "ligand_mass": ligand_mass,
        "ligand_mask": ligand_mask,
        "pocket_n": pocket_n,
        "pocket_ca": pocket_ca,
        "pocket_c": pocket_c,
        "pocket_residue": pocket_residue,
        "pocket_mask": pocket_mask,
        "target_index": target_index,
    }
    if has_topology:
        result.update(
            {
                "bond_index": bond_index,
                "bond_mask": bond_mask,
                "torsion_bond": torsion_bond,
                "torsion_quad": torsion_quad,
                "torsion_rotate_mask": torsion_rotate_mask,
                "torsion_mask": torsion_mask,
                "torsion_root": torsion_root,
            }
        )
    if has_pose:
        result.update(
            {
                "pocket_pose_history": pocket_pose_history,
                "pocket_pose_history_valid": pocket_pose_history_valid,
                "pocket_pose_target": pocket_pose_target,
                "pocket_pose_target_valid": pocket_pose_target_valid,
            }
        )
    return result
