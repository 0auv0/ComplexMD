"""GOAI public-evaluation adapter and full-complex rigid reconstruction.

The learned ligand model runs in the frame-0 protein reference frame.  This
module keeps the complete frame-0 protein as a rigid template, estimates the
global protein pose from observed frames only, and writes predictions in the
original PDB atom order.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bindmd.data.alignment import (
    apply_rigid_transform,
    first_residue_frame,
    kabsch_transform,
    pose_deltas_from_alignment,
)
from bindmd.data.topology import (
    build_torsion_topology_from_connectivity,
    build_torsion_topology_from_smiles,
)


# Matches MISATO atoms_residue_map minus one, as used by alignment.py.
AMINO_ACID_INDEX = {
    "ACE": 0,
    "ALA": 1,
    "ARG": 2,
    "ASN": 3,
    "ASP": 4,
    "CYS": 5,
    "CYX": 6,
    "GLN": 7,
    "GLU": 8,
    "GLY": 9,
    "HIS": 10,
    "HID": 10,
    "HIE": 10,
    "HIP": 10,
    "ILE": 11,
    "LEU": 12,
    "LYS": 13,
    "MET": 14,
    "PHE": 15,
    "PRO": 16,
    "SER": 17,
    "THR": 18,
    "TRP": 19,
    "TYR": 20,
    "VAL": 21,
}


@dataclass
class GOAISystem:
    identifier: str
    tier: str
    meta: dict[str, Any]
    topology: Any
    observed_angstrom: torch.Tensor
    protein_indices: torch.Tensor
    ligand_indices: torch.Tensor
    ligand_heavy_indices: torch.Tensor
    other_indices: torch.Tensor
    atom_numbers: torch.Tensor
    atom_masses: torch.Tensor


@dataclass
class CanonicalGOAI:
    system: GOAISystem
    observed_canonical: torch.Tensor
    protein_template_canonical: torch.Tensor
    ligand_template_canonical: torch.Tensor
    ligand_heavy_local_indices: torch.Tensor
    pocket_n: torch.Tensor
    pocket_ca: torch.Tensor
    pocket_c: torch.Tensor
    pocket_residue: torch.Tensor
    alignment_rotation: torch.Tensor
    alignment_mobile_center: torch.Tensor
    alignment_reference_center: torch.Tensor
    canonical_origin: torch.Tensor
    canonical_basis: torch.Tensor


ELEMENTS = {
    "H": (1, 1.008),
    "C": (6, 12.011),
    "N": (7, 14.007),
    "O": (8, 15.999),
    "F": (9, 18.998),
    "NA": (11, 22.990),
    "MG": (12, 24.305),
    "P": (15, 30.974),
    "S": (16, 32.06),
    "CL": (17, 35.45),
    "K": (19, 39.098),
    "CA": (20, 40.078),
    "BR": (35, 79.904),
    "I": (53, 126.904),
}


def _residue_name(residue: Any) -> str:
    return str(getattr(residue, "name", getattr(residue, "resname", ""))).upper()


def _element_data(atom: Any) -> tuple[int, float]:
    try:
        element = getattr(atom, "element", None)
    except Exception:
        element = None
    if isinstance(element, str) and element.strip():
        symbol = element.strip().upper()
        if symbol in ELEMENTS:
            return ELEMENTS[symbol]
    if element is not None and getattr(element, "atomic_number", 0):
        return int(element.atomic_number), float(element.mass)
    name = str(atom.name).strip().upper()
    return ELEMENTS.get(name[:2], ELEMENTS.get(name[:1], (6, 12.011)))


def _load_coordinates_and_topology(pdb_path: Path, xtc_path: Path) -> tuple[Any, np.ndarray]:
    """Read XTC with mdtraj when available, otherwise MDAnalysis."""

    try:
        import mdtraj as md
    except ImportError:
        md = None
    if md is not None:
        trajectory = md.load(str(xtc_path), top=str(pdb_path))
        return trajectory.topology, np.asarray(trajectory.xyz, dtype=np.float32) * 10.0

    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise ImportError(
            "GOAI XTC I/O requires mdtraj or MDAnalysis"
        ) from exc
    universe = mda.Universe(str(pdb_path), str(xtc_path))
    coordinates = np.stack(
        [time_step.positions.copy() for time_step in universe.trajectory]
    ).astype(np.float32)
    return universe, coordinates


def load_goai_system(root: str | Path, tier: str, identifier: str) -> GOAISystem:
    """Load one public system and expose all coordinates in Angstrom."""

    sample_dir = Path(root) / tier / identifier
    meta = json.loads((sample_dir / "meta.json").read_text())
    pdb_path = sample_dir / f"{identifier}.pdb"
    xtc_path = sample_dir / f"{identifier}_obs.xtc"
    topology, coordinates = _load_coordinates_and_topology(pdb_path, xtc_path)
    if coordinates.shape[0] != int(meta["n_obs"]):
        raise ValueError(f"{identifier}: XTC frame count disagrees with meta.json")
    if coordinates.shape[1] != int(meta["n_atoms"]):
        raise ValueError(f"{identifier}: XTC atom count disagrees with meta.json")

    ligand_resname = str(meta["ligand_resname"]).upper()
    protein, ligand, other = [], [], []
    atom_numbers, atom_masses = [], []
    for atom in list(topology.atoms):
        residue_name = _residue_name(atom.residue)
        if residue_name == ligand_resname:
            ligand.append(atom.index)
        elif residue_name in AMINO_ACID_INDEX:
            protein.append(atom.index)
        else:
            other.append(atom.index)
        number, mass = _element_data(atom)
        atom_numbers.append(number)
        atom_masses.append(mass)

    atom_numbers_t = torch.tensor(atom_numbers, dtype=torch.long)
    ligand_t = torch.tensor(ligand, dtype=torch.long)
    ligand_heavy = ligand_t[atom_numbers_t[ligand_t] != 1]
    if not protein or ligand_heavy.numel() == 0:
        raise ValueError(f"{identifier}: missing protein or ligand heavy atoms")
    return GOAISystem(
        identifier=identifier,
        tier=tier,
        meta=meta,
        topology=topology,
        observed_angstrom=torch.as_tensor(coordinates, dtype=torch.float32),
        protein_indices=torch.tensor(protein, dtype=torch.long),
        ligand_indices=ligand_t,
        ligand_heavy_indices=ligand_heavy,
        other_indices=torch.tensor(other, dtype=torch.long),
        atom_numbers=atom_numbers_t,
        atom_masses=torch.tensor(atom_masses, dtype=torch.float32),
    )


def _protein_backbone_triplets(system: GOAISystem) -> tuple[torch.Tensor, torch.Tensor]:
    triplets, residue_types = [], []
    for residue in list(system.topology.residues):
        name = _residue_name(residue)
        if name not in AMINO_ACID_INDEX:
            continue
        named = {str(atom.name).strip().upper(): atom.index for atom in residue.atoms}
        if all(atom_name in named for atom_name in ("N", "CA", "C")):
            triplets.append([named["N"], named["CA"], named["C"]])
            residue_types.append(AMINO_ACID_INDEX[name])
    if not triplets:
        raise ValueError(f"{system.identifier}: no complete protein N/CA/C triplets")
    return torch.tensor(triplets, dtype=torch.long), torch.tensor(residue_types)


def canonicalize_goai_system(
    system: GOAISystem,
    *,
    pocket_cutoff: float = 12.0,
    max_pocket_residues: int = 128,
) -> CanonicalGOAI:
    """Align all observations to the frame-0 pocket and fix the protein template."""

    coordinates = system.observed_angstrom
    triplets, residue_types = _protein_backbone_triplets(system)
    frame0_ca = coordinates[0, triplets[:, 1]]
    frame0_ligand = coordinates[0, system.ligand_heavy_indices]
    distance = torch.cdist(frame0_ca, frame0_ligand).amin(dim=-1)
    pocket = torch.nonzero(distance <= pocket_cutoff, as_tuple=False).flatten()
    if pocket.numel() == 0:
        pocket = torch.argsort(distance)[: min(16, triplets.shape[0])]
    if pocket.numel() > max_pocket_residues:
        nearest = torch.argsort(distance[pocket])[:max_pocket_residues]
        pocket = pocket[nearest]
    pocket = pocket.sort().values
    pocket_triplets = triplets[pocket]

    fit_indices = pocket_triplets.reshape(-1)
    reference = coordinates[0, fit_indices]
    rotations, mobile_centers, aligned = [], [], []
    for frame in coordinates:
        rotation, mobile_center, reference_center = kabsch_transform(
            frame[fit_indices], reference
        )
        rotations.append(rotation)
        mobile_centers.append(mobile_center)
        aligned.append(
            apply_rigid_transform(frame, rotation, mobile_center, reference_center)
        )
    aligned_t = torch.stack(aligned)

    n0 = aligned_t[0, pocket_triplets[:, 0]]
    ca0 = aligned_t[0, pocket_triplets[:, 1]]
    c0 = aligned_t[0, pocket_triplets[:, 2]]
    origin, basis = first_residue_frame(n0, ca0, c0)
    canonical = (aligned_t - origin) @ basis
    ligand_lookup = {int(atom): index for index, atom in enumerate(system.ligand_indices)}
    heavy_local = torch.tensor(
        [ligand_lookup[int(atom)] for atom in system.ligand_heavy_indices],
        dtype=torch.long,
    )
    return CanonicalGOAI(
        system=system,
        observed_canonical=canonical,
        protein_template_canonical=canonical[0, system.protein_indices],
        ligand_template_canonical=canonical[-1, system.ligand_indices],
        ligand_heavy_local_indices=heavy_local,
        pocket_n=(n0 - origin) @ basis,
        pocket_ca=(ca0 - origin) @ basis,
        pocket_c=(c0 - origin) @ basis,
        pocket_residue=residue_types[pocket].long(),
        alignment_rotation=torch.stack(rotations),
        alignment_mobile_center=torch.stack(mobile_centers),
        alignment_reference_center=reference.mean(dim=0),
        canonical_origin=origin,
        canonical_basis=basis,
    )


def build_goai_model_batch(
    canonical: CanonicalGOAI,
    history_frames: int,
    *,
    topology_source: str = "auto",
    smiles: str | None = None,
    smiles_atom_order: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Construct the single-system batch expected by existing ComplexMD weights."""

    system = canonical.system
    history = canonical.observed_canonical[:, system.ligand_heavy_indices]
    history = history[-history_frames:]
    if history.shape[0] < history_frames:
        history = torch.cat(
            [history[:1].expand(history_frames - history.shape[0], -1, -1), history]
        )
    ligand_z = system.atom_numbers[system.ligand_heavy_indices]
    ligand_mass = system.atom_masses[system.ligand_heavy_indices]
    pose_delta, pose_valid = pose_deltas_from_alignment(
        canonical.alignment_rotation,
        canonical.alignment_mobile_center,
        canonical_basis=canonical.canonical_basis,
    )
    pose_history = pose_delta[-history_frames:]
    pose_history_valid = pose_valid[-history_frames:]
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
    result = {
        "history": history.unsqueeze(0),
        "ligand_z": ligand_z.unsqueeze(0),
        "ligand_mass": ligand_mass.unsqueeze(0),
        "ligand_mask": torch.ones(1, ligand_z.numel(), dtype=torch.bool),
        "pocket_n": canonical.pocket_n.unsqueeze(0),
        "pocket_ca": canonical.pocket_ca.unsqueeze(0),
        "pocket_c": canonical.pocket_c.unsqueeze(0),
        "pocket_residue": canonical.pocket_residue.unsqueeze(0),
        "pocket_mask": torch.ones(
            1, canonical.pocket_ca.shape[0], dtype=torch.bool
        ),
        "pocket_pose_history": pose_history.unsqueeze(0),
        "pocket_pose_history_valid": pose_history_valid.unsqueeze(0),
    }
    if topology_source not in {"none", "auto", "conect", "smiles"}:
        raise ValueError("topology_source must be none, auto, conect, or smiles")
    if topology_source != "none":
        supplied_smiles = smiles or system.meta.get("smiles")
        topology = None
        if topology_source in {"auto", "smiles"} and supplied_smiles:
            try:
                topology = build_torsion_topology_from_smiles(
                    str(supplied_smiles), model_atom_order=smiles_atom_order
                )
                if topology["rigid_fragment"].numel() != ligand_z.numel():
                    raise ValueError("SMILES and ligand heavy-atom counts disagree")
            except Exception:
                if topology_source == "smiles":
                    raise
                topology = None
        if topology is None and topology_source in {"auto", "conect"}:
            full_to_heavy = {
                int(atom): local
                for local, atom in enumerate(system.ligand_heavy_indices.tolist())
            }
            compact_bonds = []
            topology_bonds = getattr(system.topology, "bonds", [])
            topology_bonds = topology_bonds() if callable(topology_bonds) else topology_bonds
            for bond in list(topology_bonds):
                atoms = getattr(bond, "atoms", bond)
                if len(atoms) != 2:
                    continue
                left = int(getattr(atoms[0], "index", atoms[0]))
                right = int(getattr(atoms[1], "index", atoms[1]))
                if left in full_to_heavy and right in full_to_heavy:
                    compact_bonds.append((full_to_heavy[left], full_to_heavy[right]))
            topology = build_torsion_topology_from_connectivity(
                ligand_z.tolist(), history[-1], compact_bonds
            )
        if topology is not None:
            torsions = topology["torsion_bond"].shape[0]
            bonds = topology["bond_index"].shape[0]
            fragment_count = int(topology["rigid_fragment_count"])
            # QM.hdf5 supplies chemistry-aware hybridisation during MISATO
            # training. Anonymous GOAI inputs only guarantee PDB connectivity,
            # so use the embedding's reserved unknown/padding class instead of
            # requiring an external molecule lookup. The fragment mask is a
            # batching field and can be derived exactly from the inferred
            # fragment count.
            ligand_hybridisation = topology.get(
                "ligand_hybridisation",
                torch.zeros(ligand_z.numel(), dtype=torch.long),
            )
            result.update(
                {
                    "bond_index": topology["bond_index"].unsqueeze(0),
                    "bond_mask": torch.ones(1, bonds, dtype=torch.bool),
                    "torsion_bond": topology["torsion_bond"].unsqueeze(0),
                    "torsion_quad": topology["torsion_quad"].unsqueeze(0),
                    "torsion_rotate_mask": topology["torsion_rotate_mask"].unsqueeze(0),
                    "torsion_mask": torch.ones(1, torsions, dtype=torch.bool),
                    "torsion_root": topology["torsion_root"].unsqueeze(0),
                    "rigid_fragment": topology["rigid_fragment"].unsqueeze(0),
                    "rigid_fragment_count": topology["rigid_fragment_count"].unsqueeze(0),
                    "rigid_fragment_mask": torch.ones(
                        1, fragment_count, dtype=torch.bool
                    ),
                    "ligand_hybridisation": ligand_hybridisation.unsqueeze(0),
                }
            )
    return result


def rigid_project_ligand(
    canonical: CanonicalGOAI, predicted_heavy: torch.Tensor
) -> torch.Tensor:
    """Infer one ligand SE(3) pose from heavy atoms and move all ligand atoms."""

    template_all = canonical.ligand_template_canonical
    template_heavy = template_all[canonical.ligand_heavy_local_indices]
    projected = []
    for target_heavy in predicted_heavy:
        rotation, mobile_center, target_center = kabsch_transform(
            template_heavy, target_heavy
        )
        projected.append(
            apply_rigid_transform(template_all, rotation, mobile_center, target_center)
        )
    return torch.stack(projected)


def fragment_project_ligand(
    canonical: CanonicalGOAI,
    predicted_heavy: torch.Tensor,
    rigid_fragment: torch.Tensor,
) -> torch.Tensor:
    """Move ligand hydrogens with their nearest topology-defined rigid piece.

    Predicted heavy atoms are retained exactly. Each hydrogen follows the
    Kabsch transform of the rigid heavy-atom fragment to which its CONECT path
    belongs, avoiding the old whole-ligand projection that erased torsions.
    """

    rigid_fragment = rigid_fragment.long().cpu()
    if rigid_fragment.numel() != canonical.ligand_heavy_local_indices.numel():
        raise ValueError("rigid_fragment must label every ligand heavy atom")
    template_all = canonical.ligand_template_canonical
    heavy_local = canonical.ligand_heavy_local_indices
    full_global = canonical.system.ligand_indices.tolist()
    global_to_local = {int(value): index for index, value in enumerate(full_global)}
    full_labels = torch.full((len(full_global),), -1, dtype=torch.long)
    for heavy_index, local_index in enumerate(heavy_local.tolist()):
        full_labels[local_index] = rigid_fragment[heavy_index]

    adjacency = [set() for _ in full_global]
    topology_bonds = getattr(canonical.system.topology, "bonds", [])
    topology_bonds = topology_bonds() if callable(topology_bonds) else topology_bonds
    for bond in list(topology_bonds):
        atoms = getattr(bond, "atoms", bond)
        if len(atoms) != 2:
            continue
        left = int(getattr(atoms[0], "index", atoms[0]))
        right = int(getattr(atoms[1], "index", atoms[1]))
        if left in global_to_local and right in global_to_local:
            left, right = global_to_local[left], global_to_local[right]
            adjacency[left].add(right)
            adjacency[right].add(left)
    queue = deque(int(index) for index in torch.nonzero(full_labels >= 0).flatten())
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if int(full_labels[neighbor]) < 0:
                full_labels[neighbor] = full_labels[node]
                queue.append(neighbor)
    for index in torch.nonzero(full_labels < 0).flatten().tolist():
        nearest = torch.cdist(
            template_all[index:index + 1], template_all[heavy_local]
        ).argmin()
        full_labels[index] = rigid_fragment[int(nearest)]

    output = []
    for target_heavy in predicted_heavy:
        frame = template_all.clone()
        for fragment in torch.unique(rigid_fragment).tolist():
            fragment_heavy = torch.nonzero(
                rigid_fragment == fragment, as_tuple=False
            ).flatten()
            fragment_all = torch.nonzero(
                full_labels == fragment, as_tuple=False
            ).flatten()
            source = template_all[heavy_local[fragment_heavy]]
            target = target_heavy[fragment_heavy]
            if fragment_heavy.numel() == 1:
                moved = template_all[fragment_all] + (target[0] - source[0])
            else:
                rotation, mobile_center, target_center = kabsch_transform(source, target)
                moved = apply_rigid_transform(
                    template_all[fragment_all], rotation, mobile_center, target_center
                )
            frame[fragment_all] = moved
        frame[heavy_local] = target_heavy
        output.append(frame)
    return torch.stack(output)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = vector.new_zeros(())
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)


def _so3_log(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((torch.trace(rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vee = torch.stack(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if float(angle.abs()) < 1e-7:
        return 0.5 * vee
    sine = torch.sin(angle)
    if float(sine.abs()) < 1e-7:
        return 0.5 * vee
    return angle * vee / (2.0 * sine)


def _so3_exp(vector: torch.Tensor) -> torch.Tensor:
    angle = vector.norm()
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    if float(angle) < 1e-7:
        return identity + _skew(vector)
    axis_skew = _skew(vector / angle)
    return identity + torch.sin(angle) * axis_skew + (1.0 - torch.cos(angle)) * (
        axis_skew @ axis_skew
    )


def future_reference_poses(
    canonical: CanonicalGOAI,
    frames: int,
    *,
    mode: str = "hold_last",
    velocity_window: int = 4,
    max_translation_step: float = 2.0,
    max_rotation_step_deg: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate frame-0-reference-to-world poses from observations only."""

    world_rotation = (
        canonical.canonical_basis.transpose(-1, -2)
        @ canonical.alignment_rotation.transpose(-1, -2)
    )
    centers = canonical.alignment_mobile_center
    if mode == "hold_last" or world_rotation.shape[0] < 2:
        return (
            world_rotation[-1:].expand(frames, -1, -1).clone(),
            centers[-1:].expand(frames, -1).clone(),
        )
    if mode != "constant_velocity":
        raise ValueError("pose mode must be 'hold_last' or 'constant_velocity'")

    window = min(velocity_window, world_rotation.shape[0] - 1)
    center_delta = (centers[-window:] - centers[-window - 1:-1]).median(dim=0).values
    center_norm = center_delta.norm().clamp_min(1e-8)
    center_delta = center_delta * min(1.0, max_translation_step / float(center_norm))
    rotation_delta = []
    for previous, current in zip(
        world_rotation[-window - 1:-1], world_rotation[-window:]
    ):
        rotation_delta.append(_so3_log(previous.T @ current))
    angular = torch.stack(rotation_delta).median(dim=0).values
    maximum = math.radians(max_rotation_step_deg)
    angular_norm = angular.norm().clamp_min(1e-8)
    angular = angular * min(1.0, maximum / float(angular_norm))
    step_rotation = _so3_exp(angular)

    rotations, future_centers = [], []
    rotation = world_rotation[-1]
    center = centers[-1]
    for _ in range(frames):
        rotation = rotation @ step_rotation
        center = center + center_delta
        rotations.append(rotation)
        future_centers.append(center)
    return torch.stack(rotations), torch.stack(future_centers)


def restore_full_complex(
    canonical: CanonicalGOAI,
    predicted_ligand_canonical: torch.Tensor,
    *,
    pose_mode: str = "hold_last",
    world_rotation: torch.Tensor | None = None,
    world_center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restore protein, ligand, and other atoms in PDB order, in Angstrom."""

    frames = predicted_ligand_canonical.shape[0]
    if (world_rotation is None) != (world_center is None):
        raise ValueError("world_rotation and world_center must be provided together")
    if world_rotation is None:
        world_rotation, world_center = future_reference_poses(
            canonical, frames, mode=pose_mode
        )
    system = canonical.system
    output = system.observed_angstrom[-1:].expand(frames, -1, -1).clone()

    def restore(values: torch.Tensor) -> torch.Tensor:
        reference_center_canonical = (
            canonical.alignment_reference_center - canonical.canonical_origin
        ) @ canonical.canonical_basis
        return (
            (values - reference_center_canonical) @ world_rotation
            + world_center[:, None]
        )

    protein = canonical.protein_template_canonical.unsqueeze(0).expand(frames, -1, -1)
    output[:, system.protein_indices] = restore(protein)
    output[:, system.ligand_indices] = restore(predicted_ligand_canonical)
    # T4 ions and any other non-protein/non-ligand atoms persist from observation.
    return output


def write_predicted_xtc(
    path: str | Path, coordinates_angstrom: torch.Tensor, canonical: CanonicalGOAI
) -> None:
    """Write exactly n_pred full-complex frames using XTC's native nm units."""

    meta = canonical.system.meta
    time = (
        np.arange(coordinates_angstrom.shape[0], dtype=np.float32)
        + float(meta["n_obs"])
    ) * float(meta["dt_ps"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = coordinates_angstrom.detach().cpu().numpy()

    try:
        import mdtraj as md
    except ImportError:
        md = None
    if md is not None:
        trajectory = md.Trajectory(
            xyz=coordinates / 10.0,
            topology=canonical.system.topology,
            time=time,
        )
        trajectory.save_xtc(str(path))
        return

    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise ImportError("GOAI XTC I/O requires mdtraj or MDAnalysis") from exc
    universe = mda.Universe.empty(
        coordinates.shape[1], n_frames=1, trajectory=True
    )
    universe.trajectory.ts.dt = float(meta["dt_ps"])
    with mda.Writer(
        str(path), n_atoms=coordinates.shape[1], dt=float(meta["dt_ps"])
    ) as writer:
        for frame_index, frame in enumerate(coordinates):
            universe.atoms.positions = frame
            universe.trajectory.ts.frame = int(meta["n_obs"]) + frame_index
            universe.trajectory.ts.time = float(time[frame_index])
            writer.write(universe.atoms)
