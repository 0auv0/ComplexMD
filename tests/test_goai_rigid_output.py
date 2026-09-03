from __future__ import annotations

from types import SimpleNamespace

import torch

from bindmd.data.goai import (
    CanonicalGOAI,
    GOAISystem,
    build_goai_model_batch,
    fragment_project_ligand,
    future_reference_poses,
    restore_full_complex,
    rigid_project_ligand,
)


def synthetic_canonical() -> CanonicalGOAI:
    protein = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    ligand = torch.tensor(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
    )
    angle = torch.tensor(0.4)
    world_rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = torch.tensor([7.0, -2.0, 4.0])
    reference_center = protein.mean(dim=0)
    full_reference = torch.cat([protein, ligand], dim=0)
    observed_last = (full_reference - reference_center) @ world_rotation + center
    system = GOAISystem(
        identifier="T1-test",
        tier="T1",
        meta={"n_obs": 2, "n_pred": 2, "dt_ps": 80.0},
        topology=SimpleNamespace(
            bonds=[
                (SimpleNamespace(index=3), SimpleNamespace(index=4)),
                (SimpleNamespace(index=4), SimpleNamespace(index=5)),
                (SimpleNamespace(index=3), SimpleNamespace(index=6)),
            ]
        ),
        observed_angstrom=torch.stack([full_reference, observed_last]),
        protein_indices=torch.tensor([0, 1, 2]),
        ligand_indices=torch.tensor([3, 4, 5, 6]),
        ligand_heavy_indices=torch.tensor([3, 4, 5]),
        other_indices=torch.empty(0, dtype=torch.long),
        atom_numbers=torch.tensor([6, 6, 6, 6, 7, 8, 1]),
        atom_masses=torch.ones(7),
    )
    return CanonicalGOAI(
        system=system,
        observed_canonical=torch.stack([full_reference, full_reference]),
        protein_template_canonical=protein,
        ligand_template_canonical=ligand,
        ligand_heavy_local_indices=torch.tensor([0, 1, 2]),
        pocket_n=protein[:1],
        pocket_ca=protein[1:2],
        pocket_c=protein[2:3],
        pocket_residue=torch.tensor([1]),
        alignment_rotation=torch.stack([torch.eye(3), world_rotation.T]),
        alignment_mobile_center=torch.stack([reference_center, center]),
        alignment_reference_center=reference_center,
        canonical_origin=torch.zeros(3),
        canonical_basis=torch.eye(3),
    )


def test_rigid_projection_moves_hydrogen_with_heavy_atoms() -> None:
    canonical = synthetic_canonical()
    angle = torch.tensor(-0.3)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target_all = canonical.ligand_template_canonical @ rotation + torch.tensor(
        [1.0, 2.0, -1.0]
    )
    projected = rigid_project_ligand(
        canonical, target_all[canonical.ligand_heavy_local_indices].unsqueeze(0)
    )[0]
    torch.testing.assert_close(projected, target_all, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        torch.cdist(projected, projected),
        torch.cdist(canonical.ligand_template_canonical, canonical.ligand_template_canonical),
        atol=1e-5,
        rtol=1e-5,
    )


def test_fragment_projection_retains_internal_heavy_motion() -> None:
    canonical = synthetic_canonical()
    target_heavy = canonical.ligand_template_canonical[:3].clone()
    target_heavy[0] += torch.tensor([0.0, 0.0, 0.5])
    target_heavy[1:] += torch.tensor([0.3, -0.2, 0.0])
    projected = fragment_project_ligand(
        canonical, target_heavy.unsqueeze(0), torch.tensor([0, 1, 1])
    )[0]
    torch.testing.assert_close(projected[:3], target_heavy)
    # Hydrogen atom 3 is CONECT-attached to heavy atom 0 and follows fragment 0.
    torch.testing.assert_close(
        projected[3] - canonical.ligand_template_canonical[3],
        target_heavy[0] - canonical.ligand_template_canonical[0],
    )


def test_conect_batch_supplies_fragment_head_fields_without_qm_hdf5() -> None:
    canonical = synthetic_canonical()
    batch = build_goai_model_batch(
        canonical, history_frames=12, topology_source="conect"
    )
    assert batch["ligand_hybridisation"].shape == (1, 3)
    assert torch.count_nonzero(batch["ligand_hybridisation"]) == 0
    assert batch["rigid_fragment_mask"].shape == (
        1,
        int(batch["rigid_fragment_count"][0]),
    )
    assert batch["rigid_fragment_mask"].all()


def test_hold_last_restores_rigid_protein_and_atom_order() -> None:
    canonical = synthetic_canonical()
    ligand = canonical.ligand_template_canonical.unsqueeze(0).expand(2, -1, -1)
    restored = restore_full_complex(canonical, ligand, pose_mode="hold_last")
    expected = canonical.system.observed_angstrom[-1]
    torch.testing.assert_close(restored[0], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(restored[1], expected, atol=1e-5, rtol=1e-5)


def test_constant_velocity_pose_uses_only_observed_transforms() -> None:
    canonical = synthetic_canonical()
    rotations, centers = future_reference_poses(
        canonical,
        2,
        mode="constant_velocity",
        max_translation_step=20.0,
        max_rotation_step_deg=90.0,
    )
    assert rotations.shape == (2, 3, 3)
    expected_center = (
        2 * canonical.alignment_mobile_center[-1]
        - canonical.alignment_mobile_center[-2]
    )
    torch.testing.assert_close(centers[0], expected_center)
    torch.testing.assert_close(
        rotations[0].T @ rotations[0], torch.eye(3), atol=1e-5, rtol=1e-5
    )
