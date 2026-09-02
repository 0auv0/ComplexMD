import torch
import h5py
import numpy as np

from bindmd.data.topology import (
    build_torsion_topology,
    build_torsion_topology_from_qm_group,
    build_torsion_topology_from_smiles,
)
from bindmd.models import build_model
from bindmd.models.hierarchical import HierarchicalPoseRigidFragmentFlowBindMD
from bindmd.models.se3_torsion import FragmentTorsionHead
from bindmd.models.geometry import (
    apply_generalized_step,
    fit_generalized_target,
    generalized_basis,
    project_velocity,
)


def chain_topology():
    return build_torsion_topology(
        [6, 6, 6, 6, 6],
        [(0, 1, 1.52), (1, 2, 1.52), (2, 3, 1.52), (3, 4, 1.52)],
    )


def batch_topology(topology):
    return (
        topology["torsion_bond"].unsqueeze(0),
        topology["torsion_quad"].unsqueeze(0),
        topology["torsion_rotate_mask"].unsqueeze(0),
        torch.ones(1, topology["torsion_bond"].shape[0], dtype=torch.bool),
    )


def test_ring_bonds_are_not_rotatable():
    topology = build_torsion_topology(
        [6] * 6,
        [(index, (index + 1) % 6, 1.40) for index in range(6)],
    )
    assert topology["torsion_bond"].shape == (0, 2)
    assert int(topology["rigid_fragment_count"]) == 1


def test_smiles_uses_chemistry_aware_rotatable_bonds():
    butane = build_torsion_topology_from_smiles("CCCC")
    assert butane["torsion_bond"].shape == (1, 2)
    assert int(butane["rigid_fragment_count"]) == 2

    acetamide = build_torsion_topology_from_smiles("CC(=O)NC")
    # The central amide C-N bond must stay within one rigid fragment.
    assert acetamide["torsion_bond"].shape == (0, 2)


def test_qm_bonds_and_hybridisation_build_rigid_fragments(tmp_path):
    path = tmp_path / "qm.hdf5"
    with h5py.File(path, "w") as handle:
        atoms = handle.create_group("TEST").create_group("atom_properties")
        atoms.create_dataset(
            "atom_names", data=np.asarray(["6", "6", "6", "6"], dtype="S2")
        )
        atoms.create_dataset(
            "atom_properties_names",
            data=np.asarray(["hybridisation"], dtype="S20"),
        )
        atoms.create_dataset(
            "atom_properties_values", data=np.asarray([[3], [3], [3], [3]])
        )
        bonds = np.asarray(
            [[0, 1, 1], [1, 0, 1], [1, 2, 1], [2, 1, 1],
             [2, 3, 1], [3, 2, 1]],
            dtype=float,
        )
        atoms.create_dataset("bonds", data=bonds)
        topology = build_torsion_topology_from_qm_group(handle["TEST"])
    assert topology["torsion_bond"].shape == (1, 2)
    assert int(topology["rigid_fragment_count"]) == 2
    assert topology["ligand_hybridisation"].tolist() == [3, 3, 3, 3]


def test_explicit_fragment_model_is_selectable():
    model = build_model(
        {
            "generation_method": "hierarchical_pose_rigid_fragment_flow",
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "num_rbf": 8,
        }
    )
    assert isinstance(model, HierarchicalPoseRigidFragmentFlowBindMD)
    assert hasattr(model, "fragment_torsion_head")


def test_twelve_frame_torsion_head_uses_six_plus_six_windows():
    topology = chain_topology()
    torsions = topology["torsion_bond"].shape[0]
    head = FragmentTorsionHead(
        hidden_dim=8,
        torsion_step_limit=0.5,
        current_window_frames=6,
        historical_window_frames=6,
    )
    reference = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.7, 0.8, 0.0],
         [3.8, 1.0, 0.9], [4.7, 1.8, 1.0]]
    )
    history = reference[None, None].expand(1, 12, -1, -1).clone()
    batch = {
        "rigid_fragment": topology["rigid_fragment"].unsqueeze(0),
        "rigid_fragment_mask": torch.ones(
            1, int(topology["rigid_fragment_count"]), dtype=torch.bool
        ),
        "ligand_hybridisation": torch.ones(1, 5, dtype=torch.long),
        "torsion_bond": topology["torsion_bond"].unsqueeze(0),
        "torsion_quad": topology["torsion_quad"].unsqueeze(0),
        "torsion_rotate_mask": topology["torsion_rotate_mask"].unsqueeze(0),
        "torsion_mask": torch.ones(1, torsions, dtype=torch.bool),
        "ligand_mask": torch.ones(1, 5, dtype=torch.bool),
        "pocket_mask": torch.ones(1, 3, dtype=torch.bool),
        "pocket_ca": torch.randn(1, 3, 3),
    }
    velocity, confidence, confidence_logit = head(
        batch,
        history,
        reference.unsqueeze(0),
        torch.tensor([0.5]),
        torch.zeros(1, 5, 8),
        torch.zeros(1, 3, 8),
    )
    assert velocity.shape == confidence.shape == (1, torsions)
    assert confidence_logit.shape == (1, torsions)
    assert head.current_window_frames == head.historical_window_frames == 6
    assert torch.isfinite(velocity).all()
    assert torch.isfinite(confidence).all()


def test_low_confidence_torsions_are_hard_zero_at_inference():
    model = build_model(
        {
            "generation_method": "hierarchical_pose_rigid_fragment_flow",
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "num_rbf": 8,
            "current_window_frames": 6,
            "historical_window_frames": 6,
            "torsion_confidence_threshold": 0.55,
        }
    )
    model.eval()
    gate = model._torsion_gate(torch.tensor([[0.10, 0.54, 0.55, 0.90]]))
    assert torch.equal(gate, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))


def test_exact_steps_preserve_bond_lengths():
    topology = chain_topology()
    bond, _, rotate, torsion_mask = batch_topology(topology)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.7, 0.8, 0.0],
          [3.8, 1.0, 0.9], [4.7, 1.8, 1.0]]]
    )
    mask = torch.ones(1, 5, dtype=torch.bool)
    before = torch.cdist(coordinates, coordinates)[0]
    output = apply_generalized_step(
        coordinates, mask, bond, rotate, torsion_mask,
        torch.tensor([[0.4, -0.2, 0.1]]),
        torch.tensor([[0.2, -0.1, 0.3]]),
        torch.full((1, bond.shape[1]), 0.7),
    )
    after = torch.cdist(output, output)[0]
    for left, right in topology["bond_index"]:
        assert torch.allclose(before[left, right], after[left, right], atol=2e-5)


def test_projection_is_identity_on_generalized_tangent():
    topology = chain_topology()
    bond, _, rotate, torsion_mask = batch_topology(topology)
    coordinates = torch.randn(1, 5, 3)
    mask = torch.ones(1, 5, dtype=torch.bool)
    basis, generalized_mask = generalized_basis(
        coordinates, mask, bond, rotate, torsion_mask
    )
    coefficients = torch.randn(1, basis.shape[-1])
    velocity = torch.einsum("bncd,bd->bnc", basis, coefficients)
    recovered, projected = project_velocity(
        velocity, basis, generalized_mask, regularization=1e-7
    )
    assert torch.allclose(projected, velocity, atol=2e-4, rtol=2e-4)
    assert torch.isfinite(recovered).all()


def test_projection_solve_is_fp32_under_cuda_autocast():
    if not torch.cuda.is_available():
        return
    topology = chain_topology()
    bond, _, rotate, torsion_mask = [value.cuda() for value in batch_topology(topology)]
    coordinates = torch.randn(1, 5, 3, device="cuda")
    mask = torch.ones(1, 5, dtype=torch.bool, device="cuda")
    with torch.cuda.amp.autocast(enabled=True):
        basis, generalized_mask = generalized_basis(
            coordinates, mask, bond, rotate, torsion_mask
        )
        velocity = torch.randn_like(coordinates)
        coefficients, projected = project_velocity(
            velocity, basis, generalized_mask
        )
    assert torch.isfinite(coefficients).all()
    assert torch.isfinite(projected).all()


def test_target_fit_reconstructs_representable_motion():
    topology = chain_topology()
    bond, quad, rotate, torsion_mask = batch_topology(topology)
    reference = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.7, 0.8, 0.0],
          [3.8, 1.0, 0.9], [4.7, 1.8, 1.0]]]
    )
    mask = torch.ones(1, 5, dtype=torch.bool)
    target = apply_generalized_step(
        reference, mask, bond, rotate, torsion_mask,
        torch.tensor([[0.3, -0.2, 0.1]]),
        torch.tensor([[0.05, -0.08, 0.12]]),
        torch.full((1, bond.shape[1]), 0.25),
    )
    fitted = fit_generalized_target(
        reference, target, mask, bond, quad, rotate, torsion_mask
    )
    rebuilt = apply_generalized_step(
        reference, mask, bond, rotate, torsion_mask,
        fitted[:, :3], fitted[:, 3:6], fitted[:, 6:],
    )
    assert torch.allclose(rebuilt, target, atol=2e-4, rtol=2e-4)
