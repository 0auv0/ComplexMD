import torch

from bindmd.data.topology import build_torsion_topology
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

