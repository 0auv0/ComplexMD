from __future__ import annotations

import torch

from bindmd.models import FlowBindMD
from conftest import synthetic_batch


def tiny_flow(internal_deformation_scale: float = 1.0) -> FlowBindMD:
    return FlowBindMD(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_rbf=8,
        rbf_max=10.0,
        dropout=0.0,
        diffusion_steps=8,
        max_displacement=5.0,
        flow_base_scale=1.0,
        flow_solver="heun",
        internal_deformation_scale=internal_deformation_scale,
    )


def test_flow_training_loss_and_backward() -> None:
    model = tiny_flow()
    output = model.training_loss(synthetic_batch(), history_noise_max=0.0)
    assert set(output) == {"loss", "flow_loss", "pair_loss"}
    assert all(torch.isfinite(value) for value in output.values())
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_flow_rollout_shapes_finite_and_padding() -> None:
    model = tiny_flow().eval()
    batch = synthetic_batch()
    generators = [torch.Generator().manual_seed(31 + index) for index in range(2)]
    result = model.rollout(batch, frames=2, ddim_steps=3, generators=generators)
    assert result.shape == (2, 2, 5, 3)
    assert torch.isfinite(result).all()
    assert torch.equal(result[1, :, -1], torch.zeros_like(result[1, :, -1]))
    displacement = result[:, 0] - batch["history"][:, -1]
    assert displacement.norm(dim=-1).max() <= 5.00001


def test_rigid_projection_preserves_reference_internal_distances() -> None:
    reference = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.2, 0.9, 0.0], [0.0, 0.0, 0.0]]]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    proposal = reference @ rotation + torch.tensor([[[2.0, -0.5, 0.3]]])
    proposal[:, 1] += torch.tensor([0.4, -0.2, 0.1])
    mask = torch.tensor([[True, True, True, False]])

    projected = FlowBindMD._blend_rigid_and_internal(
        reference, proposal, mask, internal_scale=0.0
    )
    valid = mask[0]
    assert torch.allclose(
        torch.cdist(projected[0, valid], projected[0, valid]),
        torch.cdist(reference[0, valid], reference[0, valid]),
        atol=1e-5,
    )
    assert torch.allclose(
        projected[0, valid].mean(dim=0), proposal[0, valid].mean(dim=0), atol=1e-5
    )
    assert torch.equal(projected[0, ~valid], torch.zeros_like(projected[0, ~valid]))


def test_zero_internal_scale_reproduces_exact_rigid_proposal() -> None:
    reference = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([[[2.5, -1.2, 0.7]]])
    mask = torch.tensor([[True, True, True, True, False]])
    proposal = (reference @ rotation + translation) * mask.unsqueeze(-1)

    projected = FlowBindMD._blend_rigid_and_internal(
        reference, proposal, mask, internal_scale=0.0
    )

    assert torch.allclose(projected[mask], proposal[mask], atol=1e-5)
    assert torch.equal(projected[~mask], torch.zeros_like(projected[~mask]))


def test_internal_scale_one_is_identity() -> None:
    batch = synthetic_batch()
    reference = batch["history"][:, -1]
    proposal = reference + torch.randn_like(reference) * batch["ligand_mask"].unsqueeze(-1)
    output = FlowBindMD._blend_rigid_and_internal(
        reference, proposal, batch["ligand_mask"], internal_scale=1.0
    )
    assert torch.equal(output, proposal * batch["ligand_mask"].unsqueeze(-1))


def test_zero_base_scale_is_deterministic_zero() -> None:
    model = tiny_flow()
    model.flow_base_scale = 0.0
    batch = synthetic_batch()
    base = model._sample_base(
        batch["history"],
        batch["ligand_mask"],
        generator=torch.Generator().manual_seed(9),
        generators=None,
    )
    assert torch.equal(base, torch.zeros_like(base))


def test_zero_internal_scale_rollout_preserves_initial_geometry() -> None:
    model = tiny_flow(internal_deformation_scale=0.0).eval()
    batch = synthetic_batch()
    generators = [torch.Generator().manual_seed(71 + index) for index in range(2)]
    result = model.rollout(batch, frames=3, ddim_steps=2, generators=generators)
    for sample in range(result.shape[0]):
        valid = batch["ligand_mask"][sample]
        reference_distance = torch.cdist(
            batch["history"][sample, -1, valid], batch["history"][sample, -1, valid]
        )
        for frame in result[sample]:
            assert torch.allclose(
                torch.cdist(frame[valid], frame[valid]), reference_distance, atol=2e-4
            )
