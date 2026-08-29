from __future__ import annotations

import torch

from bindmd.models import BindMD
from conftest import synthetic_batch


def tiny_model() -> BindMD:
    return BindMD(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_rbf=8,
        rbf_max=10.0,
        dropout=0.0,
        diffusion_steps=8,
        diffusion_schedule="cosine",
        max_displacement=5.0,
    )


def test_training_loss_and_backward() -> None:
    model = tiny_model()
    output = model.training_loss(synthetic_batch(), history_noise_max=0.0)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_sampling_shapes_and_padding() -> None:
    model = tiny_model().eval()
    batch = synthetic_batch()
    schedule = model.diffusion.sampling_timesteps(4, torch.device("cpu"))
    assert torch.all(schedule[:-1] > schedule[1:])
    assert model.diffusion.alpha_bar[-1] < 1e-4
    result = model.rollout(batch, frames=2, ddim_steps=2)
    assert result.shape == (2, 2, 5, 3)
    assert torch.equal(result[1, :, -1], torch.zeros_like(result[1, :, -1]))


def test_rotation_translation_equivariance() -> None:
    model = tiny_model().eval()
    batch = synthetic_batch(batch_size=1)
    noisy_delta = torch.randn(1, 5, 3)
    timestep = torch.tensor([4])
    first = model.denoise(
        history=batch["history"],
        noisy_delta=noisy_delta,
        timestep=timestep,
        **model._condition(batch),
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([2.0, -4.0, 1.5])
    transformed = dict(batch)
    for key in ("history", "pocket_n", "pocket_ca", "pocket_c"):
        transformed[key] = batch[key] @ rotation.T + translation
    second = model.denoise(
        history=transformed["history"],
        noisy_delta=noisy_delta @ rotation.T,
        timestep=timestep,
        **model._condition(transformed),
    )
    torch.testing.assert_close(second, first @ rotation.T, atol=2e-5, rtol=2e-5)


def test_batched_sampling_matches_individual_generators() -> None:
    model = tiny_model().eval()
    batch = synthetic_batch()
    generators = [torch.Generator().manual_seed(101 + index) for index in range(2)]
    batched = model.rollout(batch, frames=2, ddim_steps=2, generators=generators)
    individual = []
    for index in range(2):
        item = {key: value[index : index + 1] for key, value in batch.items()}
        generator = torch.Generator().manual_seed(101 + index)
        individual.append(
            model.rollout(item, frames=2, ddim_steps=2, generator=generator)[0]
        )
    torch.testing.assert_close(batched, torch.stack(individual), atol=2e-5, rtol=2e-5)
