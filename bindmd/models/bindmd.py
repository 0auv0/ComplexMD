"""BindMD: causal joint space-time diffusion without force integration."""

from __future__ import annotations

import torch
from torch import nn

from bindmd.models.diffusion import GaussianDisplacementDiffusion
from bindmd.models.layers import (
    JointSpaceTimeBlock,
    PocketEncoder,
    RBF,
    sinusoidal_embedding,
)


class EquivariantNoiseHead(nn.Module):
    """Turn invariant token features into an SE(3)-equivariant vector field."""

    def __init__(self, hidden_dim: int, num_rbf: int, rbf_max: float):
        super().__init__()
        self.rbf = RBF(num_rbf, rbf_max)
        self.self_coefficient = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.ligand_coefficient = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pocket_coefficient = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        current_token: torch.Tensor,
        pocket_token: torch.Tensor,
        current_coordinates: torch.Tensor,
        noisy_delta: torch.Tensor,
        ligand_mask: torch.Tensor,
        pocket_n: torch.Tensor,
        pocket_ca: torch.Tensor,
        pocket_c: torch.Tensor,
        pocket_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, atoms, hidden = current_token.shape
        relative = current_coordinates[:, :, None] - current_coordinates[:, None, :]
        distance = relative.norm(dim=-1).clamp_min(1e-6)
        direction = relative / distance.unsqueeze(-1)
        left = current_token[:, :, None, :].expand(-1, -1, atoms, -1)
        right = current_token[:, None, :, :].expand(-1, atoms, -1, -1)
        ligand_input = torch.cat([left, right, self.rbf(distance)], dim=-1)
        ligand_weight = self.ligand_coefficient(ligand_input).squeeze(-1)
        pair_mask = ligand_mask[:, :, None] & ligand_mask[:, None, :]
        pair_mask &= ~torch.eye(
            atoms, device=pair_mask.device, dtype=torch.bool
        ).unsqueeze(0)
        ligand_vector = (
            ligand_weight.unsqueeze(-1) * direction * pair_mask.unsqueeze(-1)
        ).sum(dim=2) / pair_mask.sum(dim=2, keepdim=True).clamp_min(1)

        sites = torch.stack([pocket_n, pocket_ca, pocket_c], dim=-2)
        pocket_relative = current_coordinates[:, :, None, None, :] - sites[:, None]
        pocket_distance = pocket_relative.norm(dim=-1).clamp_min(1e-6)
        pocket_direction = pocket_relative / pocket_distance.unsqueeze(-1)
        ligand_feature = current_token[:, :, None, :].expand(
            -1, -1, pocket_token.shape[1], -1
        )
        pocket_feature = pocket_token[:, None].expand(-1, atoms, -1, -1)
        pocket_rbf = self.rbf(pocket_distance).flatten(-2)
        pocket_input = torch.cat([ligand_feature, pocket_feature, pocket_rbf], dim=-1)
        pocket_weight = self.pocket_coefficient(pocket_input)
        pocket_vector = (
            pocket_weight.unsqueeze(-1)
            * pocket_direction
            * pocket_mask[:, None, :, None, None]
        ).sum(dim=(2, 3)) / (
            pocket_mask.sum(dim=1).view(batch, 1, 1).clamp_min(1) * 3
        )

        output = (
            self.self_coefficient(current_token) * noisy_delta
            + ligand_vector
            + pocket_vector
        )
        return output * ligand_mask.unsqueeze(-1)


class BindMD(nn.Module):
    """Autoregressive conditional diffusion for a protein-bound ligand.

    The protein is fixed conditioning context. The model predicts diffusion
    noise on the next coordinate displacement directly; it has no force head
    and no Newton/ODE/SDE integration loop.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 6,
        num_rbf: int = 32,
        rbf_max: float = 20.0,
        dropout: float = 0.1,
        diffusion_steps: int = 100,
        diffusion_schedule: str = "linear",
        max_displacement: float = 5.0,
        max_atomic_number: int = 118,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_displacement = max_displacement
        self.atom = nn.Embedding(max_atomic_number + 1, hidden_dim, padding_idx=0)
        self.mass = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.frame_type = nn.Embedding(2, hidden_dim)
        self.diffusion_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.pocket_encoder = PocketEncoder(hidden_dim, num_heads, dropout)
        self.blocks = nn.ModuleList(
            [
                JointSpaceTimeBlock(
                    hidden_dim, num_heads, num_rbf, rbf_max, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.noise_head = EquivariantNoiseHead(hidden_dim, num_rbf, rbf_max)
        self.diffusion = GaussianDisplacementDiffusion(
            diffusion_steps, schedule=diffusion_schedule
        )

    @staticmethod
    def _condition(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        names = (
            "ligand_z",
            "ligand_mass",
            "ligand_mask",
            "pocket_n",
            "pocket_ca",
            "pocket_c",
            "pocket_residue",
            "pocket_mask",
        )
        condition = {name: batch[name] for name in names}
        if "pocket_token" in batch:
            condition["pocket_token"] = batch["pocket_token"]
        return condition

    def denoise(
        self,
        *,
        history: torch.Tensor,
        noisy_delta: torch.Tensor,
        timestep: torch.Tensor,
        ligand_z: torch.Tensor,
        ligand_mass: torch.Tensor,
        ligand_mask: torch.Tensor,
        pocket_n: torch.Tensor,
        pocket_ca: torch.Tensor,
        pocket_c: torch.Tensor,
        pocket_residue: torch.Tensor,
        pocket_mask: torch.Tensor,
        pocket_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, history_frames, atoms, _ = history.shape
        current = history[:, -1] + noisy_delta
        coordinates = torch.cat([history, current[:, None]], dim=1)
        frames = history_frames + 1

        base = self.atom(ligand_z) + self.mass(
            torch.log1p(ligand_mass).unsqueeze(-1)
        )
        frame_ids = torch.arange(frames, device=history.device)
        frame_embedding = sinusoidal_embedding(frame_ids, self.hidden_dim)
        token = base[:, None] + frame_embedding[None, :, None]
        token = token + self.frame_type(
            (frame_ids == history_frames).long()
        )[None, :, None]
        diffusion_embedding = self.diffusion_time(
            sinusoidal_embedding(timestep, self.hidden_dim)
        )
        token[:, -1] = token[:, -1] + diffusion_embedding[:, None]

        valid = ligand_mask[:, None, :].expand(-1, frames, -1)
        token = token.reshape(batch, frames * atoms, self.hidden_dim)
        coordinate_flat = coordinates.reshape(batch, frames * atoms, 3)
        valid_flat = valid.reshape(batch, frames * atoms)
        time_ids = torch.arange(frames, device=history.device).repeat_interleave(atoms)
        atom_ids = torch.arange(atoms, device=history.device).repeat(frames)
        if pocket_token is None:
            pocket_token = self.pocket_encoder(
                pocket_n, pocket_ca, pocket_c, pocket_residue, pocket_mask
            )
        for block in self.blocks:
            token = block(
                token,
                coordinate_flat,
                valid_flat,
                time_ids,
                atom_ids,
                pocket_token,
                pocket_ca,
                pocket_mask,
            )
        current_token = self.final_norm(
            token.view(batch, frames, atoms, self.hidden_dim)[:, -1]
        )
        return self.noise_head(
            current_token,
            pocket_token,
            current,
            noisy_delta,
            ligand_mask,
            pocket_n,
            pocket_ca,
            pocket_c,
            pocket_mask,
        )

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history_noise_max: float = 0.10,
        pair_loss_weight: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        history = batch["history"]
        mask = batch["ligand_mask"]
        if history_noise_max > 0:
            scale = torch.rand(history.shape[0], 1, 1, 1, device=history.device)
            history = history + torch.randn_like(history) * scale * history_noise_max
        clean_delta = batch["target"] - history[:, -1]
        noise = torch.randn_like(clean_delta) * mask.unsqueeze(-1)
        timestep = torch.randint(
            0, self.diffusion.steps, (history.shape[0],), device=history.device
        )
        noisy_delta = self.diffusion.add_noise(clean_delta, timestep, noise)
        predicted_noise = self.denoise(
            history=history,
            noisy_delta=noisy_delta,
            timestep=timestep,
            **self._condition(batch),
        )
        denominator = mask.sum().clamp_min(1) * 3
        diffusion_loss = (
            (predicted_noise - noise) ** 2 * mask.unsqueeze(-1)
        ).sum() / denominator
        predicted_clean = self.diffusion.predict_clean(
            noisy_delta, timestep, predicted_noise
        ) + history[:, -1]
        predicted_distance = torch.cdist(predicted_clean, predicted_clean)
        target_distance = torch.cdist(batch["target"], batch["target"])
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_loss = (
            (predicted_distance - target_distance).abs() * pair_mask
        ).sum() / pair_mask.sum().clamp_min(1)
        total = diffusion_loss + pair_loss_weight * pair_loss
        return {
            "loss": total,
            "diffusion_loss": diffusion_loss.detach(),
            "pair_loss": pair_loss.detach(),
        }

    @torch.no_grad()
    def sample_next(
        self,
        batch: dict[str, torch.Tensor],
        *,
        ddim_steps: int = 10,
        generator: torch.Generator | None = None,
        generators: list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        history = batch["history"]
        mask = batch["ligand_mask"]
        if generators is not None:
            if len(generators) != history.shape[0]:
                raise ValueError("generators must match the batch size")
            noisy = torch.stack(
                [
                    torch.randn(
                        history[index, -1].shape,
                        dtype=history.dtype,
                        device=history.device,
                        generator=sample_generator,
                    )
                    for index, sample_generator in enumerate(generators)
                ]
            )
        else:
            noisy = torch.randn(
                history[:, -1].shape,
                dtype=history.dtype,
                device=history.device,
                generator=generator,
            )
        noisy = noisy * mask.unsqueeze(-1)
        schedule = self.diffusion.sampling_timesteps(ddim_steps, history.device)
        for index, scalar_t in enumerate(schedule):
            timestep = scalar_t.expand(history.shape[0])
            predicted_noise = self.denoise(
                history=history,
                noisy_delta=noisy,
                timestep=timestep,
                **self._condition(batch),
            )
            previous = int(schedule[index + 1]) if index + 1 < schedule.numel() else -1
            noisy = self.diffusion.ddim_step(
                noisy,
                timestep,
                previous,
                predicted_noise,
                max_clean_norm=self.max_displacement,
            )
            noisy = noisy * mask.unsqueeze(-1)
        return (history[:, -1] + noisy) * mask.unsqueeze(-1)

    @torch.no_grad()
    def rollout(
        self,
        batch: dict[str, torch.Tensor],
        frames: int,
        *,
        ddim_steps: int = 10,
        generator: torch.Generator | None = None,
        generators: list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        state = dict(batch)
        state["pocket_token"] = self.pocket_encoder(
            state["pocket_n"],
            state["pocket_ca"],
            state["pocket_c"],
            state["pocket_residue"],
            state["pocket_mask"],
        )
        predictions = []
        for _ in range(frames):
            next_frame = self.sample_next(
                state,
                ddim_steps=ddim_steps,
                generator=generator,
                generators=generators,
            )
            predictions.append(next_frame)
            state["history"] = torch.cat(
                [state["history"][:, 1:], next_frame[:, None]], dim=1
            )
        return torch.stack(predictions, dim=1)
