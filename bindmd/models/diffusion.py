"""A small DDPM/DDIM schedule for direct next-frame displacement generation."""

from __future__ import annotations

import math

import torch
from torch import nn


class GaussianDisplacementDiffusion(nn.Module):
    def __init__(self, steps: int = 100, schedule: str = "cosine"):
        super().__init__()
        if schedule == "cosine":
            grid = torch.linspace(0, steps, steps + 1)
            offset = 0.008
            alpha_bar_grid = torch.cos(
                ((grid / steps + offset) / (1.0 + offset)) * math.pi / 2
            ).square()
            alpha_bar_grid = alpha_bar_grid / alpha_bar_grid[0]
            beta = 1.0 - alpha_bar_grid[1:] / alpha_bar_grid[:-1]
            beta = beta.clamp(1e-5, 0.999)
        elif schedule == "linear":
            beta = torch.linspace(1e-4, 2e-2, steps)
        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule}")
        alpha = 1.0 - beta
        self.schedule = schedule
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))

    @property
    def steps(self) -> int:
        return int(self.beta.numel())

    def add_noise(
        self, clean: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bar[timestep].view(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def predict_clean(
        self, noisy: torch.Tensor, timestep: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bar[timestep].view(-1, 1, 1)
        return (noisy - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()

    def sampling_timesteps(self, ddim_steps: int, device: torch.device) -> torch.Tensor:
        count = min(ddim_steps, self.steps)
        # unique_consecutive preserves the reverse-time order; torch.unique sorts.
        return (
            torch.linspace(self.steps - 1, 0, count, device=device)
            .round()
            .long()
            .unique_consecutive()
        )

    def ddim_step(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        previous_timestep: int,
        predicted_noise: torch.Tensor,
        max_clean_norm: float | None = None,
    ) -> torch.Tensor:
        clean = self.predict_clean(noisy, timestep, predicted_noise)
        if max_clean_norm is not None:
            norm = clean.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            clean = clean * (max_clean_norm / norm).clamp_max(1.0)
        if previous_timestep < 0:
            return clean
        alpha_previous = self.alpha_bar[previous_timestep]
        return alpha_previous.sqrt() * clean + (1.0 - alpha_previous).sqrt() * predicted_noise
