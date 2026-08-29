"""Conditional rectified flow for direct protein-conditioned ligand displacements."""

from __future__ import annotations

import torch

from bindmd.models.bindmd import BindMD


class FlowBindMD(BindMD):
    """BindMD with conditional flow matching instead of DDPM/DDIM.

    The probability path is the straight interpolation
    ``x_t = (1 - t) * x_0 + t * x_1`` between Gaussian displacement noise and
    the observed next-frame displacement. The network predicts its velocity
    ``x_1 - x_0`` and inference integrates that field with Euler or Heun.
    """

    def __init__(
        self,
        *,
        flow_base_scale: float = 1.0,
        flow_time_scale: float = 1000.0,
        flow_solver: str = "heun",
        internal_deformation_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if flow_base_scale < 0:
            raise ValueError("flow_base_scale must be non-negative")
        if flow_solver not in {"euler", "heun"}:
            raise ValueError("flow_solver must be 'euler' or 'heun'")
        if not 0.0 <= internal_deformation_scale <= 1.0:
            raise ValueError("internal_deformation_scale must be in [0, 1]")
        self.flow_base_scale = float(flow_base_scale)
        self.flow_time_scale = float(flow_time_scale)
        self.flow_solver = flow_solver
        self.internal_deformation_scale = float(internal_deformation_scale)

    def _time_embedding_input(self, time: torch.Tensor) -> torch.Tensor:
        return time * self.flow_time_scale

    @staticmethod
    def _clip(delta: torch.Tensor, maximum: float) -> torch.Tensor:
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return delta * (maximum / norm).clamp_max(1.0)

    def _sample_base(
        self,
        history: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None,
        generators: list[torch.Generator] | None,
    ) -> torch.Tensor:
        if generators is not None:
            if len(generators) != history.shape[0]:
                raise ValueError("generators must match the batch size")
            base = torch.stack(
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
            base = torch.randn(
                history[:, -1].shape,
                dtype=history.dtype,
                device=history.device,
                generator=generator,
            )
        return base * self.flow_base_scale * mask.unsqueeze(-1)

    @staticmethod
    def _blend_rigid_and_internal(
        reference: torch.Tensor,
        proposal: torch.Tensor,
        mask: torch.Tensor,
        internal_scale: float,
    ) -> torch.Tensor:
        """Keep the proposal's rigid motion while damping internal deformation.

        Both structures already live in the protein-aligned canonical frame. A
        masked Kabsch fit maps the previous ligand frame onto the Flow proposal;
        the fitted structure is therefore a ligand-rigid motion relative to the
        pocket, not another protein coordinate transformation. The residual
        changes ligand internal distances and is scaled without using targets.
        """
        vector_mask = mask.unsqueeze(-1)
        if internal_scale >= 1.0:
            return proposal * vector_mask

        # SVD is more stable in float32 and is unsupported for fp16 on CUDA.
        reference_f = reference.float()
        proposal_f = proposal.float()
        weight = vector_mask.float()
        count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        reference_center = (reference_f * weight).sum(dim=1, keepdim=True) / count
        proposal_center = (proposal_f * weight).sum(dim=1, keepdim=True) / count
        reference_centered = (reference_f - reference_center) * weight
        proposal_centered = (proposal_f - proposal_center) * weight

        covariance = reference_centered.transpose(1, 2) @ proposal_centered
        left, _, right_t = torch.linalg.svd(covariance, full_matrices=False)
        raw_rotation = left @ right_t
        correction = torch.eye(
            3, device=reference.device, dtype=torch.float32
        ).expand(reference.shape[0], -1, -1).clone()
        correction[:, -1, -1] = torch.where(
            torch.linalg.det(raw_rotation) < 0.0,
            -torch.ones(reference.shape[0], device=reference.device),
            torch.ones(reference.shape[0], device=reference.device),
        )
        rotation = left @ correction @ right_t
        rigid = reference_centered @ rotation + proposal_center
        blended = rigid + internal_scale * (proposal_f - rigid)
        return blended.to(proposal.dtype) * vector_mask

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history_noise_max: float = 0.10,
        pair_loss_weight: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        history = batch["history"]
        mask = batch["ligand_mask"]
        vector_mask = mask.unsqueeze(-1)
        if history_noise_max > 0:
            scale = torch.rand(history.shape[0], 1, 1, 1, device=history.device)
            history = history + torch.randn_like(history) * scale * history_noise_max

        clean_delta = (batch["target"] - history[:, -1]) * vector_mask
        base = torch.randn_like(clean_delta) * self.flow_base_scale * vector_mask
        time = torch.rand(history.shape[0], device=history.device)
        time_view = time.view(-1, 1, 1)
        path = ((1.0 - time_view) * base + time_view * clean_delta) * vector_mask
        target_velocity = (clean_delta - base) * vector_mask
        predicted_velocity = self.denoise(
            history=history,
            noisy_delta=path,
            timestep=self._time_embedding_input(time),
            **self._condition(batch),
        )

        denominator = mask.sum().clamp_min(1) * 3
        flow_loss = (
            (predicted_velocity - target_velocity).square() * vector_mask
        ).sum() / denominator

        endpoint_delta = path + (1.0 - time_view) * predicted_velocity
        predicted_coordinates = history[:, -1] + endpoint_delta
        predicted_distance = torch.cdist(predicted_coordinates, predicted_coordinates)
        target_distance = torch.cdist(batch["target"], batch["target"])
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_loss = (
            (predicted_distance - target_distance).abs() * pair_mask
        ).sum() / pair_mask.sum().clamp_min(1)
        total = flow_loss + pair_loss_weight * pair_loss
        return {
            "loss": total,
            "flow_loss": flow_loss.detach(),
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
        if ddim_steps < 1:
            raise ValueError("flow integration requires at least one step")
        history = batch["history"]
        mask = batch["ligand_mask"]
        vector_mask = mask.unsqueeze(-1)
        delta = self._sample_base(history, mask, generator, generators)
        step_size = 1.0 / ddim_steps
        for step in range(ddim_steps):
            time = history.new_full((history.shape[0],), step / ddim_steps)
            velocity = self.denoise(
                history=history,
                noisy_delta=delta,
                timestep=self._time_embedding_input(time),
                **self._condition(batch),
            )
            if self.flow_solver == "heun":
                proposal = delta + step_size * velocity
                next_time = history.new_full(
                    (history.shape[0],), (step + 1) / ddim_steps
                )
                next_velocity = self.denoise(
                    history=history,
                    noisy_delta=proposal,
                    timestep=self._time_embedding_input(next_time),
                    **self._condition(batch),
                )
                delta = delta + 0.5 * step_size * (velocity + next_velocity)
            else:
                delta = delta + step_size * velocity
            delta = self._clip(delta, self.max_displacement) * vector_mask
        proposal = (history[:, -1] + delta) * vector_mask
        return self._blend_rigid_and_internal(
            history[:, -1],
            proposal,
            mask,
            self.internal_deformation_scale,
        )
