"""Conditional rectified flow restricted to SE(3) and rotatable bonds."""

from __future__ import annotations

import torch

from bindmd.models.flow import FlowBindMD
from bindmd.models.geometry import (
    apply_generalized_step,
    fit_generalized_target,
    generalized_basis,
    project_velocity,
)


class SE3TorsionFlowBindMD(FlowBindMD):
    """Flow Matching in ligand rigid-body and torsional degrees of freedom.

    The neural field remains atom-wise and SE(3)-equivariant, but an analytic
    Jacobian bottleneck projects it onto 6 rigid-body degrees of freedom plus
    the valid non-ring torsions supplied by the Amber topology. Integration is
    performed with exact Rodrigues rotations, so arbitrary bond stretching is
    impossible at inference time.
    """

    def __init__(
        self,
        *,
        translation_base_scale: float = 0.10,
        rotation_base_scale: float = 0.05,
        torsion_base_scale: float = 0.10,
        endpoint_loss_weight: float = 0.10,
        projection_regularization: float = 1e-4,
        **kwargs,
    ):
        super().__init__(internal_deformation_scale=1.0, **kwargs)
        self.translation_base_scale = float(translation_base_scale)
        self.rotation_base_scale = float(rotation_base_scale)
        self.torsion_base_scale = float(torsion_base_scale)
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.projection_regularization = float(projection_regularization)

    @staticmethod
    def _topology(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        required = (
            "torsion_bond",
            "torsion_quad",
            "torsion_rotate_mask",
            "torsion_mask",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"SE3TorsionFlowBindMD requires topology fields: {missing}")
        return tuple(batch[name] for name in required)

    @staticmethod
    def _random_like(
        shape: tuple[int, ...],
        reference: torch.Tensor,
        generator: torch.Generator | None,
        generators: list[torch.Generator] | None,
    ) -> torch.Tensor:
        if generators is None:
            return torch.randn(
                shape,
                device=reference.device,
                dtype=reference.dtype,
                generator=generator,
            )
        return torch.stack(
            [
                torch.randn(
                    shape[1:],
                    device=reference.device,
                    dtype=reference.dtype,
                    generator=sample_generator,
                )
                for sample_generator in generators
            ]
        )

    def _sample_generalized_base(
        self,
        reference: torch.Tensor,
        torsion_mask: torch.Tensor,
        generator: torch.Generator | None = None,
        generators: list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        batch = reference.shape[0]
        if generators is not None and len(generators) != batch:
            raise ValueError("generators must match the batch size")
        translation = self._random_like((batch, 3), reference, generator, generators)
        rotation = self._random_like((batch, 3), reference, generator, generators)
        torsion = self._random_like(
            tuple(torsion_mask.shape), reference, generator, generators
        )
        return torch.cat(
            [
                translation * self.translation_base_scale,
                rotation * self.rotation_base_scale,
                torsion * self.torsion_base_scale * torsion_mask,
            ],
            dim=-1,
        )

    @staticmethod
    def _apply(
        coordinates: torch.Tensor,
        generalized: torch.Tensor,
        ligand_mask: torch.Tensor,
        torsion_bond: torch.Tensor,
        torsion_rotate_mask: torch.Tensor,
        torsion_mask: torch.Tensor,
    ) -> torch.Tensor:
        return apply_generalized_step(
            coordinates,
            ligand_mask,
            torsion_bond,
            torsion_rotate_mask,
            torsion_mask,
            generalized[:, :3],
            generalized[:, 3:6],
            generalized[:, 6:],
        )

    def _field(
        self,
        batch: dict[str, torch.Tensor],
        history: torch.Tensor,
        coordinates: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = history[:, -1]
        raw_velocity = self.denoise(
            history=history,
            noisy_delta=(coordinates - reference) * batch["ligand_mask"].unsqueeze(-1),
            timestep=self._time_embedding_input(time),
            **self._condition(batch),
        )
        torsion_bond, _, torsion_rotate_mask, torsion_mask = self._topology(batch)
        basis, generalized_mask = generalized_basis(
            coordinates,
            batch["ligand_mask"],
            torsion_bond,
            torsion_rotate_mask,
            torsion_mask,
        )
        coefficients, projected = project_velocity(
            raw_velocity, basis, generalized_mask, self.projection_regularization
        )
        return coefficients, projected, basis

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history_noise_max: float = 0.0,
        pair_loss_weight: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        history = batch["history"]
        target = batch["target"]
        mask = batch["ligand_mask"]
        if history_noise_max > 0:
            # A shared translation augments the canonical frame without
            # corrupting ligand internal geometry.
            scale = torch.rand(history.shape[0], 1, 1, 1, device=history.device)
            shift = torch.randn(
                history.shape[0], 1, 1, 3, device=history.device, dtype=history.dtype
            )
            history = history + shift * scale * history_noise_max
            target = target + (shift * scale * history_noise_max)[:, 0]
        reference = history[:, -1]
        torsion_bond, torsion_quad, torsion_rotate_mask, torsion_mask = self._topology(batch)

        with torch.cuda.amp.autocast(enabled=False):
            target_generalized = fit_generalized_target(
                reference.float(),
                target.float(),
                mask,
                torsion_bond,
                torsion_quad,
                torsion_rotate_mask,
                torsion_mask,
            ).to(reference.dtype)
        base = self._sample_generalized_base(reference, torsion_mask)
        time = torch.rand(history.shape[0], device=history.device, dtype=history.dtype)
        path_generalized = (
            (1.0 - time[:, None]) * base + time[:, None] * target_generalized
        )
        coordinates = self._apply(
            reference,
            path_generalized,
            mask,
            torsion_bond,
            torsion_rotate_mask,
            torsion_mask,
        )
        coefficients, predicted_velocity, basis = self._field(
            batch, history, coordinates, time
        )
        target_coefficients = target_generalized - base
        target_velocity = torch.einsum(
            "bncd,bd->bnc", basis, target_coefficients
        )
        denominator = mask.sum().clamp_min(1) * 3
        flow_loss = (
            (predicted_velocity - target_velocity).square() * mask.unsqueeze(-1)
        ).sum() / denominator

        remaining = (1.0 - time[:, None]) * coefficients
        predicted_coordinates = self._apply(
            coordinates,
            remaining,
            mask,
            torsion_bond,
            torsion_rotate_mask,
            torsion_mask,
        )
        coordinate_loss = (
            (predicted_coordinates - target).square() * mask.unsqueeze(-1)
        ).sum() / denominator
        predicted_distance = torch.cdist(predicted_coordinates, predicted_coordinates)
        target_distance = torch.cdist(target, target)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_loss = (
            (predicted_distance - target_distance).abs() * pair_mask
        ).sum() / pair_mask.sum().clamp_min(1)
        total = (
            flow_loss
            + self.endpoint_loss_weight * coordinate_loss
            + pair_loss_weight * pair_loss
        )
        return {
            "loss": total,
            "flow_loss": flow_loss.detach(),
            "pair_loss": pair_loss.detach(),
            "coordinate_loss": coordinate_loss.detach(),
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
        reference = history[:, -1]
        mask = batch["ligand_mask"]
        torsion_bond, _, torsion_rotate_mask, torsion_mask = self._topology(batch)
        generalized = self._sample_generalized_base(
            reference, torsion_mask, generator, generators
        )
        coordinates = self._apply(
            reference,
            generalized,
            mask,
            torsion_bond,
            torsion_rotate_mask,
            torsion_mask,
        )
        step_size = 1.0 / ddim_steps
        for step in range(ddim_steps):
            time = history.new_full((history.shape[0],), step / ddim_steps)
            coefficient, _, _ = self._field(batch, history, coordinates, time)
            if self.flow_solver == "heun":
                proposal = self._apply(
                    coordinates,
                    coefficient * step_size,
                    mask,
                    torsion_bond,
                    torsion_rotate_mask,
                    torsion_mask,
                )
                next_time = history.new_full(
                    (history.shape[0],), (step + 1) / ddim_steps
                )
                next_coefficient, _, _ = self._field(
                    batch, history, proposal, next_time
                )
                coefficient = 0.5 * (coefficient + next_coefficient)
            coordinates = self._apply(
                coordinates,
                coefficient * step_size,
                mask,
                torsion_bond,
                torsion_rotate_mask,
                torsion_mask,
            )
        return coordinates * mask.unsqueeze(-1)
