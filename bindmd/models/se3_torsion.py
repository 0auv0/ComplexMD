"""Conditional rectified flow restricted to SE(3) and rotatable bonds."""

from __future__ import annotations

import math

import torch
from torch import nn

from bindmd.models.flow import FlowBindMD
from bindmd.models.geometry import (
    apply_generalized_step,
    dihedral,
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
        torsion_target_scale: float = 0.75,
        torsion_step_limit_deg: float = 30.0,
        translation_step_limit: float = 0.5,
        rotation_step_limit_deg: float = 5.0,
        endpoint_loss_weight: float = 0.10,
        projection_regularization: float = 1e-4,
        **kwargs,
    ):
        super().__init__(internal_deformation_scale=1.0, **kwargs)
        self.translation_base_scale = float(translation_base_scale)
        self.rotation_base_scale = float(rotation_base_scale)
        self.torsion_base_scale = float(torsion_base_scale)
        self.torsion_target_scale = float(torsion_target_scale)
        self.torsion_step_limit = math.radians(float(torsion_step_limit_deg))
        self.translation_step_limit = float(translation_step_limit)
        self.rotation_step_limit = math.radians(float(rotation_step_limit_deg))
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

    @staticmethod
    def _clip_vector(vector: torch.Tensor, maximum: float) -> torch.Tensor:
        norm = vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return vector * (maximum / norm).clamp_max(1.0)

    def _clip_coefficients(self, coefficient: torch.Tensor) -> torch.Tensor:
        coefficient = coefficient.clone()
        coefficient[:, :3] = self._clip_vector(
            coefficient[:, :3], self.translation_step_limit
        )
        coefficient[:, 3:6] = self._clip_vector(
            coefficient[:, 3:6], self.rotation_step_limit
        )
        coefficient[:, 6:] = coefficient[:, 6:].clamp(
            -self.torsion_step_limit, self.torsion_step_limit
        )
        return coefficient

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
    def _apply_generalized(
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
        # Thermal vibration and imperfect bond typing make raw frame-to-frame
        # dihedral differences noisy, especially when dozens of nested bonds
        # are eligible. A conservative target preserves useful internal motion
        # without forcing every numerical angle fluctuation into the rollout.
        target_generalized = target_generalized.clone()
        target_generalized[:, 6:] *= self.torsion_target_scale
        base = self._sample_generalized_base(reference, torsion_mask)
        time = torch.rand(history.shape[0], device=history.device, dtype=history.dtype)
        path_generalized = (
            (1.0 - time[:, None]) * base + time[:, None] * target_generalized
        )
        coordinates = self._apply_generalized(
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
        predicted_coordinates = self._apply_generalized(
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
        coordinates = self._apply_generalized(
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
            coefficient = self._clip_coefficients(coefficient)
            if self.flow_solver == "heun":
                proposal = self._apply_generalized(
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
                next_coefficient = self._clip_coefficients(next_coefficient)
                coefficient = 0.5 * (coefficient + next_coefficient)
            coordinates = self._apply_generalized(
                coordinates,
                coefficient * step_size,
                mask,
                torsion_bond,
                torsion_rotate_mask,
                torsion_mask,
            )
        return coordinates * mask.unsqueeze(-1)


class FragmentTorsionHead(nn.Module):
    """Predict relative rigid-fragment rotations from torsion history.

    Every rotatable bond joins two automatically detected rigid fragments.
    The head reads the wrapped dihedral history, endpoint atom chemistry,
    fragment sizes and protein-pocket context, and returns one angular
    velocity for each bond.  The downstream exact geometry operator rotates
    the complete child subtree, never individual atoms.
    """

    def __init__(
        self,
        hidden_dim: int,
        torsion_step_limit: float,
        *,
        current_window_frames: int = 6,
        historical_window_frames: int = 6,
        confidence_threshold: float = 0.55,
    ):
        super().__init__()
        if current_window_frames < 1 or historical_window_frames < 1:
            raise ValueError("both temporal windows must contain at least one frame")
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must lie strictly between 0 and 1")
        self.torsion_step_limit = float(torsion_step_limit)
        self.current_window_frames = int(current_window_frames)
        self.historical_window_frames = int(historical_window_frames)
        self.confidence_threshold = float(confidence_threshold)
        self.sequence_input = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Keep the old name for the current-window encoder so v2 checkpoints
        # remain valid initializers. The second GRU reads the preceding six
        # frames and enters as a zero-initialized residual.
        self.sequence_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.historical_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.historical_residual = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.trend_residual = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.hybridisation = nn.Embedding(8, hidden_dim, padding_idx=0)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 8, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.confidence_output = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start from the already verified atom-field projection.  The fragment
        # branch learns only the residual rotation supported by the data.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        nn.init.zeros_(self.historical_residual.weight)
        nn.init.zeros_(self.trend_residual.weight)
        nn.init.zeros_(self.confidence_output[-1].weight)
        nn.init.zeros_(self.confidence_output[-1].bias)

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            values,
            1,
            indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        history: torch.Tensor,
        coordinates: torch.Tensor,
        time: torch.Tensor,
        atom_token: torch.Tensor,
        pocket_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = (
            "rigid_fragment",
            "rigid_fragment_mask",
            "ligand_hybridisation",
            "torsion_bond",
            "torsion_quad",
            "torsion_rotate_mask",
            "torsion_mask",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"fragment torsion head requires: {missing}")
        torsion_mask = batch["torsion_mask"]
        torsions = torsion_mask.shape[1]
        if torsions == 0:
            empty = history.new_zeros(history.shape[0], 0)
            return empty, empty, empty

        angles = torch.stack(
            [dihedral(history[:, frame], batch["torsion_quad"])
             for frame in range(history.shape[1])],
            dim=1,
        )
        delta = torch.zeros_like(angles)
        delta[:, 1:] = torch.atan2(
            torch.sin(angles[:, 1:] - angles[:, :-1]),
            torch.cos(angles[:, 1:] - angles[:, :-1]),
        )
        sequence = torch.stack(
            [torch.sin(angles), torch.cos(angles), torch.sin(delta), torch.cos(delta)],
            dim=-1,
        )
        batch_size, frames = history.shape[:2]
        encoded = self.sequence_input(
            sequence.permute(0, 2, 1, 3).reshape(batch_size * torsions, frames, 4)
        )
        requested = self.current_window_frames + self.historical_window_frames
        encoded = encoded[:, -requested:]
        current_frames = min(self.current_window_frames, encoded.shape[1])
        current_sequence = encoded[:, -current_frames:]
        historical_sequence = encoded[:, :-current_frames]
        # Short synthetic/test histories do not have a preceding window. Use
        # a zero token; real train/eval inputs are padded to exactly 12 frames.
        if historical_sequence.shape[1] == 0:
            historical_sequence = torch.zeros_like(current_sequence[:, :1])
        elif historical_sequence.shape[1] > self.historical_window_frames:
            historical_sequence = historical_sequence[:, -self.historical_window_frames:]
        _, current_state = self.sequence_gru(current_sequence)
        _, historical_state = self.historical_gru(historical_sequence)
        current_state = current_state[-1]
        historical_state = historical_state[-1]
        state = (
            current_state
            + self.historical_residual(historical_state)
            + self.trend_residual(current_state - historical_state)
        ).reshape(batch_size, torsions, -1)

        endpoints = batch["torsion_bond"]
        left_token = self._gather(atom_token, endpoints[..., 0])
        right_token = self._gather(atom_token, endpoints[..., 1])
        endpoint_token = 0.5 * (left_token + right_token)
        hybridisation = self.hybridisation(
            batch["ligand_hybridisation"].clamp(0, 7)
        )
        hybrid_token = 0.5 * (
            self._gather(hybridisation, endpoints[..., 0])
            + self._gather(hybridisation, endpoints[..., 1])
        )

        pocket_weight = batch["pocket_mask"].unsqueeze(-1).to(pocket_token.dtype)
        pocket_state = (pocket_token * pocket_weight).sum(dim=1) / pocket_weight.sum(
            dim=1
        ).clamp_min(1.0)
        pocket_state = pocket_state[:, None].expand(-1, torsions, -1)

        fragment_count = batch["rigid_fragment_mask"].shape[1]
        fragment_sizes = history.new_zeros(batch_size, fragment_count)
        fragment_sizes.scatter_add_(
            1,
            batch["rigid_fragment"],
            batch["ligand_mask"].to(history.dtype),
        )
        left_fragment = torch.gather(
            batch["rigid_fragment"], 1, endpoints[..., 0]
        )
        right_fragment = torch.gather(
            batch["rigid_fragment"], 1, endpoints[..., 1]
        )
        atom_count = batch["ligand_mask"].sum(dim=1, keepdim=True).clamp_min(1)
        left_fraction = torch.gather(fragment_sizes, 1, left_fragment) / atom_count
        right_fraction = torch.gather(fragment_sizes, 1, right_fragment) / atom_count
        rotating_fraction = (
            batch["torsion_rotate_mask"].sum(dim=-1).to(history.dtype) / atom_count
        )

        left = self._gather(coordinates, endpoints[..., 0])
        right = self._gather(coordinates, endpoints[..., 1])
        midpoint = 0.5 * (left + right)
        pocket_distance = torch.cdist(midpoint, batch["pocket_ca"])
        pocket_distance = pocket_distance.masked_fill(
            ~batch["pocket_mask"][:, None], 1e6
        ).amin(dim=-1)
        time_feature = torch.stack(
            [time, torch.sin(math.pi * time), torch.cos(math.pi * time)], dim=-1
        )[:, None].expand(-1, torsions, -1)
        geometry = torch.cat(
            [
                left_fraction.unsqueeze(-1),
                right_fraction.unsqueeze(-1),
                rotating_fraction.unsqueeze(-1),
                (1.0 - rotating_fraction).unsqueeze(-1),
                (pocket_distance / 20.0).unsqueeze(-1),
                time_feature,
            ],
            dim=-1,
        )
        feature = torch.cat(
            [state, endpoint_token, hybrid_token, pocket_state, geometry], dim=-1
        )
        feature = torch.nan_to_num(feature, nan=0.0, posinf=1e4, neginf=-1e4)
        velocity = torch.tanh(
            torch.nan_to_num(
                self.output(feature).squeeze(-1), nan=0.0, posinf=20.0, neginf=-20.0
            )
        )
        confidence_logit = torch.nan_to_num(
            self.confidence_output(feature).squeeze(-1),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        confidence = torch.sigmoid(confidence_logit)
        return (
            velocity * self.torsion_step_limit * torsion_mask,
            confidence * torsion_mask,
            confidence_logit,
        )


class RigidFragmentSE3TorsionFlowBindMD(SE3TorsionFlowBindMD):
    """SE(3)+torsion Flow with an explicit rigid-fragment temporal branch."""

    def __init__(
        self,
        *,
        fragment_velocity_scale: float = 1.0,
        current_window_frames: int = 6,
        historical_window_frames: int = 6,
        fragment_current_window_frames: int | None = None,
        fragment_historical_window_frames: int | None = None,
        torsion_confidence_threshold: float = 0.55,
        torsion_confidence_target_deg: float = 5.0,
        torsion_confidence_loss_weight: float = 0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fragment_velocity_scale = float(fragment_velocity_scale)
        self.torsion_confidence_target = math.radians(
            float(torsion_confidence_target_deg)
        )
        self.torsion_confidence_loss_weight = float(
            torsion_confidence_loss_weight
        )
        self.fragment_torsion_head = FragmentTorsionHead(
            self.hidden_dim,
            self.torsion_step_limit,
            current_window_frames=(
                current_window_frames
                if fragment_current_window_frames is None
                else fragment_current_window_frames
            ),
            historical_window_frames=(
                historical_window_frames
                if fragment_historical_window_frames is None
                else fragment_historical_window_frames
            ),
            confidence_threshold=torsion_confidence_threshold,
        )
        self._last_fragment_confidence: torch.Tensor | None = None
        self._last_fragment_confidence_logit: torch.Tensor | None = None

    def _torsion_gate(self, confidence: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Classification alone calibrates confidence. Detaching here stops
            # the Flow objective from opening every gate simply to lower its
            # coordinate loss.
            return confidence.detach()
        return (
            confidence >= self.fragment_torsion_head.confidence_threshold
        ).to(confidence.dtype)

    def _field(
        self,
        batch: dict[str, torch.Tensor],
        history: torch.Tensor,
        coordinates: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficients, _, basis = super()._field(batch, history, coordinates, time)
        if coefficients.shape[1] == 6:
            return coefficients, torch.einsum("bncd,bd->bnc", basis, coefficients), basis
        pocket_token = batch.get("pocket_token")
        if pocket_token is None:
            pocket_token = self.pocket_encoder(
                batch["pocket_n"],
                batch["pocket_ca"],
                batch["pocket_c"],
                batch["pocket_residue"],
                batch["pocket_mask"],
            )
        atom_token = self.atom(batch["ligand_z"]) + self.mass(
            torch.log1p(batch["ligand_mass"]).unsqueeze(-1)
        )
        fragment_velocity, confidence, confidence_logit = self.fragment_torsion_head(
            batch,
            history,
            coordinates,
            time,
            atom_token,
            pocket_token,
        )
        self._last_fragment_confidence = confidence
        self._last_fragment_confidence_logit = confidence_logit
        coefficients = coefficients.clone()
        torsion_candidate = (
            coefficients[:, 6:]
            + self.fragment_velocity_scale * fragment_velocity
        )
        # During optimization the soft probability keeps gradients smooth.
        # Evaluation uses an exact hard gate: every torsion below the learned
        # confidence threshold is identically zero, including the projected
        # atom-field contribution (not merely the fragment residual).
        torsion_gate = self._torsion_gate(confidence)
        coefficients[:, 6:] = (
            torsion_candidate * torsion_gate * batch["torsion_mask"]
        )
        projected = torch.einsum("bncd,bd->bnc", basis, coefficients)
        return coefficients, projected, basis

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history_noise_max: float = 0.0,
        pair_loss_weight: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        output = super().training_loss(
            batch,
            history_noise_max=history_noise_max,
            pair_loss_weight=pair_loss_weight,
        )
        confidence = self._last_fragment_confidence
        confidence_logit = self._last_fragment_confidence_logit
        if (
            confidence is None
            or confidence_logit is None
            or confidence.shape[1] == 0
        ):
            zero = output["loss"].new_zeros(())
            return {
                **output,
                "loss": output["loss"] + zero,
                "torsion_confidence_loss": zero.detach(),
                "torsion_active_rate": zero.detach(),
                "torsion_confidence_mean": zero.detach(),
                "torsion_predicted_active_rate": zero.detach(),
            }
        reference_angle = dihedral(
            batch["history"][:, -1], batch["torsion_quad"]
        )
        target_angle = dihedral(batch["target"], batch["torsion_quad"])
        target_delta = torch.atan2(
            torch.sin(target_angle - reference_angle),
            torch.cos(target_angle - reference_angle),
        )
        target_active = (
            target_delta.abs() >= self.torsion_confidence_target
        ).to(confidence.dtype)
        torsion_mask = batch["torsion_mask"].to(confidence.dtype)
        valid_count = torsion_mask.float().sum().clamp_min(1.0)
        positive_rate = (
            target_active.float() * torsion_mask.float()
        ).sum() / valid_count
        positive_weight = 0.5 / positive_rate.clamp_min(1e-3)
        negative_weight = 0.5 / (1.0 - positive_rate).clamp_min(1e-3)
        class_weight = (
            target_active.float() * positive_weight
            + (1.0 - target_active.float()) * negative_weight
        )
        # Logit-form BCE is evaluated in FP32 for stable mixed-precision
        # training. Class balancing prevents the 70-80% active majority from
        # producing an all-confident gate.
        with torch.cuda.amp.autocast(enabled=False):
            confidence_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    confidence_logit.float(),
                    target_active.float(),
                    reduction="none",
                )
                * class_weight
                * torsion_mask.float()
            ).sum() / valid_count
        total = output["loss"] + self.torsion_confidence_loss_weight * confidence_loss
        active_rate = (target_active * torsion_mask).sum() / torsion_mask.sum().clamp_min(1.0)
        return {
            **output,
            "loss": total,
            "torsion_confidence_loss": confidence_loss.detach(),
            "torsion_active_rate": active_rate.detach(),
            "torsion_confidence_mean": (
                (confidence * torsion_mask).sum()
                / torsion_mask.sum().clamp_min(1.0)
            ).detach(),
            "torsion_predicted_active_rate": (
                (
                    confidence
                    >= self.fragment_torsion_head.confidence_threshold
                ).to(confidence.dtype)
                * torsion_mask
            ).sum().div(torsion_mask.sum().clamp_min(1.0)).detach(),
        }
