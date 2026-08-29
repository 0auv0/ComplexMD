"""Hierarchical protein-pose and ligand-relative trajectory forecasting."""

from __future__ import annotations

import math

import torch
from torch import nn

from bindmd.models.flow import FlowBindMD
from bindmd.models.geometry import axis_angle_to_matrix, rotation_geodesic_angle


class PocketPoseHead(nn.Module):
    """Predict a bounded SE(3) residual around the hold-last protein pose."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        max_translation: float,
        max_rotation_deg: float,
    ):
        super().__init__()
        self.max_translation = float(max_translation)
        self.max_rotation = math.radians(max_rotation_deg)
        self.pose_input = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.ligand_input = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.pose_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ligand_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, 6),
        )
        # Zero residual is exactly hold-last and is the safest initialization.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        pose_history: torch.Tensor,
        pose_valid: torch.Tensor,
        ligand_history: torch.Tensor,
        ligand_mask: torch.Tensor,
        pocket_token: torch.Tensor,
        pocket_mask: torch.Tensor,
    ) -> torch.Tensor:
        pose_feature = self.pose_input(
            torch.cat([pose_history, pose_valid.unsqueeze(-1).float()], dim=-1)
        )
        _, pose_state = self.pose_gru(pose_feature)

        weight = ligand_mask[:, None, :, None].to(ligand_history.dtype)
        center = (ligand_history * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        center_delta = torch.zeros_like(center)
        center_delta[:, 1:] = center[:, 1:] - center[:, :-1]
        radial = torch.sqrt(
            (((ligand_history - center[:, :, None]) ** 2).sum(dim=-1)
             * ligand_mask[:, None].float()).sum(dim=-1)
            / ligand_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        )
        ligand_feature = self.ligand_input(
            torch.cat([center_delta, radial.unsqueeze(-1)], dim=-1)
        )
        _, ligand_state = self.ligand_gru(ligand_feature)

        pocket_weight = pocket_mask.unsqueeze(-1).to(pocket_token.dtype)
        pocket_state = (pocket_token * pocket_weight).sum(dim=1) / pocket_weight.sum(
            dim=1
        ).clamp_min(1.0)
        raw = torch.tanh(
            self.output(
                torch.cat([pose_state[-1], ligand_state[-1], pocket_state], dim=-1)
            )
        )
        scale = raw.new_tensor(
            [self.max_translation] * 3 + [self.max_rotation] * 3
        )
        return raw * scale


class HierarchicalPoseFlowBindMD(FlowBindMD):
    """Joint ligand-relative Flow Matching and protein-pocket SE(3) dynamics."""

    def __init__(
        self,
        *,
        pose_max_translation: float = 2.0,
        pose_max_rotation_deg: float = 5.0,
        pose_loss_weight: float = 1.0,
        ligand_loss_weight: float = 0.25,
        pose_rotation_loss_weight: float = 25.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pose_head = PocketPoseHead(
            self.hidden_dim,
            max_translation=pose_max_translation,
            max_rotation_deg=pose_max_rotation_deg,
        )
        self.pose_loss_weight = float(pose_loss_weight)
        self.ligand_loss_weight = float(ligand_loss_weight)
        self.pose_rotation_loss_weight = float(pose_rotation_loss_weight)

    def predict_pose_delta(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        required = (
            "pocket_pose_history",
            "pocket_pose_history_valid",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"hierarchical pose model requires: {missing}")
        pocket_token = batch.get("pocket_token")
        if pocket_token is None:
            pocket_token = self.pocket_encoder(
                batch["pocket_n"],
                batch["pocket_ca"],
                batch["pocket_c"],
                batch["pocket_residue"],
                batch["pocket_mask"],
            )
        return self.pose_head(
            batch["pocket_pose_history"],
            batch["pocket_pose_history_valid"],
            batch["history"],
            batch["ligand_mask"],
            pocket_token,
            batch["pocket_mask"],
        )

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        history_noise_max: float = 0.10,
        pair_loss_weight: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        ligand = super().training_loss(
            batch,
            history_noise_max=history_noise_max,
            pair_loss_weight=pair_loss_weight,
        )
        predicted = self.predict_pose_delta(batch)
        target = batch["pocket_pose_target"]
        valid = batch["pocket_pose_target_valid"].float()
        denominator = valid.sum().clamp_min(1.0)
        translation = (
            torch.nn.functional.smooth_l1_loss(
                predicted[:, :3], target[:, :3], reduction="none"
            ).sum(dim=-1) * valid
        ).sum() / denominator
        predicted_rotation = axis_angle_to_matrix(predicted[:, 3:])
        target_rotation = axis_angle_to_matrix(target[:, 3:])
        # Chordal SO(3) loss is locally equivalent to squared geodesic error,
        # but stays differentiable at the exactly identity residual produced by
        # the hold-last zero initialization.
        rotation_per_item = 0.5 * (
            predicted_rotation - target_rotation
        ).square().sum(dim=(-2, -1))
        rotation = (rotation_per_item * valid).sum() / denominator
        pose_loss = translation + self.pose_rotation_loss_weight * rotation
        total = (
            self.ligand_loss_weight * ligand["loss"]
            + self.pose_loss_weight * pose_loss
        )
        return {
            **ligand,
            "loss": total,
            "pose_loss": pose_loss.detach(),
            "pose_translation_loss": translation.detach(),
            "pose_rotation_loss": rotation.detach(),
        }

    @torch.no_grad()
    def rollout_complex(
        self,
        batch: dict[str, torch.Tensor],
        frames: int,
        *,
        ddim_steps: int = 10,
        generator: torch.Generator | None = None,
        generators: list[torch.Generator] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = dict(batch)
        state["pocket_token"] = self.pocket_encoder(
            state["pocket_n"],
            state["pocket_ca"],
            state["pocket_c"],
            state["pocket_residue"],
            state["pocket_mask"],
        )
        ligand_predictions, pose_predictions = [], []
        for _ in range(frames):
            next_ligand = super().sample_next(
                state,
                ddim_steps=ddim_steps,
                generator=generator,
                generators=generators,
            )
            next_pose = self.predict_pose_delta(state)
            ligand_predictions.append(next_ligand)
            pose_predictions.append(next_pose)
            state["history"] = torch.cat(
                [state["history"][:, 1:], next_ligand[:, None]], dim=1
            )
            state["pocket_pose_history"] = torch.cat(
                [state["pocket_pose_history"][:, 1:], next_pose[:, None]], dim=1
            )
            state["pocket_pose_history_valid"] = torch.cat(
                [
                    state["pocket_pose_history_valid"][:, 1:],
                    torch.ones(
                        next_pose.shape[0], 1, dtype=torch.bool, device=next_pose.device
                    ),
                ],
                dim=1,
            )
        return torch.stack(ligand_predictions, dim=1), torch.stack(
            pose_predictions, dim=1
        )
