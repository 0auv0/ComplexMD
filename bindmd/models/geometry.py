"""Differentiable SE(3) and ligand-torsion geometry operations."""

from __future__ import annotations

import torch


def masked_center(coordinates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(coordinates.dtype)
    return (coordinates * weight).sum(dim=1, keepdim=True) / weight.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta2 = axis_angle.square().sum(dim=-1, keepdim=True)
    # Keep the derivative finite at the zero axis-angle initialization.  The
    # small-angle Taylor branch below supplies the exact limiting values.
    theta = theta2.clamp_min(1e-12).sqrt()
    small = theta2 < 1e-8
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2.square() / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-8),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-8),
    )
    cross = skew(axis_angle)
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    identity = identity.expand(*axis_angle.shape[:-1], 3, 3)
    return identity + a.unsqueeze(-1) * cross + b.unsqueeze(-1) * (cross @ cross)


def matrix_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    angle = torch.acos(cosine)
    vector = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    scale = angle / (2.0 * torch.sin(angle).clamp_min(1e-7))
    scale = torch.where(angle < 1e-4, 0.5 + angle.square() / 12.0, scale)
    return vector * scale.unsqueeze(-1)


def rotation_geodesic_angle(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    relative = first.transpose(-1, -2) @ second
    cosine = (
        (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
    ).clamp(-1.0, 1.0)
    # atan2 avoids acos' infinite derivative at an exactly identity residual.
    # The old acos form produced NaN gradients for zero-motion examples, which
    # caused AMP to skip every optimizer step when the pose head was zero-init.
    skew_vee = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * skew_vee.norm(dim=-1)
    return torch.atan2(sine, cosine)


def integrate_pose_deltas(
    initial_rotation: torch.Tensor,
    initial_center: torch.Tensor,
    pose_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate body-frame translation/rotation increments.

    Rotation matrices use the row-vector convention throughout ComplexMD.
    ``pose_delta[..., :3]`` is expressed in the preceding pocket frame.
    """

    squeeze = pose_delta.ndim == 2
    if squeeze:
        pose_delta = pose_delta.unsqueeze(0)
        initial_rotation = initial_rotation.unsqueeze(0)
        initial_center = initial_center.unsqueeze(0)
    rotation, center = initial_rotation, initial_center
    rotations, centers = [], []
    for step in range(pose_delta.shape[1]):
        local_translation = pose_delta[:, step, :3]
        local_rotation = axis_angle_to_matrix(pose_delta[:, step, 3:])
        center = center + torch.bmm(
            local_translation[:, None], rotation
        ).squeeze(1)
        rotation = rotation @ local_rotation
        rotations.append(rotation)
        centers.append(center)
    rotation_trajectory = torch.stack(rotations, dim=1)
    center_trajectory = torch.stack(centers, dim=1)
    if squeeze:
        return rotation_trajectory[0], center_trajectory[0]
    return rotation_trajectory, center_trajectory


def _gather_atoms(coordinates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        coordinates,
        1,
        indices.unsqueeze(-1).expand(-1, -1, 3),
    )


def apply_torsions(
    coordinates: torch.Tensor,
    torsion_bond: torch.Tensor,
    torsion_rotate_mask: torch.Tensor,
    torsion_mask: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    output = coordinates
    for index in range(torsion_bond.shape[1]):
        active = torsion_mask[:, index]
        if not bool(active.any()):
            continue
        left = torsion_bond[:, index, 0]
        right = torsion_bond[:, index, 1]
        pivot = _gather_atoms(output, left[:, None])[:, 0]
        endpoint = _gather_atoms(output, right[:, None])[:, 0]
        axis = torch.nn.functional.normalize(endpoint - pivot, dim=-1, eps=1e-8)
        rotation = axis_angle_to_matrix(axis * angles[:, index:index + 1])
        relative = output - pivot[:, None]
        rotated = relative @ rotation.transpose(-1, -2) + pivot[:, None]
        atom_mask = torsion_rotate_mask[:, index] & active[:, None]
        output = torch.where(atom_mask.unsqueeze(-1), rotated, output)
    return output


def apply_generalized_step(
    coordinates: torch.Tensor,
    ligand_mask: torch.Tensor,
    torsion_bond: torch.Tensor,
    torsion_rotate_mask: torch.Tensor,
    torsion_mask: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    torsion: torch.Tensor,
) -> torch.Tensor:
    output = apply_torsions(
        coordinates, torsion_bond, torsion_rotate_mask, torsion_mask, torsion
    )
    center = masked_center(output, ligand_mask)
    matrix = axis_angle_to_matrix(rotation)
    output = (output - center) @ matrix.transpose(-1, -2) + center
    output = output + translation[:, None]
    return output * ligand_mask.unsqueeze(-1)


def dihedral(coordinates: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
    points = [
        _gather_atoms(coordinates, quad[..., index]) for index in range(4)
    ]
    first, second, third, fourth = points
    b0 = first - second
    b1 = third - second
    b2 = fourth - third
    b1_unit = torch.nn.functional.normalize(b1, dim=-1, eps=1e-8)
    v = b0 - (b0 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1_unit, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def generalized_basis(
    coordinates: torch.Tensor,
    ligand_mask: torch.Tensor,
    torsion_bond: torch.Tensor,
    torsion_rotate_mask: torch.Tensor,
    torsion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, atoms, _ = coordinates.shape
    torsions = torsion_bond.shape[1]
    identity = torch.eye(3, device=coordinates.device, dtype=coordinates.dtype)
    translation = identity.view(1, 1, 3, 3).expand(batch, atoms, -1, -1)
    relative = coordinates - masked_center(coordinates, ligand_mask)
    rotation = -skew(relative)

    if torsions:
        left = _gather_atoms(coordinates, torsion_bond[..., 0])
        right = _gather_atoms(coordinates, torsion_bond[..., 1])
        axis = torch.nn.functional.normalize(right - left, dim=-1, eps=1e-8)
        displacement = coordinates[:, None] - left[:, :, None]
        torsion = torch.cross(axis[:, :, None].expand_as(displacement), displacement, dim=-1)
        torsion = torsion * torsion_rotate_mask.unsqueeze(-1)
        torsion = torsion * torsion_mask[:, :, None, None]
        torsion = torsion.permute(0, 2, 3, 1)
    else:
        torsion = coordinates.new_zeros(batch, atoms, 3, 0)
    basis = torch.cat([translation, rotation, torsion], dim=-1)
    generalized_mask = torch.cat(
        [
            torch.ones(batch, 6, dtype=torch.bool, device=coordinates.device),
            torsion_mask,
        ],
        dim=-1,
    )
    basis = basis * ligand_mask[:, :, None, None] * generalized_mask[:, None, None]
    return basis, generalized_mask


def project_velocity(
    velocity: torch.Tensor,
    basis: torch.Tensor,
    generalized_mask: torch.Tensor,
    regularization: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    basis_f = basis.float()
    velocity_f = velocity.float()
    gram = torch.einsum("bncd,bnce->bde", basis_f, basis_f)
    rhs = torch.einsum("bncd,bnc->bd", basis_f, velocity_f)
    valid = generalized_mask.float()
    diagonal = regularization * valid + (1.0 - valid)
    gram = gram + torch.diag_embed(diagonal)
    coefficients = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1) * valid
    projected = torch.einsum("bncd,bd->bnc", basis_f, coefficients)
    return coefficients.to(velocity.dtype), projected.to(velocity.dtype)


def masked_rigid_parameters(
    reference: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_f, target_f = reference.float(), target.float()
    center_reference = masked_center(reference_f, mask)
    center_target = masked_center(target_f, mask)
    weight = mask.unsqueeze(-1).float()
    left = (reference_f - center_reference) * weight
    right = (target_f - center_target) * weight
    u, _, vh = torch.linalg.svd(left.transpose(1, 2) @ right, full_matrices=False)
    row_rotation = u @ vh
    correction = torch.eye(3, device=reference.device).expand(reference.shape[0], -1, -1).clone()
    correction[:, -1, -1] = torch.where(
        torch.linalg.det(row_rotation) < 0,
        -torch.ones(reference.shape[0], device=reference.device),
        torch.ones(reference.shape[0], device=reference.device),
    )
    row_rotation = u @ correction @ vh
    axis_angle = matrix_to_axis_angle(row_rotation.transpose(-1, -2))
    translation = (center_target - center_reference).squeeze(1)
    return translation.to(reference.dtype), axis_angle.to(reference.dtype)


def fit_generalized_target(
    reference: torch.Tensor,
    target: torch.Tensor,
    ligand_mask: torch.Tensor,
    torsion_bond: torch.Tensor,
    torsion_quad: torch.Tensor,
    torsion_rotate_mask: torch.Tensor,
    torsion_mask: torch.Tensor,
) -> torch.Tensor:
    if torsion_quad.shape[1]:
        reference_angle = dihedral(reference, torsion_quad)
        target_angle = dihedral(target, torsion_quad)
        torsion = torch.atan2(
            torch.sin(target_angle - reference_angle),
            torch.cos(target_angle - reference_angle),
        ) * torsion_mask
    else:
        torsion = reference.new_zeros(reference.shape[0], 0)
    internal = apply_torsions(
        reference, torsion_bond, torsion_rotate_mask, torsion_mask, torsion
    )
    translation, rotation = masked_rigid_parameters(internal, target, ligand_mask)
    return torch.cat([translation, rotation, torsion], dim=-1)
