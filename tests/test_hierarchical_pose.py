import math

import torch

from bindmd.data.alignment import pose_deltas_from_alignment
from bindmd.models.geometry import axis_angle_to_matrix, integrate_pose_deltas
from bindmd.models import build_model
from bindmd.models.hierarchical import HierarchicalPoseSE3TorsionFlowBindMD


def test_pose_delta_and_integration_are_inverse():
    angles = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.10], [0.02, -0.01, 0.16]]
    )
    world_rotation = axis_angle_to_matrix(angles)
    # alignment rotation is world rotation transpose when basis is identity.
    alignment_rotation = world_rotation.transpose(-1, -2)
    centers = torch.tensor([[1.0, 2.0, 3.0], [1.2, 1.9, 3.0], [1.5, 2.0, 3.1]])
    delta, valid = pose_deltas_from_alignment(alignment_rotation, centers)
    recovered_rotation, recovered_center = integrate_pose_deltas(
        world_rotation[0], centers[0], delta
    )
    assert valid.all()
    assert torch.allclose(recovered_rotation, world_rotation[1:], atol=1e-5)
    assert torch.allclose(recovered_center, centers[1:], atol=1e-5)


def test_pose_delta_respects_canonical_basis():
    basis = axis_angle_to_matrix(torch.tensor([0.3, -0.2, 0.1]))
    alignment = axis_angle_to_matrix(
        torch.tensor([[0.0, 0.0, 0.0], [0.03, -0.02, 0.01]])
    ).transpose(-1, -2)
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]])
    delta, valid = pose_deltas_from_alignment(
        alignment, centers, canonical_basis=basis
    )
    initial = basis.T @ alignment[0].T
    target = basis.T @ alignment[1].T
    recovered_rotation, recovered_center = integrate_pose_deltas(
        initial, centers[0], delta
    )
    assert bool(valid[0])
    assert torch.allclose(recovered_rotation[0], target, atol=1e-5)
    assert torch.allclose(recovered_center[0], centers[1], atol=1e-5)


def test_hierarchical_fragment_model_combines_both_heads():
    model = build_model(
        {
            "generation_method": "hierarchical_pose_se3_torsion_flow",
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "num_rbf": 8,
        }
    )
    assert isinstance(model, HierarchicalPoseSE3TorsionFlowBindMD)
    assert hasattr(model, "pose_head")
    assert hasattr(model, "torsion_target_scale")
    # SE3 geometry helpers must not shadow nn.Module._apply, which powers .to().
    model.to(torch.device("cpu"))
    clipped = model._clip_coefficients(torch.full((2, 9), 10.0))
    assert bool((clipped[:, :3].norm(dim=-1) <= model.translation_step_limit + 1e-6).all())
    assert bool((clipped[:, 3:6].norm(dim=-1) <= model.rotation_step_limit + 1e-6).all())


def test_hierarchical_rigid_fragment_model_uses_independent_window_aliases():
    model = build_model(
        {
            "generation_method": "hierarchical_pose_rigid_fragment_flow",
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "num_rbf": 8,
            "current_window_frames": 8,
            "historical_window_frames": 4,
            "fragment_current_window_frames": 8,
            "fragment_historical_window_frames": 4,
        }
    )
    assert model.pose_head.current_window_frames == 8
    assert model.pose_head.historical_window_frames == 4
    assert model.fragment_torsion_head.current_window_frames == 8
    assert model.fragment_torsion_head.historical_window_frames == 4
