import torch

from bindmd.data.alignment import (
    apply_rigid_transform,
    first_residue_frame,
    kabsch_transform,
)


def test_kabsch_recovers_a_proper_rigid_transform():
    torch.manual_seed(3)
    reference = torch.randn(20, 3)
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mobile = reference @ rotation + torch.tensor([8.0, -3.0, 4.0])
    fitted_rotation, mobile_center, reference_center = kabsch_transform(
        mobile, reference
    )
    fitted = apply_rigid_transform(
        mobile, fitted_rotation, mobile_center, reference_center
    )
    torch.testing.assert_close(fitted, reference, atol=2e-5, rtol=2e-5)
    assert torch.det(fitted_rotation) > 0.999


def test_first_residue_frame_uses_n_as_origin_and_fixed_axes():
    n = torch.tensor([[5.0, 2.0, -1.0], [8.0, 1.0, 0.0]])
    ca = torch.tensor([[6.0, 2.0, -1.0], [9.0, 1.0, 0.0]])
    c = torch.tensor([[6.0, 3.0, -1.0], [9.0, 2.0, 0.0]])
    origin, basis = first_residue_frame(n, ca, c)
    canonical_n = (n - origin) @ basis
    canonical_ca = (ca - origin) @ basis
    canonical_c = (c - origin) @ basis
    torch.testing.assert_close(canonical_n[0], torch.zeros(3))
    torch.testing.assert_close(canonical_ca[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(canonical_c[0], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(basis.T @ basis, torch.eye(3))
