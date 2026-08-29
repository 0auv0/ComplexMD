from __future__ import annotations

import torch

from bindmd.evaluation.metrics import compute_all_metrics


def test_perfect_prediction_has_zero_geometry_error() -> None:
    torch.manual_seed(1)
    target = torch.randn(12, 4, 3)
    metrics = compute_all_metrics(
        target=target,
        pred=target.clone(),
        ligand_z=torch.tensor([6, 6, 7, 8]),
        ligand_mass=torch.tensor([12.0, 12.0, 14.0, 16.0]),
        protein_pos=torch.randn(9, 3) + 4,
        protein_z=torch.tensor([7, 6, 6] * 3),
    )
    assert metrics["neuralmd_mae"] == 0.0
    assert metrics["neuralmd_rmse"] == 0.0
    assert metrics["geo_ligand_rmsd"] == 0.0
    assert metrics["geo_internal_distance_rmse"] == 0.0

