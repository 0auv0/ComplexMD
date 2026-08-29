"""NeuralMD-compatible metrics plus competition-oriented trajectory proxies."""

from __future__ import annotations

import math

import torch


COVALENT_RADII = {
    1: 0.31, 5: 0.84, 6: 0.69, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05,
    17: 1.02, 19: 2.03, 20: 1.76, 34: 1.20, 35: 1.20, 53: 1.39,
}


def _value(x: torch.Tensor | float) -> float:
    return float(x.detach().cpu()) if isinstance(x, torch.Tensor) else float(x)


def _radii(z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [COVALENT_RADII.get(int(atom), 0.77) for atom in z.detach().cpu()],
        device=z.device,
        dtype=dtype,
    )


def _rg(trajectory: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    weight = mass / mass.sum().clamp_min(1e-8)
    center = (trajectory * weight[None, :, None]).sum(dim=1, keepdim=True)
    return torch.sqrt(
        (weight[None] * ((trajectory - center) ** 2).sum(dim=-1)).sum(dim=-1)
    )


def _rmsf(trajectory: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(
        ((trajectory - trajectory.mean(dim=0, keepdim=True)) ** 2)
        .sum(dim=-1)
        .mean(dim=0)
    )


def _js(a: torch.Tensor, b: torch.Tensor, bins: int = 32) -> torch.Tensor:
    lower, upper = torch.minimum(a.min(), b.min()), torch.maximum(a.max(), b.max())
    if _value(upper - lower) < 1e-12:
        return a.new_zeros(())
    pa = torch.histc(a.float(), bins, _value(lower), _value(upper)) + 1e-8
    pb = torch.histc(b.float(), bins, _value(lower), _value(upper)) + 1e-8
    pa, pb = pa / pa.sum(), pb / pb.sum()
    middle = 0.5 * (pa + pb)
    return 0.5 * ((pa * (pa / middle).log()).sum() + (pb * (pb / middle).log()).sum())


def _acf(series: torch.Tensor, lag: int) -> torch.Tensor:
    if series.numel() <= lag:
        return series.new_tensor(float("nan"))
    centered = series - series.mean()
    denominator = (centered**2).sum()
    if _value(denominator) < 1e-12:
        return series.new_zeros(())
    return (centered[:-lag] * centered[lag:]).sum() / denominator


def compute_all_metrics(
    *,
    target: torch.Tensor,
    pred: torch.Tensor,
    ligand_z: torch.Tensor,
    ligand_mass: torch.Tensor,
    protein_pos: torch.Tensor,
    protein_z: torch.Tensor,
    stability_threshold: float = 0.5,
    contact_cutoff: float = 4.5,
) -> dict[str, float]:
    """Return raw metrics; official competition normalization is not public."""
    if target.shape != pred.shape or target.ndim != 3:
        raise ValueError(f"Expected matching [T,N,3], got {target.shape}, {pred.shape}")
    target, pred = target.float(), pred.float()
    internal_target = torch.cdist(target, target)
    internal_pred = torch.cdist(pred, pred)
    pocket_target = torch.cdist(target, protein_pos.float())
    pocket_pred = torch.cdist(pred, protein_pos.float())
    atom_error = (pred - target).norm(dim=-1)
    frame_rmsd = ((pred - target).square().sum(dim=-1).mean(dim=-1)).sqrt()

    ligand_radius = _radii(ligand_z, pred.dtype)
    protein_radius = _radii(protein_z, pred.dtype)
    ligand_threshold = ligand_radius[:, None] + ligand_radius[None, :]
    binding_threshold = ligand_radius[:, None] + protein_radius[None, :]
    matching_per_frame = ((internal_pred - internal_target).square().mean(dim=(-2, -1))).sqrt()
    stability = (
        (internal_pred - internal_target).abs() <= stability_threshold
    ).float().mean(dim=(-2, -1)) * 100.0

    result = {
        # Exact reductions used by the released NeuralMD evaluator.
        "neuralmd_mae": _value((pred - target).abs().sum() / (target.shape[0] * target.shape[1])),
        "neuralmd_rmse": _value(atom_error.mean()),
        "neuralmd_matching": _value(matching_per_frame.mean()),
        "neuralmd_stability": _value(stability.mean()),
        "neuralmd_ligand_collision": _value(
            (internal_pred < ligand_threshold).float().mean() * 100.0
        ),
        "neuralmd_binding_collision": _value(
            (pocket_pred < binding_threshold).float().mean() * 100.0
        ),
        # Geo
        "geo_ligand_rmsd": _value(frame_rmsd.mean()),
        "geo_ligand_rmsd_last": _value(frame_rmsd[-1]),
        "geo_internal_distance_rmse": _value(
            ((internal_pred - internal_target).square().mean()).sqrt()
        ),
        "geo_backbone_pocket_distance_rmse": _value(
            ((pocket_pred - pocket_target).square().mean()).sqrt()
        ),
    }

    upper = torch.triu(torch.ones_like(internal_target[0], dtype=torch.bool), diagonal=1)
    inferred_bond = upper & (internal_target[0] <= 1.25 * ligand_threshold)
    inferred_nonbond = upper & ~inferred_bond
    result["phys_inferred_bond_count"] = float(inferred_bond.sum())
    result["phys_inferred_bond_length_rmse"] = _value(
        ((internal_pred[:, inferred_bond] - internal_target[:, inferred_bond]) ** 2)
        .mean()
        .sqrt()
    ) if inferred_bond.any() else float("nan")
    result["phys_ligand_clash_rate"] = _value(
        (internal_pred[:, inferred_nonbond] < 0.75 * ligand_threshold[inferred_nonbond])
        .float()
        .mean()
    ) if inferred_nonbond.any() else float("nan")
    result["phys_binding_clash_rate"] = _value(
        (pocket_pred < 0.75 * binding_threshold).float().mean()
    )

    target_rmsf, pred_rmsf = _rmsf(target), _rmsf(pred)
    target_rg, pred_rg = _rg(target, ligand_mass), _rg(pred, ligand_mass)
    target_contacts = pocket_target < contact_cutoff
    pred_contacts = pocket_pred < contact_cutoff
    result.update(
        {
            "dyn_rmsf_mae": _value((pred_rmsf - target_rmsf).abs().mean()),
            "dyn_rmsf_correlation": _value(
                torch.corrcoef(torch.stack([target_rmsf, pred_rmsf]))[0, 1]
            ) if target_rmsf.numel() > 1 else float("nan"),
            "dyn_rg_mae": _value((pred_rg - target_rg).abs().mean()),
            "dyn_rg_wasserstein": _value(
                (pred_rg.sort().values - target_rg.sort().values).abs().mean()
            ),
            "dyn_rg_js": _value(_js(target_rg, pred_rg)),
            "dyn_contact_occupancy_mae": _value(
                (
                    pred_contacts.float().mean(dim=0)
                    - target_contacts.float().mean(dim=0)
                ).abs().mean()
            ),
        }
    )
    target_count = target_contacts.float().sum(dim=(-2, -1))
    pred_count = pred_contacts.float().sum(dim=(-2, -1))
    for lag in (1, 5, 10):
        result[f"dyn_contact_acf_error_lag_{lag}"] = _value(
            (_acf(pred_count, lag) - _acf(target_count, lag)).abs()
        )

    bounds = [0, round(len(frame_rmsd) * 0.25), round(len(frame_rmsd) * 0.50),
              round(len(frame_rmsd) * 0.75), len(frame_rmsd)]
    for index in range(4):
        start, end = bounds[index], max(bounds[index + 1], bounds[index] + 1)
        result[f"stab_error_window_{index + 1}"] = _value(frame_rmsd[start:end].mean())
    time = torch.linspace(0, 1, len(frame_rmsd), device=pred.device)
    centered_time = time - time.mean()
    result["stab_error_growth_slope"] = _value(
        (centered_time * (frame_rmsd - frame_rmsd.mean())).sum()
        / centered_time.square().sum().clamp_min(1e-12)
    )
    step = (pred[1:] - pred[:-1]).norm(dim=-1)
    result.update(
        {
            "stab_finite_coordinate_rate": _value(torch.isfinite(pred).float().mean()),
            "stab_step_displacement_p95": _value(torch.quantile(step, 0.95)) if step.numel() else 0.0,
            "stab_frame_rmsd_max": _value(frame_rmsd.max()),
        }
    )
    return result


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")
