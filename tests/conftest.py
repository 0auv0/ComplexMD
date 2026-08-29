from __future__ import annotations

import torch


def synthetic_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    history_frames, atoms, residues = 3, 5, 7
    history = torch.randn(batch_size, history_frames, atoms, 3)
    target = history[:, -1] + 0.05 * torch.randn(batch_size, atoms, 3)
    pocket_ca = torch.randn(batch_size, residues, 3) + 3.0
    pocket_n = pocket_ca + torch.tensor([-1.2, 0.2, 0.0])
    pocket_c = pocket_ca + torch.tensor([1.3, 0.1, 0.0])
    ligand_mask = torch.ones(batch_size, atoms, dtype=torch.bool)
    pocket_mask = torch.ones(batch_size, residues, dtype=torch.bool)
    if batch_size > 1:
        ligand_mask[1, -1] = False
        pocket_mask[1, -2:] = False
        history[1, :, -1] = 0
        target[1, -1] = 0
    return {
        "history": history,
        "target": target,
        "ligand_z": torch.tensor([[6, 6, 7, 8, 16]]).expand(batch_size, -1).clone(),
        "ligand_mass": torch.tensor(
            [[12.011, 12.011, 14.007, 15.999, 32.06]]
        ).expand(batch_size, -1).clone(),
        "ligand_mask": ligand_mask,
        "pocket_n": pocket_n,
        "pocket_ca": pocket_ca,
        "pocket_c": pocket_c,
        "pocket_residue": torch.arange(residues).expand(batch_size, -1).clone(),
        "pocket_mask": pocket_mask,
    }

