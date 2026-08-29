#!/usr/bin/env python
"""Overfit BindMD on a tiny MISATO-train subset and compare short rollouts."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from bindmd.data import collate_bindmd, prepare_complex
from bindmd.evaluation.metrics import compute_all_metrics, finite_mean
from bindmd.models import BindMD, build_model


class FixedWindowDataset(Dataset):
    def __init__(
        self,
        cases: dict,
        identifiers: list[str],
        target_frames: list[int],
        history_frames: int,
        pocket_cutoff: float,
        max_pocket_residues: int,
    ):
        self.cases = cases
        self.examples = [
            (identifier, target)
            for identifier in identifiers
            for target in target_frames
        ]
        self.history_frames = history_frames
        self.pocket_cutoff = pocket_cutoff
        self.max_pocket_residues = max_pocket_residues

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        identifier, target_index = self.examples[index]
        raw = self.cases[identifier]
        raw_trajectory = raw.ligand_trajectory_pos.transpose(0, 1)
        item = prepare_complex(
            raw,
            reference_ligand=raw_trajectory[target_index - 1],
            pocket_cutoff=self.pocket_cutoff,
            max_pocket_residues=self.max_pocket_residues,
        )
        trajectory = item.pop("trajectory")
        history = trajectory[
            max(0, target_index - self.history_frames) : target_index
        ]
        if history.shape[0] < self.history_frames:
            history = torch.cat(
                [
                    history[:1].expand(
                        self.history_frames - history.shape[0], -1, -1
                    ),
                    history,
                ],
                dim=0,
            )
        item.update(
            {
                "history": history,
                "target": trajectory[target_index],
                "target_index": torch.tensor(target_index),
            }
        )
        return item


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_bindmd,
        num_workers=0,
        generator=generator,
    )


@torch.no_grad()
def evaluate_teacher_forced(
    model: BindMD,
    dataset: Dataset,
    batch_size: int,
    ddim_steps: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    loader = make_loader(dataset, batch_size, False, seed)
    sums = {"model_squared": 0.0, "persistence_squared": 0.0, "count": 0.0}
    for batch_index, batch in enumerate(loader):
        batch = move(batch, device)
        generator = torch.Generator(device=device).manual_seed(seed + batch_index)
        prediction = model.sample_next(
            batch, ddim_steps=ddim_steps, generator=generator
        )
        mask = batch["ligand_mask"].unsqueeze(-1)
        sums["model_squared"] += float(
            ((prediction - batch["target"]) ** 2 * mask).sum()
        )
        sums["persistence_squared"] += float(
            ((batch["history"][:, -1] - batch["target"]) ** 2 * mask).sum()
        )
        sums["count"] += float(mask.sum() * 3)
    return {
        "coordinate_rmse": math.sqrt(sums["model_squared"] / sums["count"]),
        "persistence_coordinate_rmse": math.sqrt(
            sums["persistence_squared"] / sums["count"]
        ),
    }


@torch.no_grad()
def evaluate_diffusion_objective(
    model: BindMD,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    values = []
    for batch in make_loader(dataset, batch_size, False, seed):
        output = model.training_loss(
            move(batch, device), history_noise_max=0.0, pair_loss_weight=0.05
        )
        values.append({key: float(value) for key, value in output.items()})
    return {
        key: sum(row[key] for row in values) / len(values)
        for key in values[0]
    }


def rollout_batch(
    raw,
    config: dict,
    observed_frames: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    data_config = config["data"]
    raw_trajectory = raw.ligand_trajectory_pos.transpose(0, 1)
    item = prepare_complex(
        raw,
        reference_ligand=raw_trajectory[observed_frames - 1],
        pocket_cutoff=data_config["pocket_cutoff"],
        max_pocket_residues=data_config["max_pocket_residues"],
    )
    trajectory = item.pop("trajectory")
    history_frames = data_config["history_frames"]
    history = trajectory[observed_frames - history_frames : observed_frames]
    target = trajectory[
        observed_frames : observed_frames + config["evaluation"]["rollout_frames"]
    ]
    item.update(
        {
            "history": history,
            "target": target[0],
            "target_index": torch.tensor(observed_frames),
        }
    )
    canonical = bool(getattr(raw, "bindmd_canonical", torch.tensor(False)).item())
    if canonical:
        protein_pos = raw.protein_pos
    else:
        center = raw.protein_pos[raw.mask_ca].mean(dim=0)
        protein_pos = raw.protein_pos - center
    protein_z = torch.full((protein_pos.shape[0],), 6, dtype=torch.long)
    protein_z[raw.mask_n] = 7
    return collate_bindmd([item]), target, protein_pos, protein_z


@torch.no_grad()
def evaluate_rollout(
    model: BindMD,
    cases: dict,
    identifiers: list[str],
    config: dict,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, float]]:
    model.eval()
    observed = config["evaluation"]["rollout_observed_frames"]
    frames = config["evaluation"]["rollout_frames"]
    ddim_steps = config["evaluation"]["ddim_steps"]
    model_rows, persistence_rows = [], []
    for index, identifier in enumerate(identifiers):
        raw = cases[identifier]
        batch, target, protein_pos, protein_z = rollout_batch(raw, config, observed)
        batch = move(batch, device)
        generator = torch.Generator(device=device).manual_seed(seed + index)
        prediction = model.rollout(
            batch, frames, ddim_steps=ddim_steps, generator=generator
        )[0]
        persistence = batch["history"][0, -1].unsqueeze(0).expand_as(prediction)
        common = {
            "target": target.to(device),
            "ligand_z": raw.ligand_x.to(device) + 1,
            "ligand_mass": raw.ligand_mass.to(device),
            "protein_pos": protein_pos.to(device),
            "protein_z": protein_z.to(device),
        }
        model_rows.append(compute_all_metrics(pred=prediction, **common))
        persistence_rows.append(compute_all_metrics(pred=persistence, **common))

    keys = (
        "neuralmd_rmse",
        "neuralmd_matching",
        "neuralmd_stability",
        "geo_ligand_rmsd",
        "geo_ligand_rmsd_last",
        "dyn_rg_mae",
        "stab_error_growth_slope",
    )
    return {
        "model": {
            key: finite_mean([row[key] for row in model_rows]) for key in keys
        },
        "persistence": {
            key: finite_mean([row[key] for row in persistence_rows]) for key in keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small_sample.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(config["data"]["cache"])
    if cache["source_split"] != "train":
        raise ValueError("small-sample cache must come from MISATO train split")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = FixedWindowDataset(
        cache["cases"],
        cache["train_ids"],
        config["data"]["target_frames"],
        config["data"]["history_frames"],
        config["data"]["pocket_cutoff"],
        config["data"]["max_pocket_residues"],
    )
    holdout_dataset = FixedWindowDataset(
        cache["cases"],
        cache["holdout_ids"],
        config["data"]["target_frames"],
        config["data"]["history_frames"],
        config["data"]["pocket_cutoff"],
        config["data"]["max_pocket_residues"],
    )
    train_config = config["training"]
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    result = {
        "config": config,
        "source_split": cache["source_split"],
        "train_ids": cache["train_ids"],
        "holdout_ids": cache["holdout_ids"],
        "num_train_windows": len(train_dataset),
        "num_holdout_windows": len(holdout_dataset),
        "initial": {},
        "trained": {},
        "curve": [],
    }
    result["initial"]["train_objective"] = evaluate_diffusion_objective(
        model, train_dataset, train_config["batch_size"], device, seed + 1000
    )
    result["initial"]["holdout_objective"] = evaluate_diffusion_objective(
        model, holdout_dataset, train_config["batch_size"], device, seed + 2000
    )
    result["initial"]["train_one_step"] = evaluate_teacher_forced(
        model,
        train_dataset,
        train_config["batch_size"],
        config["evaluation"]["ddim_steps"],
        device,
        seed + 3000,
    )
    result["initial"]["holdout_one_step"] = evaluate_teacher_forced(
        model,
        holdout_dataset,
        train_config["batch_size"],
        config["evaluation"]["ddim_steps"],
        device,
        seed + 4000,
    )
    result["initial"]["train_rollout"] = evaluate_rollout(
        model, cache["cases"], cache["train_ids"], config, device, seed + 5000
    )
    result["initial"]["holdout_rollout"] = evaluate_rollout(
        model, cache["cases"], cache["holdout_ids"], config, device, seed + 6000
    )
    print("initial evaluation complete", flush=True)

    loader = make_loader(
        train_dataset, train_config["batch_size"], True, seed
    )
    iterator = cycle(loader)
    model.train()
    start = time.time()
    for step in range(1, train_config["steps"] + 1):
        batch = move(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        output = model.training_loss(
            batch,
            history_noise_max=train_config["history_noise_max"],
            pair_loss_weight=train_config["pair_loss_weight"],
        )
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
        optimizer.step()
        objective_name = "flow_loss" if "flow_loss" in output else "diffusion_loss"
        row = {
            "step": step,
            "loss": float(output["loss"]),
            objective_name: float(output[objective_name]),
            "pair_loss": float(output["pair_loss"]),
        }
        if step == 1 or step % train_config["log_every"] == 0:
            result["curve"].append(row)
            print(json.dumps(row), flush=True)

    result["train_seconds"] = time.time() - start
    result["peak_gpu_memory_mb"] = (
        torch.cuda.max_memory_allocated() / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    result["trained"]["train_objective"] = evaluate_diffusion_objective(
        model, train_dataset, train_config["batch_size"], device, seed + 1000
    )
    result["trained"]["holdout_objective"] = evaluate_diffusion_objective(
        model, holdout_dataset, train_config["batch_size"], device, seed + 2000
    )
    result["trained"]["train_one_step"] = evaluate_teacher_forced(
        model,
        train_dataset,
        train_config["batch_size"],
        config["evaluation"]["ddim_steps"],
        device,
        seed + 3000,
    )
    result["trained"]["holdout_one_step"] = evaluate_teacher_forced(
        model,
        holdout_dataset,
        train_config["batch_size"],
        config["evaluation"]["ddim_steps"],
        device,
        seed + 4000,
    )
    result["trained"]["train_rollout"] = evaluate_rollout(
        model, cache["cases"], cache["train_ids"], config, device, seed + 5000
    )
    result["trained"]["holdout_rollout"] = evaluate_rollout(
        model, cache["cases"], cache["holdout_ids"], config, device, seed + 6000
    )
    checkpoint = output_dir / "small_sample_last.pt"
    torch.save(
        {"model": model.state_dict(), "config": config, "result": result},
        checkpoint,
    )
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(f"saved checkpoint={checkpoint}", flush=True)
    print(f"saved result={result_path}", flush=True)


if __name__ == "__main__":
    main()
