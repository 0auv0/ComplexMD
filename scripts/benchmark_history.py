#!/usr/bin/env python
"""Benchmark ComplexMD training cost for different causal history lengths."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from bindmd.data import MISATOAlignedDataset, MISATOFrameDataset, collate_bindmd
from bindmd.models import build_model


def move(batch: dict[str, torch.Tensor], device: torch.device):
    return {name: value.to(device) for name, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--histories", type=int, nargs="+", default=[8, 20])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    payload = torch.load(args.checkpoint, map_location="cpu")
    model_config = payload.get("config", config)["model"]
    state = payload["model"] if "model" in payload else payload
    data_config = config["data"]
    base = MISATOAlignedDataset(
        data_config["aligned_cache_dir"],
        "train",
        data_config.get("topology_cache_dir"),
    )
    device = torch.device(args.device)
    results = []

    for history_frames in args.histories:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        dataset = MISATOFrameDataset(
            base,
            history_frames=history_frames,
            pocket_cutoff=data_config["pocket_cutoff"],
            max_pocket_residues=data_config["max_pocket_residues"],
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_bindmd,
        )
        batches = list(itertools.islice(loader, args.warmup + args.steps))
        model = build_model(dict(model_config)).to(device)
        model.load_state_dict(state)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        def one_step(cpu_batch):
            batch = move(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                output = model.training_loss(
                    batch, history_noise_max=0.10, pair_loss_weight=0.05
                )
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            return float(output["loss"])

        for batch in batches[: args.warmup]:
            one_step(batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        losses = [one_step(batch) for batch in batches[args.warmup :]]
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        atom_counts = [int(batch["ligand_mask"].sum(1).float().mean()) for batch in batches]
        results.append(
            {
                "history_frames": history_frames,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "seconds": elapsed,
                "seconds_per_step": elapsed / args.steps,
                "samples_per_second": args.batch_size * args.steps / elapsed,
                "peak_allocated_gib": peak,
                "mean_loss": sum(losses) / len(losses),
                "mean_ligand_atoms_in_batches": sum(atom_counts) / len(atom_counts),
            }
        )
        del model, optimizer, scaler, batches
        torch.cuda.empty_cache()

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
