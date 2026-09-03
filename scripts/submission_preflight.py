#!/usr/bin/env python
"""Fail early when the GOAI reproducibility package or input is inconsistent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from bindmd.models import build_model


EXPECTED_SHA256 = "9493faa931d305ec3a78b4c14a1e6a3257d400fc9114a935bdab9606c81901ee"
EXPECTED_COUNTS = {"T1": (10, 10, 30), "T2": (80, 20, 30), "T3": (20, 80, 30)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(args.input_root)
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config)
    if sha256(checkpoint) != EXPECTED_SHA256:
        raise RuntimeError("checkpoint SHA256 mismatch")

    config = yaml.safe_load(config_path.read_text())
    model_config = config["model"]
    assert config["data"]["history_frames"] == 12
    assert model_config["historical_window_frames"] == 6
    assert model_config["current_window_frames"] == 6
    assert model_config["torsion_confidence_threshold"] == 0.75
    assert model_config["torsion_step_limit_deg"] == 5.0

    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(model_config)
    model.load_state_dict(payload["model"] if "model" in payload else payload, strict=True)

    for tier, (n_obs, n_pred, n_systems) in EXPECTED_COUNTS.items():
        tier_dir = root / tier
        ids_path = tier_dir / "ids.txt"
        if not ids_path.is_file():
            raise FileNotFoundError(ids_path)
        identifiers = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
        if len(identifiers) != n_systems:
            raise RuntimeError(f"{tier}: expected {n_systems} systems, found {len(identifiers)}")
        for identifier in identifiers:
            sample = tier_dir / identifier
            meta = json.loads((sample / "meta.json").read_text())
            if (int(meta["n_obs"]), int(meta["n_pred"])) != (n_obs, n_pred):
                raise RuntimeError(f"{identifier}: unexpected observation/prediction counts")
            for suffix in (".pdb", "_obs.xtc"):
                path = sample / f"{identifier}{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(path)

    print("preflight: checkpoint, 6+6 configuration and T1-T3 input contract are valid")
    print(f"preflight: torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")


if __name__ == "__main__":
    main()

