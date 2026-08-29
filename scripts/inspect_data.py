#!/usr/bin/env python
"""Print the effective BindMD tensors for one processed MISATO complex."""

from __future__ import annotations

import argparse

from bindmd.data import MISATOProcessedDataset, prepare_complex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    dataset = MISATOProcessedDataset(args.root, args.split)
    sample = prepare_complex(dataset[args.index])
    for key, value in sample.items():
        print(f"{key:18s} shape={tuple(value.shape)} dtype={value.dtype}")


if __name__ == "__main__":
    main()
