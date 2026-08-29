#!/usr/bin/env python
"""Create a clean material-A ZIP containing only the required XTC files."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prediction_root = Path(args.prediction_root)
    output = Path(args.output)
    expected_counts = {"T1": 30, "T2": 30, "T3": 30}
    members: list[tuple[Path, str]] = []
    for tier, expected_count in expected_counts.items():
        paths = sorted(
            (prediction_root / tier).glob(f"{tier}-*_pred.xtc"),
            key=lambda path: int(re.search(r"-(\d+)_pred", path.name).group(1)),
        )
        if len(paths) != expected_count:
            raise ValueError(
                f"{tier}: expected {expected_count} XTC files, found {len(paths)}"
            )
        members.extend((path, f"{tier}/{path.name}") for path in paths)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, archive_name in members:
            archive.write(path, archive_name)
    print(f"saved {output} with {len(members)} XTC files")


if __name__ == "__main__":
    main()

