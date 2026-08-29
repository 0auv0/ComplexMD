#!/usr/bin/env python
"""Merge non-overlapping ComplexMD evaluation shards and recompute aggregates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bindmd.evaluation.metrics import finite_mean


METADATA = {
    "complex_index",
    "identifier",
    "scenario",
    "observed_frames",
    "predicted_frames",
}


def aggregate(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["scenario"]].append(record)
    return {
        scenario: {
            "num_complexes": len(rows),
            "mean": {
                name: finite_mean([float(row[name]) for row in rows])
                for name in rows[0]
                if name not in METADATA
            },
        }
        for scenario, rows in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-complexes", type=int, required=True)
    args = parser.parse_args()
    payloads = [json.loads(Path(path).read_text()) for path in args.inputs]
    records = [row for payload in payloads for row in payload["records"]]
    persistence = [
        row
        for payload in payloads
        for row in payload["baselines"]["persistence"]["records"]
    ]
    indices = [int(row["complex_index"]) for row in records]
    if len(records) != args.expected_complexes:
        raise ValueError(
            f"expected {args.expected_complexes} records, received {len(records)}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("evaluation shards overlap")
    template = dict(payloads[0])
    template["start_index"] = min(indices)
    template["end_index_exclusive"] = max(indices) + 1
    template["aggregate"] = aggregate(records)
    template["records"] = sorted(records, key=lambda row: row["complex_index"])
    template["baselines"] = {
        "persistence": {
            "aggregate": aggregate(persistence),
            "records": sorted(
                persistence, key=lambda row: row["complex_index"]
            ),
        }
    }
    template["merged_shards"] = args.inputs
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template, indent=2, allow_nan=True))
    print(json.dumps(template["aggregate"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
