#!/usr/bin/env python
"""Extract compact ligand torsion trees from the MISATO Amber archive."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

import h5py
import torch

from bindmd.data.topology import build_torsion_topology


INTEGER_SECTIONS = {
    "ATOMIC_NUMBER",
    "BONDS_INC_HYDROGEN",
    "BONDS_WITHOUT_HYDROGEN",
}
FLOAT_SECTIONS = {"BOND_EQUIL_VALUE"}


def parse_sections(raw: bytes) -> dict[str, list[int] | list[float]]:
    text = raw.decode("ascii")
    result = {}
    for chunk in text.split("%FLAG ")[1:]:
        name, rest = chunk.split("\n", 1)
        if name not in INTEGER_SECTIONS | FLOAT_SECTIONS:
            continue
        _, body = rest.split("\n", 1)
        values = body.split()
        cast = int if name in INTEGER_SECTIONS else lambda value: float(value.replace("D", "E"))
        result[name] = [cast(value) for value in values]
    missing = (INTEGER_SECTIONS | FLOAT_SECTIONS) - result.keys()
    if missing:
        raise ValueError(f"Amber topology missing sections: {sorted(missing)}")
    return result


def one_topology(
    sections: dict[str, list[int] | list[float]],
    group: h5py.Group,
    minimum_rotatable_length: float,
) -> dict[str, torch.Tensor]:
    atomic_numbers = sections["ATOMIC_NUMBER"]
    system_atomic_numbers = group["atoms_number"][:].tolist()
    if atomic_numbers[: len(system_atomic_numbers)] != system_atomic_numbers:
        raise ValueError("Amber and HDF5 atom ordering disagree")
    ligand_begin = int(group["molecules_begin_atom_index"][-1])
    ligand_end = len(system_atomic_numbers)
    heavy_full = [
        index
        for index in range(ligand_begin, ligand_end)
        if atomic_numbers[index] != 1
    ]
    compact = {full: index for index, full in enumerate(heavy_full)}
    equilibrium = sections["BOND_EQUIL_VALUE"]
    bonds = []
    for section_name in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = sections[section_name]
        for offset in range(0, len(values), 3):
            left, right, parameter = values[offset : offset + 3]
            left, right = left // 3, right // 3
            if left in compact and right in compact:
                bonds.append(
                    (compact[left], compact[right], equilibrium[parameter - 1])
                )
    topology = build_torsion_topology(
        [atomic_numbers[index] for index in heavy_full],
        bonds,
        minimum_rotatable_length=minimum_rotatable_length,
    )
    topology["topology_ligand_atoms"] = torch.tensor(len(heavy_full))
    return topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--aligned-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-rotatable-length", type=float, default=1.38)
    args = parser.parse_args()

    aligned_dir = Path(args.aligned_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_identifiers = {}
    wanted = set()
    for split in ("train", "val", "test"):
        payload = torch.load(aligned_dir / f"aligned_{split}.pt", map_location="cpu")
        identifiers = [str(value).upper() for value in payload["identifiers"]]
        split_identifiers[split] = identifiers
        wanted.update(identifiers)

    found: dict[str, dict[str, torch.Tensor]] = {}
    with h5py.File(args.hdf5, "r") as hdf5, tarfile.open(args.archive, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("/production.top.gz"):
                continue
            identifier = Path(member.name).parent.name.upper()
            if identifier not in wanted:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            sections = parse_sections(gzip.decompress(handle.read()))
            found[identifier] = one_topology(
                sections,
                hdf5[identifier],
                args.minimum_rotatable_length,
            )
            if len(found) % 250 == 0:
                print(f"parsed {len(found)}/{len(wanted)}", flush=True)

    missing = sorted(wanted - found.keys())
    if missing:
        raise RuntimeError(f"missing {len(missing)} topologies; first entries: {missing[:20]}")

    statistics = {}
    for split, identifiers in split_identifiers.items():
        cases = [found[identifier] for identifier in identifiers]
        output = {
            "identifiers": identifiers,
            "cases": cases,
            "source": str(Path(args.archive).resolve()),
            "minimum_rotatable_length": args.minimum_rotatable_length,
        }
        torch.save(output, output_dir / f"topology_{split}.pt")
        counts = [case["torsion_bond"].shape[0] for case in cases]
        statistics[split] = {
            "complexes": len(cases),
            "mean_torsions": sum(counts) / max(len(counts), 1),
            "max_torsions": max(counts, default=0),
            "zero_torsion_complexes": sum(value == 0 for value in counts),
        }
    (output_dir / "statistics.json").write_text(json.dumps(statistics, indent=2))
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    main()

