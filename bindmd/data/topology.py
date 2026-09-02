"""Ligand rigid fragments and torsion trees from chemistry or connectivity."""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import torch


def _graph_center(adjacency: list[set[int]]) -> int:
    """Choose a stable root in the most central heavy-atom region."""
    best_score = (10**9, 10**9, 10**9)
    best_root = 0
    for root in range(len(adjacency)):
        distance = [-1] * len(adjacency)
        distance[root] = 0
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        disconnected = sum(value < 0 for value in distance)
        score = (disconnected, sum(value for value in distance if value >= 0), -len(adjacency[root]))
        if score < best_score:
            best_score = score
            best_root = root
    return best_root


def _component_without_edge(
    adjacency: list[set[int]], start: int, edge: tuple[int, int]
) -> set[int]:
    blocked = {edge, (edge[1], edge[0])}
    component = {start}
    queue = [start]
    while queue:
        node = queue.pop()
        for neighbor in adjacency[node]:
            if (node, neighbor) in blocked or neighbor in component:
                continue
            component.add(neighbor)
            queue.append(neighbor)
    return component


def build_torsion_topology(
    atomic_numbers: list[int],
    bonds: Sequence[tuple[int, int, float] | tuple[int, int, float, bool]],
    *,
    minimum_rotatable_length: float = 1.38,
) -> dict[str, torch.Tensor]:
    """Build a rooted rotatable-bond tree for a heavy-atom ligand.

    ``bonds`` contains compact heavy-atom indices, an equilibrium/observed
    length and, optionally, a chemistry-derived rotatable flag. Ring bonds are
    rejected by an explicit bridge test. Without a chemistry flag, terminal
    and short double/amide-like bonds are excluded using the conservative
    connectivity fallback used for Amber and PDB CONECT inputs.
    """
    atom_count = len(atomic_numbers)
    adjacency = [set() for _ in range(atom_count)]
    unique: dict[tuple[int, int], tuple[float, bool | None]] = {}
    for bond in bonds:
        if len(bond) == 3:
            left, right, length = bond
            rotatable = None
        elif len(bond) == 4:
            left, right, length, rotatable = bond
            rotatable = bool(rotatable)
        else:
            raise ValueError("bonds must contain (i, j, length[, rotatable])")
        if left == right:
            continue
        edge = (min(left, right), max(left, right))
        unique[edge] = (float(length), rotatable)
        adjacency[left].add(right)
        adjacency[right].add(left)

    if atom_count and any(not neighbors for neighbors in adjacency) and atom_count > 1:
        # Keep the topology usable for disconnected counterions, but never make
        # a torsion that crosses components.
        pass
    root = _graph_center(adjacency) if atom_count else 0
    distance = [-1] * atom_count
    if atom_count:
        distance[root] = 0
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
    torsions: list[tuple[int, list[int], list[int], torch.Tensor]] = []

    for (left, right), (length, chemistry_rotatable) in sorted(unique.items()):
        if chemistry_rotatable is False:
            continue
        if chemistry_rotatable is None and length < minimum_rotatable_length:
            continue
        if len(adjacency[left]) < 2 or len(adjacency[right]) < 2:
            continue
        component = _component_without_edge(adjacency, right, (left, right))
        if left in component:  # The edge lies in a ring.
            continue
        if root in component:
            parent, child = right, left
            rotating = set(range(atom_count)) - component
        else:
            parent, child = left, right
            rotating = component
        if len(rotating) < 2 or atom_count - len(rotating) < 2:
            continue
        parent_neighbors = sorted(adjacency[parent] - {child} - rotating)
        child_neighbors = sorted((adjacency[child] - {parent}) & rotating)
        if not parent_neighbors or not child_neighbors:
            continue
        mask = torch.zeros(atom_count, dtype=torch.bool)
        mask[list(sorted(rotating))] = True
        torsions.append(
            (
                distance[child] if distance[child] >= 0 else atom_count,
                [parent, child],
                [parent_neighbors[0], parent, child, child_neighbors[0]],
                mask,
            )
        )

    # Nested rotations are non-commutative.  Applying them from the central
    # rigid fragment outwards preserves each downstream target dihedral.
    torsions.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    torsion_bonds = [item[1] for item in torsions]
    torsion_quads = [item[2] for item in torsions]
    rotate_masks = [item[3] for item in torsions]

    rotatable_edges = {tuple(sorted(bond)) for bond in torsion_bonds}
    rigid_fragment = torch.full((atom_count,), -1, dtype=torch.long)
    fragment_count = 0
    for start in range(atom_count):
        if int(rigid_fragment[start]) >= 0:
            continue
        stack = [start]
        rigid_fragment[start] = fragment_count
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if tuple(sorted((node, neighbor))) in rotatable_edges:
                    continue
                if int(rigid_fragment[neighbor]) < 0:
                    rigid_fragment[neighbor] = fragment_count
                    stack.append(neighbor)
        fragment_count += 1

    bond_index = torch.tensor(list(unique), dtype=torch.long)
    if bond_index.numel() == 0:
        bond_index = torch.empty(0, 2, dtype=torch.long)
    torsion_bond = torch.tensor(torsion_bonds, dtype=torch.long)
    torsion_quad = torch.tensor(torsion_quads, dtype=torch.long)
    if torsion_bond.numel() == 0:
        torsion_bond = torch.empty(0, 2, dtype=torch.long)
        torsion_quad = torch.empty(0, 4, dtype=torch.long)
        torsion_rotate_mask = torch.empty(0, atom_count, dtype=torch.bool)
    else:
        torsion_rotate_mask = torch.stack(rotate_masks)
    return {
        "bond_index": bond_index,
        "torsion_bond": torsion_bond,
        "torsion_quad": torsion_quad,
        "torsion_rotate_mask": torsion_rotate_mask,
        "torsion_root": torch.tensor(root, dtype=torch.long),
        "rigid_fragment": rigid_fragment,
        "rigid_fragment_count": torch.tensor(fragment_count, dtype=torch.long),
    }


def build_torsion_topology_from_connectivity(
    atomic_numbers: Sequence[int],
    coordinates: torch.Tensor,
    bonds: Sequence[tuple[int, int]],
    *,
    minimum_rotatable_length: float = 1.38,
) -> dict[str, torch.Tensor]:
    """Build rigid fragments from PDB CONECT or another untyped bond graph."""

    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    measured = [
        (left, right, float((coordinates[left] - coordinates[right]).norm()))
        for left, right in bonds
    ]
    return build_torsion_topology(
        list(atomic_numbers),
        measured,
        minimum_rotatable_length=minimum_rotatable_length,
    )


def build_torsion_topology_from_qm_group(
    group: Any,
    *,
    heavy_only: bool = True,
    model_atomic_numbers: Sequence[int] | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Infer rigid fragments directly from one ``QM.hdf5`` molecule group.

    MISATO stores atom numbers, hybridisation labels and a directed bond table
    ``(left, right, order)``.  The trajectory model uses heavy atoms, so this
    routine removes hydrogens while preserving the remaining QM/MD atom order.
    A rotatable edge must be a non-terminal, non-ring single bond.  In
    addition, carbonyl-/thiocarbonyl-/sulfonyl-to-hetero bonds are kept inside
    one rigid fragment because resonance gives them substantial double-bond
    character.  Ring and terminal checks are performed by
    :func:`build_torsion_topology` after the chemical flags are assembled.
    """

    atom_group = group["atom_properties"]
    atomic_numbers_all = [int(value) for value in atom_group["atom_names"].asstr()[:]]
    property_names = atom_group["atom_properties_names"].asstr()[:].tolist()
    property_values = atom_group["atom_properties_values"][:]
    if "hybridisation" not in property_names:
        raise ValueError("QM molecule does not contain a hybridisation property")
    hybridisation_all = property_values[
        :, property_names.index("hybridisation")
    ].astype(int)

    selected = [
        index
        for index, atomic_number in enumerate(atomic_numbers_all)
        if not heavy_only or atomic_number != 1
    ]
    if model_atomic_numbers is not None:
        model_atomic_numbers = [int(value) for value in model_atomic_numbers]
        selected_atomic_numbers = [atomic_numbers_all[index] for index in selected]
        if selected_atomic_numbers[: len(model_atomic_numbers)] != model_atomic_numbers:
            raise ValueError("QM and model ligand atom order disagree")
        # A small number of MISATO QM entries include a trailing solvent/ion
        # atom that NeuralMD intentionally omitted.  Restrict the bond graph to
        # exactly the atoms present in the trajectory model.
        selected = selected[: len(model_atomic_numbers)]
    compact = {original: index for index, original in enumerate(selected)}
    atomic_numbers = [atomic_numbers_all[index] for index in selected]
    hybridisation = [int(hybridisation_all[index]) for index in selected]

    # The QM bond table stores both directions.  Keep one edge and the largest
    # reported order to make the result independent of HDF5 row ordering.
    bond_orders: dict[tuple[int, int], float] = {}
    for left_raw, right_raw, order_raw in atom_group["bonds"][:]:
        left_original, right_original = int(left_raw), int(right_raw)
        if left_original not in compact or right_original not in compact:
            continue
        left, right = compact[left_original], compact[right_original]
        if left == right:
            continue
        edge = (min(left, right), max(left, right))
        bond_orders[edge] = max(float(order_raw), bond_orders.get(edge, 0.0))

    adjacency: list[set[int]] = [set() for _ in atomic_numbers]
    for left, right in bond_orders:
        adjacency[left].add(right)
        adjacency[right].add(left)

    def resonance_locked(left: int, right: int) -> bool:
        # C/S/P(=N/O/S)-N/O/S links cover amides, esters, thioamides,
        # sulfonamides and closely related conjugated bonds.  Hybridisation is
        # deliberately retained as a feature rather than used to reject every
        # sp2-sp2 single bond: biaryl single bonds remain rotatable.
        for center, hetero in ((left, right), (right, left)):
            if atomic_numbers[center] not in {6, 15, 16}:
                continue
            if atomic_numbers[hetero] not in {7, 8, 16}:
                continue
            for neighbor in adjacency[center] - {hetero}:
                edge = (min(center, neighbor), max(center, neighbor))
                if (
                    bond_orders[edge] >= 1.75
                    and atomic_numbers[neighbor] in {7, 8, 16}
                ):
                    return True
        return False

    bonds = []
    for (left, right), order in sorted(bond_orders.items()):
        chemistry_rotatable = (
            abs(order - 1.0) < 0.2
            and hybridisation[left] != 1
            and hybridisation[right] != 1
            and not resonance_locked(left, right)
        )
        bonds.append((left, right, 1.5, chemistry_rotatable))

    topology = build_torsion_topology(atomic_numbers, bonds)
    topology.update(
        {
            "ligand_hybridisation": torch.tensor(hybridisation, dtype=torch.long),
            "topology_atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long),
            "topology_ligand_atoms": torch.tensor(len(atomic_numbers), dtype=torch.long),
        }
    )
    return topology


def build_torsion_topology_from_smiles(
    smiles: str,
    *,
    model_atom_order: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Build chemistry-aware rigid fragments from an ordered heavy-atom SMILES.

    ``model_atom_order[smiles_atom]`` maps each RDKit heavy atom to the model's
    compact atom index. If omitted, SMILES and model atom order are assumed to
    match. Mapped SMILES can therefore be used without relying on atom names.
    """

    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except ImportError as exc:
        raise ImportError("SMILES topology requires RDKit") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse ligand SMILES")
    molecule = Chem.RemoveHs(molecule)
    atom_count = molecule.GetNumAtoms()
    if model_atom_order is None:
        mapped = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
        if mapped and all(value > 0 for value in mapped) and len(set(mapped)) == atom_count:
            order = [value - 1 for value in mapped]
        else:
            order = list(range(atom_count))
    else:
        order = [int(value) for value in model_atom_order]
    if len(order) != atom_count or sorted(order) != list(range(atom_count)):
        raise ValueError("model_atom_order must be a permutation of heavy atoms")

    rotatable_matches = {
        tuple(sorted(match))
        for match in molecule.GetSubstructMatches(Lipinski.RotatableBondSmarts)
    }
    # RDKit's public RotatableBondSmarts is the non-strict definition. Remove
    # resonance-locked C(=O/S/N)-N/O/S links to match the strict descriptor and
    # keep amides, esters, thioamides, and related groups inside rigid pieces.
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        edge = tuple(sorted((left.GetIdx(), right.GetIdx())))
        for center, hetero in ((left, right), (right, left)):
            if hetero.GetAtomicNum() not in {7, 8, 16}:
                continue
            conjugated = any(
                neighbor.GetIdx() != hetero.GetIdx()
                and neighbor_bond.GetBondType() == Chem.BondType.DOUBLE
                and neighbor.GetAtomicNum() in {7, 8, 16}
                for neighbor in center.GetNeighbors()
                for neighbor_bond in [molecule.GetBondBetweenAtoms(
                    center.GetIdx(), neighbor.GetIdx()
                )]
            )
            if conjugated:
                rotatable_matches.discard(edge)
    bonds = []
    for bond in molecule.GetBonds():
        left_smiles, right_smiles = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append(
            (
                order[left_smiles],
                order[right_smiles],
                1.5,
                tuple(sorted((left_smiles, right_smiles))) in rotatable_matches,
            )
        )
    atomic_numbers = [0] * atom_count
    for atom in molecule.GetAtoms():
        atomic_numbers[order[atom.GetIdx()]] = atom.GetAtomicNum()
    return build_torsion_topology(atomic_numbers, bonds)
