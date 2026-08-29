"""Ligand bond graphs and torsion trees derived from MISATO Amber topologies."""

from __future__ import annotations

from collections import deque

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
    bonds: list[tuple[int, int, float]],
    *,
    minimum_rotatable_length: float = 1.38,
) -> dict[str, torch.Tensor]:
    """Build a rooted rotatable-bond tree for a heavy-atom ligand.

    ``bonds`` contains compact heavy-atom indices and the Amber equilibrium
    bond length. Ring bonds are rejected by an explicit bridge test; terminal
    bonds and short double/amide-like bonds are also excluded.
    """
    atom_count = len(atomic_numbers)
    adjacency = [set() for _ in range(atom_count)]
    unique: dict[tuple[int, int], float] = {}
    for left, right, length in bonds:
        if left == right:
            continue
        edge = (min(left, right), max(left, right))
        unique[edge] = float(length)
        adjacency[left].add(right)
        adjacency[right].add(left)

    if atom_count and any(not neighbors for neighbors in adjacency) and atom_count > 1:
        # Keep the topology usable for disconnected counterions, but never make
        # a torsion that crosses components.
        pass
    root = _graph_center(adjacency) if atom_count else 0
    torsion_bonds: list[list[int]] = []
    torsion_quads: list[list[int]] = []
    rotate_masks: list[torch.Tensor] = []

    for (left, right), length in sorted(unique.items()):
        if length < minimum_rotatable_length:
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
        torsion_bonds.append([parent, child])
        torsion_quads.append(
            [parent_neighbors[0], parent, child, child_neighbors[0]]
        )
        rotate_masks.append(mask)

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
    }
