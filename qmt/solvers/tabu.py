"""Tabu search for MaxCut with Kernighan-Lin-style moves."""
from __future__ import annotations
import numpy as np
import networkx as nx
from .local_search import _compute_cut_value, _flip_gain


def solve_maxcut_tabu(
    G: nx.Graph, budget: int = 1000, tabu_size: int = 20, seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Tabu search with single-flip moves and tabu list.

    Returns
    -------
    cut_value : float
    assignment : np.ndarray, +1/-1
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0, np.ones(n)

    nodes = sorted(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    rng = np.random.default_rng(seed)

    assignment = rng.choice([-1, 1], size=n).astype(np.int8)
    current_cut = _compute_cut_value(G, assignment, nodes)
    best_cut = current_cut
    best_assignment = assignment.copy()

    tabu = {}  # node_idx -> iteration until which it's tabu
    tabu_size = min(tabu_size, n // 2 + 1)

    for step in range(budget):
        best_gain = -np.inf
        best_i = -1

        order = rng.permutation(n)
        for i in order:
            if i in tabu and tabu[i] > step:
                continue
            v = nodes[i]
            gain = _flip_gain(G, assignment, i, v, node_to_idx)
            if gain > best_gain:
                best_gain = gain
                best_i = i

        if best_i < 0:
            # All tabu, clear oldest
            if tabu:
                min_step = min(tabu.values())
                for k in list(tabu.keys()):
                    if tabu[k] == min_step:
                        del tabu[k]
            continue

        assignment[best_i] *= -1
        current_cut += best_gain
        tabu[best_i] = step + tabu_size

        if current_cut > best_cut:
            best_cut = current_cut
            best_assignment = assignment.copy()

    return float(best_cut), best_assignment
