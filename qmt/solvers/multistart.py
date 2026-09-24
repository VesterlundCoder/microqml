"""Multi-start local search for MaxCut."""
from __future__ import annotations
import numpy as np
import networkx as nx
from .local_search import _compute_cut_value, _flip_gain


def solve_maxcut_multistart(
    G: nx.Graph, budget: int = 1000, n_starts: int = 10, seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Multi-start random assignment + 1-flip local search.

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

    best_cut = -1.0
    best_assignment = None
    total_budget = budget
    budget_per_start = max(total_budget // max(n_starts, 1), 1)

    for _ in range(n_starts):
        if total_budget <= 0:
            break
        assignment = rng.choice([-1, 1], size=n).astype(np.int8)
        cut = _compute_cut_value(G, assignment, nodes)

        flips = 0
        improved = True
        while improved and flips < budget_per_start:
            improved = False
            order = rng.permutation(n)
            for i in order:
                v = nodes[i]
                gain = _flip_gain(G, assignment, i, v, node_to_idx)
                if gain > 1e-10:
                    assignment[i] *= -1
                    cut += gain
                    flips += 1
                    improved = True
                    total_budget -= 1
                    if total_budget <= 0 or flips >= budget_per_start:
                        break

        if cut > best_cut:
            best_cut = cut
            best_assignment = assignment.copy()

    if best_assignment is None:
        best_assignment = np.ones(n, dtype=np.int8)
        best_cut = _compute_cut_value(G, best_assignment, nodes)

    return float(best_cut), best_assignment
