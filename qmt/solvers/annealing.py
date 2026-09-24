"""Simulated annealing for MaxCut."""
from __future__ import annotations
import numpy as np
import networkx as nx
from .local_search import _compute_cut_value, _flip_gain


def solve_maxcut_annealing(
    G: nx.Graph, budget: int = 1000, t_start: float = 2.0,
    t_end: float = 0.01, seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Simulated annealing with single-flip moves.

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

    if budget <= 0:
        return float(best_cut), best_assignment

    for step in range(budget):
        frac = step / max(budget, 1)
        temp = t_start * (t_end / t_start) ** frac

        i = int(rng.integers(0, n))
        v = nodes[i]
        gain = _flip_gain(G, assignment, i, v, node_to_idx)

        if gain > 0 or rng.random() < np.exp(gain / max(temp, 1e-10)):
            assignment[i] *= -1
            current_cut += gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_assignment = assignment.copy()

    return float(best_cut), best_assignment
