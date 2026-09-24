"""Greedy + 1-flip local search for MaxCut."""
from __future__ import annotations
import numpy as np
import networkx as nx


def solve_maxcut_greedy(G: nx.Graph, budget: int = 1000) -> tuple[float, np.ndarray]:
    """Greedy assignment followed by 1-flip local search.

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

    # Greedy: assign each node to the side that maximizes cut
    assignment = np.zeros(n, dtype=np.int8)
    for i, v in enumerate(nodes):
        s_same = 0.0
        s_diff = 0.0
        for u in G.neighbors(v):
            j = node_to_idx[u]
            if j < i:
                if assignment[j] == 1:
                    s_diff += 1.0
                else:
                    s_same += 1.0
        assignment[i] = 1 if s_diff >= s_same else -1

    cut_value = _compute_cut_value(G, assignment, nodes)

    # 1-flip local search
    flips = 0
    improved = True
    while improved and flips < budget:
        improved = False
        for i, v in enumerate(nodes):
            gain = _flip_gain(G, assignment, i, v, node_to_idx)
            if gain > 1e-10:
                assignment[i] *= -1
                cut_value += gain
                flips += 1
                improved = True
                if flips >= budget:
                    break

    return float(cut_value), assignment


def _flip_gain(G, assignment, i, v, node_to_idx):
    """Compute the change in cut value if node v is flipped."""
    gain = 0.0
    for u in G.neighbors(v):
        j = node_to_idx[u]
        if j == i:
            continue
        w = G.edges[v, u].get("weight", 1.0)
        if assignment[i] != assignment[j]:
            gain -= w  # was crossing, now same
        else:
            gain += w  # was same, now crossing
    return gain


def _compute_cut_value(G, assignment, nodes):
    total = 0.0
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if assignment[node_to_idx[u]] != assignment[node_to_idx[v]]:
            total += w
    return total
