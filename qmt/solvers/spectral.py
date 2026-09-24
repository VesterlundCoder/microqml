"""Spectral rounding for MaxCut: use the Fiedler vector sign as assignment."""
from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh, eigh


def solve_maxcut_spectral(G: nx.Graph, budget: int = 1000) -> tuple[float, np.ndarray]:
    """Spectral MaxCut via Fiedler vector sign rounding.

    Returns
    -------
    cut_value : float
        Total weight of edges crossing the cut.
    assignment : np.ndarray
        +1/-1 array of length n.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0, np.ones(n)

    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).astype(float).toarray()
    degrees = A.sum(axis=1)
    degrees_safe = np.where(degrees > 0, degrees, 1.0)

    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    L_norm = np.nan_to_num(L_norm, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        eigvals, eigvecs = eigh(L_norm)
        # Fiedler vector is the second smallest
        fiedler = eigvecs[:, 1]
    except Exception:
        fiedler = np.random.default_rng(0).standard_normal(n)

    assignment = np.where(fiedler >= 0, 1, -1).astype(np.int8)
    cut_value = _compute_cut_value(G, assignment)
    return float(cut_value), assignment


def _compute_cut_value(G: nx.Graph, assignment: np.ndarray) -> float:
    """Compute the cut value for a given assignment."""
    total = 0.0
    nodes = sorted(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if assignment[node_to_idx[u]] != assignment[node_to_idx[v]]:
            total += w
    return total
