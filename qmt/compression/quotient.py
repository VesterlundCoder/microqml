"""Quotient graph utilities."""
from __future__ import annotations
import numpy as np
import networkx as nx
from .base import CompressedGraph


def quotient_to_networkx(cg: CompressedGraph) -> nx.Graph:
    """Convert a CompressedGraph to a networkx weighted graph."""
    k = cg.k
    G = nx.Graph()
    G.add_nodes_from(range(k))
    for a in range(k):
        for b in range(a + 1, k):
            w = cg.A_c[a, b]
            if w > 0:
                G.add_edge(a, b, weight=float(w))
    G.graph["k"] = k
    G.graph["n_original"] = cg.n_original
    G.graph["method"] = cg.method
    return G


def get_edge_list(cg: CompressedGraph) -> list[tuple[int, int, float]]:
    """Return list of (a, b, J_ab) for all supernode pairs with non-zero interaction."""
    edges = []
    k = cg.k
    for a in range(k):
        for b in range(a + 1, k):
            j = cg.J[a, b]
            if abs(j) > 1e-10:
                edges.append((a, b, float(j)))
    return edges


def get_node_weights(cg: CompressedGraph, w: np.ndarray | None = None) -> np.ndarray:
    """Compute h_a = w^T z_a for each supernode.

    Parameters
    ----------
    cg : CompressedGraph
    w : np.ndarray, shape (4,)
        Projection weights. If None, uses [0.25, 0.25, 0.25, 0.25].

    Returns
    -------
    np.ndarray, shape (k,)
    """
    if w is None:
        w = np.array([0.25, 0.25, 0.25, 0.25])
    return cg.node_descriptors @ w
