"""Base compressor interface and quotient graph construction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
from scipy import sparse


@dataclass
class CompressedGraph:
    """Result of compressing a graph to k supernodes."""
    S: np.ndarray               # (N, k) partition matrix, S[i,a]=1 if node i in supernode a
    A_c: np.ndarray             # (k, k) quotient adjacency matrix
    J: np.ndarray               # (k, k) normalized interaction strengths
    node_descriptors: np.ndarray  # (k, 4) per-supernode descriptors [mass/N, internal, degree, boundary]
    k: int
    n_original: int
    method: str = ""
    extra: dict = field(default_factory=dict)


class Compressor(ABC):
    """Abstract base class for graph compressors."""

    @abstractmethod
    def compress(self, G: nx.Graph, k: int) -> CompressedGraph:
        """Compress graph G to exactly k supernodes."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


def build_quotient(
    G: nx.Graph, S: np.ndarray, k: int, method: str = "",
) -> CompressedGraph:
    """Build quotient graph from partition matrix S.

    Parameters
    ----------
    G : nx.Graph
        Original graph with N nodes.
    S : np.ndarray, shape (N, k)
        Partition matrix. S[i, a] = 1 if node i belongs to supernode a.
    k : int
        Number of supernodes.

    Returns
    -------
    CompressedGraph
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    # Build adjacency matrix
    A = nx.adjacency_matrix(G, nodelist=nodes).astype(float)
    A = np.asarray(A.todense())
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Quotient adjacency: A_c = S^T A S
    A_c = S.T @ A @ S

    # Cluster masses
    masses = S.sum(axis=0).astype(float)  # (k,)
    masses = np.maximum(masses, 1.0)

    # Normalized interaction strengths
    J = np.zeros((k, k), dtype=float)
    for a in range(k):
        for b in range(k):
            if a == b:
                J[a, b] = 0.0
            else:
                J[a, b] = A_c[a, b] / (np.sqrt(masses[a] * masses[b]) + 1e-10)

    # Per-supernode descriptors
    node_desc = np.zeros((k, 4), dtype=float)
    for a in range(k):
        members = np.where(S[:, a] > 0)[0]
        m_a = len(members)
        node_desc[a, 0] = m_a / max(n, 1)  # mass fraction

        # Internal edge mass
        internal = 0.0
        for i in members:
            for j in members:
                if i < j:
                    internal += A[i, j]
        node_desc[a, 1] = internal / max(m_a * (m_a - 1) / 2, 1) if m_a > 1 else 0.0

        # Degree (total edges from this supernode to others)
        deg_a = A_c[a, :].sum() - A_c[a, a]
        node_desc[a, 2] = deg_a / max(m_a, 1)

        # Boundary fraction: fraction of edges that go to other supernodes
        total_a = A_c[a, :].sum()
        if total_a > 0:
            node_desc[a, 3] = deg_a / total_a
        else:
            node_desc[a, 3] = 0.0

    return CompressedGraph(
        S=S,
        A_c=A_c,
        J=J,
        node_descriptors=node_desc,
        k=k,
        n_original=n,
        method=method,
    )
