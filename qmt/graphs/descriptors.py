"""Cheap classical graph-level descriptors (x_G in R^12).

These are the ONLY features the deployed micro-student sees at inference.
The quantum teacher receives the much richer k x k quotient graph.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh


def compute_descriptors(G: nx.Graph) -> np.ndarray:
    """Compute 12 cheap graph-level descriptors.

    Returns
    -------
    np.ndarray, shape (12,)
        1. log(N)
        2. normalized edge density
        3. mean degree
        4. degree std
        5. degree skewness
        6. max degree / N
        7. average clustering
        8. triangle density
        9. assortativity
        10. connected component count
        11. approx spectral radius
        12. approx algebraic connectivity
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()], dtype=float)

    # 1. log(N)
    log_n = np.log(max(n, 2))

    # 2. normalized edge density
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    density = m / max_edges if max_edges > 0 else 0.0

    # 3. mean degree
    mean_deg = degrees.mean() if len(degrees) > 0 else 0.0

    # 4. degree std
    std_deg = degrees.std() if len(degrees) > 0 else 0.0

    # 5. degree skewness
    if len(degrees) > 2 and std_deg > 1e-10:
        skew = float(((degrees - degrees.mean()) ** 3).mean() / (std_deg ** 3))
    else:
        skew = 0.0

    # 6. max degree / N
    max_deg_ratio = degrees.max() / max(n, 1) if len(degrees) > 0 else 0.0

    # 7. average clustering
    try:
        avg_clustering = nx.average_clustering(G)
    except Exception:
        avg_clustering = 0.0

    # 8. triangle density
    try:
        triangles = sum(nx.triangles(G).values()) / 3.0
        max_triangles = n * (n - 1) * (n - 2) / 6.0 if n > 2 else 1
        tri_density = triangles / max_triangles if max_triangles > 0 else 0.0
    except Exception:
        tri_density = 0.0

    # 9. assortativity
    try:
        assort = nx.degree_assortativity_coefficient(G)
        if not np.isfinite(assort):
            assort = 0.0
    except Exception:
        assort = 0.0

    # 10. connected component count
    n_components = nx.number_connected_components(G)

    # 11. approx spectral radius (largest eigenvalue of adjacency)
    if n > 1 and m > 0:
        try:
            L = nx.laplacian_matrix(G).astype(float)
            eigs = eigvalsh(L.toarray(), subset_by_index=[0, min(2, n - 1)])
            spectral_radius = float(eigs[-1])  # largest of smallest 3
            # Actually compute largest eigenvalue of adjacency
            A = nx.adjacency_matrix(G).astype(float)
            adj_eigs = eigvalsh(A.toarray(), subset_by_index=[n - 1, n - 1])
            spectral_radius = float(adj_eigs[0])
        except Exception:
            spectral_radius = float(mean_deg)
    else:
        spectral_radius = 0.0

    # 12. approx algebraic connectivity (Fiedler value)
    if n > 2 and m > 0:
        try:
            L = nx.laplacian_matrix(G).astype(float)
            eigs = eigvalsh(L.toarray(), subset_by_index=[0, min(1, n - 1)])
            algebraic_conn = float(eigs[1]) if len(eigs) > 1 else 0.0
        except Exception:
            algebraic_conn = 0.0
    else:
        algebraic_conn = 0.0

    x = np.array([
        log_n, density, mean_deg, std_deg, skew, max_deg_ratio,
        avg_clustering, tri_density, assort, float(n_components),
        spectral_radius, algebraic_conn,
    ], dtype=np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def compute_descriptors_batch(graphs: list[nx.Graph]) -> np.ndarray:
    """Compute descriptors for a list of graphs. Returns (N, 12) array."""
    return np.stack([compute_descriptors(G) for G in graphs])
