"""Classical features extracted from compressed graphs (M4 teacher).

These are non-quantum features derived from the compressed graph structure,
used as a classical compressed teacher to test whether quantum measurement
adds information beyond classical compressed features.
"""
from __future__ import annotations

import numpy as np
from ..compression.base import CompressedGraph


def extract_classical_compressed_features(cg: CompressedGraph) -> np.ndarray:
    """Extract classical features from a compressed graph.

    Features (12 total):
    - A_c mean, std, max (3)
    - J mean, std, max (3)
    - node_descriptors mean per column (4)
    - spectral radius of A_c (1)
    - density of compressed graph (1)

    Parameters
    ----------
    cg : CompressedGraph

    Returns
    -------
    np.ndarray, shape (12,)
    """
    k = cg.k
    A_c = cg.A_c.copy()
    J = cg.J.copy()
    nd = cg.node_descriptors.copy()

    # Clean
    A_c = np.nan_to_num(A_c, nan=0.0, posinf=0.0, neginf=0.0)
    J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    nd = np.nan_to_num(nd, nan=0.0, posinf=0.0, neginf=0.0)

    # Adjacency stats
    ac_upper = A_c[np.triu_indices(k, k=1)]
    ac_mean = float(np.mean(ac_upper)) if len(ac_upper) > 0 else 0.0
    ac_std = float(np.std(ac_upper)) if len(ac_upper) > 0 else 0.0
    ac_max = float(np.max(ac_upper)) if len(ac_upper) > 0 else 0.0

    # Interaction stats
    j_upper = J[np.triu_indices(k, k=1)]
    j_mean = float(np.mean(j_upper)) if len(j_upper) > 0 else 0.0
    j_std = float(np.std(j_upper)) if len(j_upper) > 0 else 0.0
    j_max = float(np.max(j_upper)) if len(j_upper) > 0 else 0.0

    # Node descriptor means
    nd_means = np.mean(nd, axis=0).tolist()  # 4 values

    # Spectral radius of A_c
    try:
        eigvals = np.linalg.eigvals(A_c)
        spectral_radius = float(np.max(np.abs(eigvals)))
    except Exception:
        spectral_radius = 0.0

    # Density of compressed graph
    n_possible = k * (k - 1) / 2
    n_actual = np.count_nonzero(ac_upper)
    density = float(n_actual / max(n_possible, 1))

    features = np.array([
        ac_mean, ac_std, ac_max,
        j_mean, j_std, j_max,
        nd_means[0], nd_means[1], nd_means[2], nd_means[3],
        spectral_radius, density,
    ], dtype=np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_classical_compressed_batch(
    compressed_graphs: list[CompressedGraph],
) -> np.ndarray:
    """Extract classical compressed features for a batch.

    Returns
    -------
    np.ndarray, shape (N, 12)
    """
    features = [extract_classical_compressed_features(cg) for cg in compressed_graphs]
    return np.stack(features)
