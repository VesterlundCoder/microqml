"""Spectral coarsening: use normalized Laplacian eigenvectors + k-means."""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from sklearn.cluster import KMeans

from .base import Compressor, CompressedGraph, build_quotient


class SpectralCompressor(Compressor):
    """Spectral coarsening via normalized Laplacian eigenvectors + k-means."""

    @property
    def name(self) -> str:
        return "spectral"

    def compress(self, G: nx.Graph, k: int) -> CompressedGraph:
        n = G.number_of_nodes()
        nodes = sorted(G.nodes())

        if n <= k:
            # Each node is its own supernode
            S = np.eye(n, k, dtype=float)
            return build_quotient(G, S, k, method=self.name)

        A = nx.adjacency_matrix(G, nodelist=nodes).astype(float)
        A = np.asarray(A.todense())
        degrees = A.sum(axis=1)
        degrees_safe = np.where(degrees > 0, degrees, 1.0)

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
        L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        L_norm = np.nan_to_num(L_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute lowest k eigenvectors (skip the trivial first)
        n_eigs = min(k, n - 1)
        try:
            eigvals, eigvecs = eigh(L_norm, subset_by_index=[0, n_eigs - 1])
        except Exception:
            eigvals, eigvecs = eigh(L_norm)
            eigvecs = eigvecs[:, :n_eigs]

        # Use eigenvectors 1..k-1 (skip the constant first eigenvector)
        embedding = eigvecs[:, :n_eigs]
        if n_eigs > 1:
            embedding = embedding[:, 1:]  # skip trivial
        if embedding.shape[1] < 1:
            embedding = eigvecs[:, :1]

        # K-means cluster into k groups
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(embedding)

        # Build partition matrix
        S = np.zeros((n, k), dtype=float)
        for i, label in enumerate(labels):
            S[i, label] = 1.0

        # Handle empty clusters by assigning orphan nodes
        for a in range(k):
            if S[:, a].sum() == 0:
                # Find largest cluster and split
                sizes = S.sum(axis=0)
                largest = np.argmax(sizes)
                members = np.where(S[:, largest] > 0)[0]
                if len(members) > 1:
                    move = members[len(members) // 2]
                    S[move, largest] = 0
                    S[move, a] = 1.0

        return build_quotient(G, S, k, method=self.name)
