"""Random balanced partition compressor (negative control)."""
from __future__ import annotations
import numpy as np
import networkx as nx
from .base import Compressor, CompressedGraph, build_quotient


class RandomCompressor(Compressor):
    """Random balanced partition preserving approximate cluster sizes."""

    @property
    def name(self) -> str:
        return "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def compress(self, G: nx.Graph, k: int) -> CompressedGraph:
        n = G.number_of_nodes()
        nodes = sorted(G.nodes())

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)

        # Balanced partition
        S = np.zeros((n, k), dtype=float)
        cluster_size = n // k
        for a in range(k):
            start = a * cluster_size
            end = start + cluster_size
            if a == k - 1:
                end = n
            for i in range(start, end):
                S[perm[i], a] = 1.0

        return build_quotient(G, S, k, method=self.name)
