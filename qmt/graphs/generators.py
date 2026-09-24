"""Graph generators for the Quantum Micro-Teacher program.

Training families: ER, random regular, Watts-Strogatz, stochastic block models.
OOD families: Barabasi-Albert, random geometric, grid+rewiring, heavy-tailed config.
"""
from __future__ import annotations

import numpy as np
import networkx as nx


TRAINING_FAMILIES = ["er", "regular", "ws", "sbm"]
OOD_FAMILIES = ["ba", "geometric", "grid_rewire", "config"]


def generate_graph(family: str, n: int, seed: int, **params) -> nx.Graph:
    """Generate a single graph from the specified family.

    Parameters
    ----------
    family : str
        One of TRAINING_FAMILIES or OOD_FAMILIES.
    n : int
        Number of nodes.
    seed : int
        Random seed.
    **params : dict
        Family-specific parameters.

    Returns
    -------
    nx.Graph
        Graph with metadata attributes: family, n, seed.
    """
    rng = np.random.default_rng(seed)
    if family == "er":
        p = params.get("p", rng.uniform(0.1, 0.5))
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    elif family == "regular":
        d = params.get("d", int(rng.choice([3, 4, 5, 6])))
        d = min(d, n - 1)
        try:
            G = nx.random_regular_graph(d, n, seed=seed)
        except nx.NetworkXError:
            G = nx.erdos_renyi_graph(n, d / max(n - 1, 1), seed=seed)
    elif family == "ws":
        k = params.get("k", int(rng.choice([4, 6, 8])))
        k = min(k, n - 1)
        if k % 2 != 0:
            k += 1
        p = params.get("p", rng.uniform(0.05, 0.3))
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    elif family == "sbm":
        n_communities = params.get("n_communities", int(rng.choice([2, 3, 4])))
        n_communities = min(n_communities, n // 5)
        n_communities = max(n_communities, 2)
        sizes = [n // n_communities] * n_communities
        sizes[-1] += n - sum(sizes)
        intra_p = params.get("intra_p", rng.uniform(0.4, 0.8))
        inter_p = params.get("inter_p", rng.uniform(0.02, 0.1))
        p_matrix = np.full((n_communities, n_communities), inter_p, dtype=float)
        np.fill_diagonal(p_matrix, intra_p)
        G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
    elif family == "ba":
        m = params.get("m", int(rng.choice([2, 3, 4])))
        m = min(m, n - 1)
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif family == "geometric":
        radius = params.get("radius", rng.uniform(0.15, 0.3))
        G = nx.random_geometric_graph(n, radius, seed=seed)
    elif family == "grid_rewire":
        side = int(np.sqrt(n))
        n = side * side
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
        p_rewire = params.get("p_rewire", rng.uniform(0.05, 0.2))
        G = nx.watts_strogatz_graph(n, 4, p_rewire, seed=seed)
    elif family == "config":
        alpha = params.get("alpha", rng.uniform(1.5, 3.0))
        degrees = rng.zipf(alpha, size=n).clip(1, n - 1)
        if degrees.sum() % 2 != 0:
            degrees[0] += 1
        G = nx.configuration_model(degrees.tolist(), seed=seed)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.convert_node_labels_to_integers(G)
    else:
        raise ValueError(f"Unknown family: {family}")

    G = nx.convert_node_labels_to_integers(G)
    G.graph["family"] = family
    G.graph["n"] = G.number_of_nodes()
    G.graph["seed"] = seed
    return G


def generate_dataset(
    families: list[str],
    n_graphs: int,
    n_range: tuple[int, int] = (30, 80),
    seed_start: int = 0,
    **family_params,
) -> list[nx.Graph]:
    """Generate a dataset of graphs from multiple families.

    Parameters
    ----------
    families : list[str]
        Graph families to sample from.
    n_graphs : int
        Total number of graphs to generate.
    n_range : tuple[int, int]
        (min, max) number of nodes.
    seed_start : int
        Starting seed for reproducibility.
    family_params : dict
        Parameters passed to generate_graph.

    Returns
    -------
    list[nx.Graph]
        List of generated graphs.
    """
    rng = np.random.default_rng(seed_start)
    graphs = []
    for i in range(n_graphs):
        family = families[i % len(families)]
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        seed = seed_start + i
        G = generate_graph(family, n, seed, **family_params)
        graphs.append(G)
    return graphs
