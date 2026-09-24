"""Dataset splits for the Quantum Micro-Teacher program.

Training: N=30-80 from training families.
OOD-size: N=100-300 from same training families.
OOD-family: N=30-80 from OOD-only families.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from .generators import generate_dataset, TRAINING_FAMILIES, OOD_FAMILIES


def make_splits(
    n_train: int = 2000,
    n_ood_size: int = 0,
    n_ood_family: int = 0,
    seed: int = 42,
    train_n_range: tuple[int, int] = (30, 80),
    ood_size_range: tuple[int, int] = (100, 300),
    ood_family_n_range: tuple[int, int] = (30, 80),
) -> dict[str, list[nx.Graph]]:
    """Create train/val/OOD-size/OOD-family splits.

    Returns
    -------
    dict[str, list[nx.Graph]]
        Keys: "train", "ood_size", "ood_family".
        Train is split internally into train/val by the caller.
    """
    splits: dict[str, list[nx.Graph]] = {}

    splits["train"] = generate_dataset(
        TRAINING_FAMILIES, n_train, n_range=train_n_range, seed_start=seed,
    )

    if n_ood_size > 0:
        splits["ood_size"] = generate_dataset(
            TRAINING_FAMILIES, n_ood_size, n_range=ood_size_range,
            seed_start=seed + 10_000_000,
        )

    if n_ood_family > 0:
        splits["ood_family"] = generate_dataset(
            OOD_FAMILIES, n_ood_family, n_range=ood_family_n_range,
            seed_start=seed + 20_000_000,
        )

    return splits


def train_val_split(
    graphs: list[nx.Graph], val_frac: float = 0.15, seed: int = 42,
) -> tuple[list[nx.Graph], list[nx.Graph]]:
    """Split a list of graphs into train/val by random permutation."""
    rng = np.random.default_rng(seed)
    n = len(graphs)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]
