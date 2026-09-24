"""D2: Boundary dataset — oversample graphs where solvers disagree.

Solver margin: m_i = (s_(2) - s_(1)) / (|s_(1)| + eps)
where s_(1), s_(2) are best and second-best solver cut values.

Small margin → hard to select solver → boundary case.
We oversample from low-margin regions.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Callable
from .generators import generate_graph, TRAINING_FAMILIES


def compute_solver_margin(cut_values: np.ndarray) -> float:
    """Compute solver margin for a single graph.

    m = (s_(2) - s_(1)) / (|s_(1)| + eps)

    Small margin → solvers are close → boundary case.
    """
    sorted_cuts = np.sort(cut_values)[::-1]  # descending
    if len(sorted_cuts) < 2:
        return 1.0
    best = sorted_cuts[0]
    second = sorted_cuts[1]
    eps = 1e-10
    return float((best - second) / (abs(best) + eps))


def generate_boundary_dataset(
    n_target: int = 5000,
    n_range: tuple[int, int] = (30, 80),
    seed: int = 42,
    solver_fn: Callable | None = None,
    solver_budget: int = 50,
    max_attempts: int | None = None,
    margin_threshold: float = 0.05,
    dominance_threshold: float = 0.45,
) -> tuple[list[nx.Graph], dict]:
    """Generate a boundary dataset by oversampling low-margin graphs.

    Strategy:
    1. Generate graphs from training families
    2. Run solver portfolio on each
    3. Compute margin
    4. Keep graphs with margin < threshold (boundary cases)
    5. Also keep some high-margin graphs for contrast
    6. Check solver dominance < threshold

    Parameters
    ----------
    n_target : int
        Target number of graphs in the final dataset.
    margin_threshold : float
        Graphs with margin below this are "boundary" cases.
    dominance_threshold : float
        Max fraction any single solver can win.

    Returns
    -------
    (graphs, metadata)
    """
    from ..solvers.portfolio import run_portfolio, check_solver_dominance, SOLVER_NAMES

    if solver_fn is None:
        solver_fn = run_portfolio

    if max_attempts is None:
        max_attempts = n_target * 5

    rng = np.random.default_rng(seed)
    graphs = []
    solver_results = []
    margins = []
    n_boundary = 0
    n_easy = 0

    # Target: ~60% boundary, ~40% easy (for contrast)
    n_boundary_target = int(n_target * 0.6)
    n_easy_target = n_target - n_boundary_target

    for i in range(max_attempts):
        if len(graphs) >= n_target:
            break
        if i >= max_attempts:
            break

        family = TRAINING_FAMILIES[i % len(TRAINING_FAMILIES)]
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        g_seed = seed + i * 7 + 1
        G = generate_graph(family, n, g_seed)

        sr = solver_fn(G, budget=solver_budget, seed=g_seed)
        margin = compute_solver_margin(sr.cut_values)

        is_boundary = margin < margin_threshold

        if is_boundary and n_boundary < n_boundary_target:
            graphs.append(G)
            solver_results.append(sr)
            margins.append(margin)
            n_boundary += 1
        elif not is_boundary and n_easy < n_easy_target:
            graphs.append(G)
            solver_results.append(sr)
            margins.append(margin)
            n_easy += 1

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{max_attempts}, kept {len(graphs)}/{n_target} "
                  f"(boundary={n_boundary}, easy={n_easy})")

    # Check dominance
    dominance = check_solver_dominance(solver_results)

    metadata = {
        "n_graphs": len(graphs),
        "n_boundary": n_boundary,
        "n_easy": n_easy,
        "margin_threshold": margin_threshold,
        "margins": margins,
        "margin_mean": float(np.mean(margins)) if margins else 0.0,
        "margin_std": float(np.std(margins)) if margins else 0.0,
        "solver_dominance": dominance,
        "n_attempts": i + 1,
    }

    return graphs, metadata


def generate_controlled_families(
    n_per_family: int = 500,
    n_range: tuple[int, int] = (30, 80),
    seed: int = 42,
    families: list[str] | None = None,
) -> dict[str, list[nx.Graph]]:
    """D3: Generate controlled graph families for OOD experiments.

    Each family gets exactly n_per_family graphs.
    Returns a dict mapping family name → list of graphs.
    """
    if families is None:
        families = ["er", "regular", "ws", "sbm", "ba", "geometric", "grid_rewire", "config"]

    result = {}
    for fam_idx, family in enumerate(families):
        graphs = []
        for i in range(n_per_family):
            n = int(np.random.default_rng(seed + fam_idx * 10000 + i).integers(
                n_range[0], n_range[1] + 1))
            G = generate_graph(family, n, seed + fam_idx * 10000 + i)
            graphs.append(G)
        result[family] = graphs
        print(f"  {family}: {len(graphs)} graphs")

    return result


def generate_size_ood(
    train_n_range: tuple[int, int] = (30, 80),
    test_n_ranges: list[tuple[int, int]] = None,
    n_train: int = 1000,
    n_test_per_range: int = 500,
    seed: int = 42,
) -> dict[str, list[nx.Graph]]:
    """Generate size-OOD datasets.

    Train on small graphs, test on progressively larger ones.
    """
    if test_n_ranges is None:
        test_n_ranges = [(100, 150), (150, 250), (250, 400)]

    from .generators import generate_dataset, TRAINING_FAMILIES

    result = {}
    result["train"] = generate_dataset(
        TRAINING_FAMILIES, n_train, n_range=train_n_range, seed_start=seed,
    )

    for idx, (n_min, n_max) in enumerate(test_n_ranges):
        key = f"test_{n_min}_{n_max}"
        result[key] = generate_dataset(
            TRAINING_FAMILIES, n_test_per_range,
            n_range=(n_min, n_max), seed_start=seed + (idx + 1) * 1000000,
        )

    return result


def generate_family_ood(
    train_families: list[str] | None = None,
    test_families: list[str] | None = None,
    n_train: int = 1000,
    n_test: int = 500,
    n_range: tuple[int, int] = (30, 80),
    seed: int = 42,
) -> dict[str, list[nx.Graph]]:
    """Generate family-OOD datasets.

    Train on some families, test on held-out families.
    """
    if train_families is None:
        train_families = ["er", "ws", "regular"]
    if test_families is None:
        test_families = ["sbm", "ba", "geometric"]

    from .generators import generate_dataset

    result = {}
    result["train"] = generate_dataset(
        train_families, n_train, n_range=n_range, seed_start=seed,
    )

    for fam in test_families:
        result[f"test_{fam}"] = generate_dataset(
            [fam], n_test, n_range=n_range, seed_start=seed + 5000000 + hash(fam) % 100000,
        )

    return result
