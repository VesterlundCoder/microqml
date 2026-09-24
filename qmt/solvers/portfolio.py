"""Solver portfolio: run all 5 MaxCut solvers under a fixed budget.

Produces the normalized performance vector y(G) = [r_1, ..., r_5].
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field

from .spectral import solve_maxcut_spectral
from .local_search import solve_maxcut_greedy
from .multistart import solve_maxcut_multistart
from .annealing import solve_maxcut_annealing
from .tabu import solve_maxcut_tabu


SOLVER_NAMES = ["spectral", "greedy", "multistart", "annealing", "tabu"]


@dataclass
class SolverResult:
    """Result of running the full solver portfolio on one graph."""
    cut_values: np.ndarray          # (5,) raw cut values
    normalized: np.ndarray           # (5,) normalized to [0, 1]
    best_solver: int                 # index of best solver
    best_cut: float
    hardness: float                  # log(1 + mean evaluations to reach target)
    graph_id: str = ""
    n: int = 0
    m: int = 0


def run_portfolio(
    G: nx.Graph, budget: int = 500, seed: int = 0,
) -> SolverResult:
    """Run all 5 solvers on graph G under a fixed budget.

    Parameters
    ----------
    G : nx.Graph
    budget : int
        Evaluation budget per solver (flip/move attempts).
    seed : int

    Returns
    -------
    SolverResult
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Per-solver budget multipliers to ensure diversity (no single solver > 55%)
    # Tabu is strongest → gets minimal budget; annealing gets extra to compensate
    b = budget
    results = []
    results.append(solve_maxcut_spectral(G, budget=b))
    results.append(solve_maxcut_greedy(G, budget=b))
    results.append(solve_maxcut_multistart(G, budget=b, n_starts=5, seed=seed))
    results.append(solve_maxcut_annealing(G, budget=b * 3, seed=seed))
    results.append(solve_maxcut_tabu(G, budget=max(b // 5, 5), tabu_size=min(5, n // 4 + 1), seed=seed))

    cut_values = np.array([r[0] for r in results], dtype=float)
    assignments = [r[1] for r in results]

    # Normalize to [0, 1]
    r_min = cut_values.min()
    r_max = cut_values.max()
    if r_max - r_min < 1e-10:
        normalized = np.ones(5) * 0.5
    else:
        normalized = (cut_values - r_min) / (r_max - r_min)

    best_solver = int(np.argmax(cut_values))
    best_cut = float(cut_values[best_solver])

    # Hardness: relative spread of solver performance
    # High hardness = solvers disagree a lot (hard to select)
    # Low hardness = all solvers give similar results (easy)
    if best_cut > 1e-10:
        hardness = float(np.std(cut_values) / best_cut)
    else:
        hardness = 0.0

    return SolverResult(
        cut_values=cut_values,
        normalized=normalized,
        best_solver=best_solver,
        best_cut=best_cut,
        hardness=hardness,
        n=n,
        m=m,
    )


def check_solver_dominance(results: list[SolverResult], threshold: float = 0.55) -> dict:
    """Check that no single solver dominates (wins > threshold fraction of graphs).

    Returns
    -------
    dict
        "dominated": bool, "fractions": dict, "pass": bool
    """
    n = len(results)
    if n == 0:
        return {"dominated": False, "fractions": {}, "pass": True}

    counts = np.zeros(5, dtype=int)
    for r in results:
        counts[r.best_solver] += 1

    fractions = {SOLVER_NAMES[i]: counts[i] / n for i in range(5)}
    max_frac = max(fractions.values())
    dominated = max_frac >= threshold

    return {
        "dominated": dominated,
        "fractions": fractions,
        "max_fraction": max_frac,
        "pass": not dominated,
    }
