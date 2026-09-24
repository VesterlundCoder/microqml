"""Feature attribution and solver-boundary analysis.

1. Per-graph gain: g_i = L_C(i) - L_{C+Q}(i)
   Positive g_i means Q improved prediction for graph i.

2. Regress g_i against graph properties to find which structural regimes
   benefit from quantum features.

3. Solver-boundary analysis: split dataset by solver margin and plot
   ΔR² as a function of margin.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import networkx as nx
from typing import Callable


def compute_per_graph_gain(
    X_classical: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute per-graph gain: g_i = L_C(i) - L_{C+Q}(i).

    Uses cross-validation: for each graph, the prediction comes from
    the fold whose validation set contains that graph.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-graph gain. Positive = Q helped.
    """
    from sklearn.model_selection import KFold

    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    X_classical = np.nan_to_num(np.asarray(X_classical, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Q = np.nan_to_num(np.asarray(Q, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    n = X_classical.shape[0]
    gains = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for tr_idx, va_idx in kf.split(X_classical):
        Xc_tr, Xc_va = X_classical[tr_idx], X_classical[va_idx]
        Q_tr, Q_va = Q[tr_idx], Q[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        # Classical model
        sc_c = StandardScaler().fit(Xc_tr)
        Xc_tr_s = sc_c.transform(Xc_tr)
        Xc_va_s = sc_c.transform(Xc_va)

        m_c = Ridge(alpha=1.0)
        m_c.fit(Xc_tr_s, Y_tr)
        pred_c = m_c.predict(Xc_va_s)
        if pred_c.ndim == 1:
            pred_c = pred_c.reshape(-1, 1)

        # Combined model
        sc_q = StandardScaler().fit(Q_tr)
        Q_tr_s = sc_q.transform(Q_tr)
        Q_va_s = sc_q.transform(Q_va)

        X_tr_combined = np.hstack([Xc_tr_s, Q_tr_s])
        X_va_combined = np.hstack([Xc_va_s, Q_va_s])

        m_cq = Ridge(alpha=1.0)
        m_cq.fit(X_tr_combined, Y_tr)
        pred_cq = m_cq.predict(X_va_combined)
        if pred_cq.ndim == 1:
            pred_cq = pred_cq.reshape(-1, 1)

        # Per-graph squared error
        loss_c = np.sum((Y_va - pred_c) ** 2, axis=1)
        loss_cq = np.sum((Y_va - pred_cq) ** 2,1)
        gains[va_idx] = loss_c - loss_cq

    return gains


def compute_graph_properties(graphs: list[nx.Graph]) -> dict:
    """Compute structural properties for a list of graphs.

    Returns
    -------
    dict mapping property name → np.ndarray of shape (N,)
    """
    properties = {
        "n_nodes": [],
        "n_edges": [],
        "edge_density": [],
        "degree_mean": [],
        "degree_std": [],
        "degree_variance": [],
        "clustering_mean": [],
        "spectral_gap": [],
        "diameter": [],
        "assortativity": [],
        "modularity": [],
        "n_components": [],
    }

    for G in graphs:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        max_edges = n * (n - 1) / 2

        degrees = [d for _, d in G.degree()]
        deg_mean = np.mean(degrees) if degrees else 0
        deg_std = np.std(degrees) if degrees else 0

        properties["n_nodes"].append(n)
        properties["n_edges"].append(m)
        properties["edge_density"].append(m / max(max_edges, 1))
        properties["degree_mean"].append(deg_mean)
        properties["degree_std"].append(deg_std)
        properties["degree_variance"].append(deg_std ** 2)

        try:
            properties["clustering_mean"].append(nx.average_clustering(G))
        except Exception:
            properties["clustering_mean"].append(0.0)

        try:
            eigenvals = nx.laplacian_spectrum(G)
            sorted_eigenvals = np.sort(eigenvals)
            gap = sorted_eigenvals[1] if len(sorted_eigenvals) > 1 else 0
            properties["spectral_gap"].append(float(gap))
        except Exception:
            properties["spectral_gap"].append(0.0)

        try:
            if nx.is_connected(G):
                properties["diameter"].append(nx.diameter(G))
            else:
                properties["diameter"].append(-1)
        except Exception:
            properties["diameter"].append(-1)

        try:
            properties["assortativity"].append(nx.degree_assortativity_coefficient(G))
        except Exception:
            properties["assortativity"].append(0.0)

        try:
            if n > 2 and m > 0:
                communities = nx.community.greedy_modularity_communities(G)
                properties["modularity"].append(nx.community.modularity(G, communities))
            else:
                properties["modularity"].append(0.0)
        except Exception:
            properties["modularity"].append(0.0)

        properties["n_components"].append(nx.number_connected_components(G))

    return {k: np.array(v) for k, v in properties.items()}


def attribute_gain_to_properties(
    gains: np.ndarray,
    properties: dict,
) -> dict:
    """Regress per-graph gains against graph properties.

    Returns
    -------
    dict with regression coefficients and R²
    """
    prop_names = list(properties.keys())
    X = np.column_stack([properties[name] for name in prop_names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model = Ridge(alpha=1.0)
    model.fit(Xs, y)
    pred = model.predict(Xs)

    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - pred) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-20)

    # Correlation of each property with gains
    correlations = {}
    for name in prop_names:
        vals = properties[name]
        if np.std(vals) > 1e-10 and np.std(y) > 1e-10:
            r = np.corrcoef(vals, y)[0, 1]
            correlations[name] = float(r) if np.isfinite(r) else 0.0
        else:
            correlations[name] = 0.0

    return {
        "r2": float(r2),
        "coefficients": {name: float(model.coef_[i]) for i, name in enumerate(prop_names)},
        "correlations": correlations,
        "mean_gain": float(np.mean(y)),
        "median_gain": float(np.median(y)),
        "n_positive": int(np.sum(y > 0)),
        "n_negative": int(np.sum(y < 0)),
        "n_zero": int(np.sum(np.abs(y) < 1e-10)),
    }


def solver_boundary_analysis(
    gains: np.ndarray,
    solver_margins: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Analyze ΔR² as a function of solver margin.

    Splits graphs into bins by solver margin and computes mean gain per bin.

    Parameters
    ----------
    gains : per-graph gain (positive = Q helped)
    solver_margins : per-graph solver margin (small = hard to select solver)
    """
    gains = np.asarray(gains, dtype=float)
    margins = np.asarray(solver_margins, dtype=float)

    # Sort by margin
    sort_idx = np.argsort(margins)
    sorted_margins = margins[sort_idx]
    sorted_gains = gains[sort_idx]

    # Bin
    bin_size = max(1, len(gains) // n_bins)
    bins = []
    for i in range(0, len(gains), bin_size):
        end = min(i + bin_size, len(gains))
        bin_margins = sorted_margins[i:end]
        bin_gains = sorted_gains[i:end]
        bins.append({
            "margin_mean": float(np.mean(bin_margins)),
            "margin_range": [float(np.min(bin_margins)), float(np.max(bin_margins))],
            "gain_mean": float(np.mean(bin_gains)),
            "gain_std": float(np.std(bin_gains)),
            "n": len(bin_gains),
        })

    # Classify: easy/medium/hard/boundary
    m_median = np.median(margins)
    m_quartile = np.percentile(margins, 25)

    categories = {
        "boundary": gains[margins < m_quartile],
        "medium": gains[(margins >= m_quartile) & (margins < m_median)],
        "easy": gains[margins >= m_median],
    }

    category_stats = {}
    for name, g in categories.items():
        if len(g) > 0:
            category_stats[name] = {
                "n": len(g),
                "gain_mean": float(np.mean(g)),
                "gain_std": float(np.std(g)),
                "n_positive": int(np.sum(g > 0)),
                "frac_positive": float(np.mean(g > 0)),
            }
        else:
            category_stats[name] = {"n": 0, "gain_mean": 0, "gain_std": 0, "n_positive": 0, "frac_positive": 0}

    # Correlation between margin and gain
    if np.std(margins) > 1e-10 and np.std(gains) > 1e-10:
        corr = float(np.corrcoef(margins, gains)[0, 1])
    else:
        corr = 0.0

    return {
        "bins": bins,
        "categories": category_stats,
        "correlation_margin_gain": corr,
        "hypothesis": "ΔR² ↑ when margin → 0" if corr < 0 else "ΔR² ↓ when margin → 0",
        "hypothesis_supported": corr < 0,
    }


def compute_solver_margins(solver_results: list) -> np.ndarray:
    """Compute solver margin for each graph from solver results.

    m_i = (s_(2) - s_(1)) / (|s_(1)| + eps)
    """
    from ..graphs.boundary_datasets import compute_solver_margin
    margins = []
    for sr in solver_results:
        margins.append(compute_solver_margin(sr.cut_values))
    return np.array(margins)
