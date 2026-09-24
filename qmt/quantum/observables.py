"""Observable extraction from CGQP circuits using Qiskit EstimatorV2.

Extracts permutation-resistant pooled statistics:
  - mu_Z, sigma_Z, mu_X, sigma_X (4 features)
  - C_XX, C_ZZ (2 features)
  - C_contrast (1 feature)
  - V_H (1 feature)
Total: ~8 features per probe.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2

from ..compression.base import CompressedGraph
from ..compression.quotient import get_edge_list, get_node_weights
from .cgqp import build_cgqp_circuit


def _make_z_observables(k: int) -> list[SparsePauliOp]:
    """Create single-qubit Z observables for each qubit."""
    ops = []
    for i in range(k):
        label = ["I"] * k
        label[k - 1 - i] = "Z"
        ops.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return ops


def _make_x_observables(k: int) -> list[SparsePauliOp]:
    """Create single-qubit X observables for each qubit."""
    ops = []
    for i in range(k):
        label = ["I"] * k
        label[k - 1 - i] = "X"
        ops.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return ops


def _make_zz_observables(k: int, edges: list[tuple[int, int, float]]) -> list[SparsePauliOp]:
    """Create ZZ observables for each edge."""
    ops = []
    for a, b, _ in edges:
        label = ["I"] * k
        label[k - 1 - a] = "Z"
        label[k - 1 - b] = "Z"
        ops.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return ops


def _make_xx_observables(k: int, edges: list[tuple[int, int, float]]) -> list[SparsePauliOp]:
    """Create XX observables for each edge."""
    ops = []
    for a, b, _ in edges:
        label = ["I"] * k
        label[k - 1 - a] = "X"
        label[k - 1 - b] = "X"
        ops.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return ops


def _make_nonedge_observables(k: int, edges: list[tuple[int, int, float]]) -> list[SparsePauliOp]:
    """Create ZZ observables for non-edges (for contrast)."""
    edge_set = set()
    for a, b, _ in edges:
        edge_set.add((min(a, b), max(a, b)))

    nonedges = []
    for a in range(k):
        for b in range(a + 1, k):
            if (a, b) not in edge_set:
                nonedges.append((a, b, 0.0))

    return _make_zz_observables(k, nonedges), _make_xx_observables(k, nonedges)


def extract_observables(
    cg: CompressedGraph,
    probe_params: dict,
    w: np.ndarray | None = None,
    estimator: EstimatorV2 | None = None,
    circuit: QuantumCircuit | None = None,
) -> np.ndarray:
    """Extract quantum features from a compressed graph using a CGQP probe.

    Parameters
    ----------
    cg : CompressedGraph
        Compressed graph.
    probe_params : dict
        Probe parameters (gamma, beta, layers).
    w : np.ndarray, optional
        Node descriptor projection weights.
    estimator : EstimatorV2, optional
        Qiskit estimator. If None, creates a new one.
    circuit : QuantumCircuit, optional
        Pre-built circuit. If None, builds from cg using build_cgqp_circuit.
        This allows passing custom circuits (e.g., quantum controls Q0-Q4).

    Returns
    -------
    np.ndarray, shape (8,)
        Quantum feature vector: [mu_Z, sigma_Z, mu_X, sigma_X, C_XX, C_ZZ, C_contrast, V_H]
    """
    k = cg.k
    if estimator is None:
        estimator = EstimatorV2()

    # Build circuit (or use provided one)
    if circuit is not None:
        qc = circuit
    else:
        qc = build_cgqp_circuit(cg, probe_params, w)
    edges = get_edge_list(cg)
    h = get_node_weights(cg, w)

    # Build all observables
    z_ops = _make_z_observables(k)
    x_ops = _make_x_observables(k)
    zz_edge_ops = _make_zz_observables(k, edges)
    xx_edge_ops = _make_xx_observables(k, edges)

    # Run estimation
    all_ops = z_ops + x_ops + zz_edge_ops + xx_edge_ops
    job = estimator.run([(qc, all_ops)])
    result = job.result()
    evs = result[0].data.evs

    n_z = len(z_ops)
    n_x = len(x_ops)
    n_zz_edge = len(zz_edge_ops)
    n_xx_edge = len(xx_edge_ops)

    z_vals = np.array(evs[:n_z], dtype=float)
    x_vals = np.array(evs[n_z:n_z + n_x], dtype=float)
    zz_edge_vals = np.array(evs[n_z + n_x:n_z + n_x + n_zz_edge], dtype=float)
    xx_edge_vals = np.array(evs[n_z + n_x + n_zz_edge:], dtype=float)

    # Pooled statistics
    mu_Z = float(np.mean(z_vals))
    sigma_Z = float(np.std(z_vals))
    mu_X = float(np.mean(x_vals))
    sigma_X = float(np.std(x_vals))

    # Weighted correlations
    edge_weights = np.array([j for _, _, j in edges], dtype=float)
    if len(edge_weights) > 0 and edge_weights.sum() > 1e-10:
        w_norm = np.abs(edge_weights) / (np.abs(edge_weights).sum() + 1e-10)
        C_XX = float(np.sum(w_norm * xx_edge_vals)) if len(xx_edge_vals) > 0 else 0.0
        C_ZZ = float(np.sum(w_norm * zz_edge_vals)) if len(zz_edge_vals) > 0 else 0.0
    else:
        C_XX = 0.0
        C_ZZ = 0.0

    # Contrast: edges vs non-edges (for ZZ)
    if k > 2 and len(edges) > 0:
        edge_set = set()
        for a, b, _ in edges:
            edge_set.add((min(a, b), max(a, b)))
        nonedge_pairs = [(a, b) for a in range(k) for b in range(a + 1, k)
                         if (a, b) not in edge_set]
        if len(nonedge_pairs) > 0:
            nonedge_zz_ops = []
            for a, b in nonedge_pairs:
                label = ["I"] * k
                label[k - 1 - a] = "Z"
                label[k - 1 - b] = "Z"
                nonedge_zz_ops.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
            job2 = estimator.run([(qc, nonedge_zz_ops)])
            result2 = job2.result()
            nonedge_zz_vals = np.array(result2[0].data.evs, dtype=float)
            C_edge = float(np.mean(zz_edge_vals)) if len(zz_edge_vals) > 0 else 0.0
            C_nonedge = float(np.mean(nonedge_zz_vals))
            C_contrast = C_edge - C_nonedge
        else:
            C_contrast = 0.0
    else:
        C_contrast = 0.0

    # Quantum variance: V_H = <H^2> - <H>^2
    # Approximate using measured observables
    # H = sum J_ab X_a X_b + sum h_a Z_a
    H_mean = 0.0
    if len(edge_weights) > 0:
        H_mean += float(np.sum(edge_weights * xx_edge_vals))
    H_mean += float(np.sum(h * z_vals))

    # V_H approximation: variance of individual terms
    var_terms = []
    if len(edge_weights) > 0:
        var_terms.extend(edge_weights * (xx_edge_vals - np.mean(xx_edge_vals)) ** 2)
    if len(h) > 0:
        var_terms.extend(h * (z_vals - np.mean(z_vals)) ** 2)
    V_H = float(np.sum(var_terms)) if len(var_terms) > 0 else 0.0

    features = np.array([
        mu_Z, sigma_Z, mu_X, sigma_X, C_XX, C_ZZ, C_contrast, V_H,
    ], dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_observables_batch(
    compressed_graphs: list[CompressedGraph],
    probe_params: dict,
    w: np.ndarray | None = None,
    circuits: list[QuantumCircuit] | None = None,
) -> np.ndarray:
    """Extract quantum features for a batch of compressed graphs.

    Parameters
    ----------
    circuits : list[QuantumCircuit], optional
        Pre-built circuits (one per graph). If None, builds from compressed_graphs.

    Returns
    -------
    np.ndarray, shape (N, 8)
        Quantum feature matrix.
    """
    estimator = EstimatorV2()
    features = []
    for i, cg in enumerate(compressed_graphs):
        circ = circuits[i] if circuits is not None else None
        f = extract_observables(cg, probe_params, w, estimator, circuit=circ)
        features.append(f)
    return np.stack(features)
