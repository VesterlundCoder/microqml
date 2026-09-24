"""Compressed Graph Quantum Probe (CGQP).

Builds Qiskit circuits with AQT-native gates (RXX, RY, RZ) that probe
the structure of a globally compressed graph.

Circuit structure:
  1. Encoding: R_y(theta_a) where theta_a = pi * clip(w^T z_a, -1, 1)
  2. Interaction: R_XX(2 * gamma * J_ab) for all supernode pairs
  3. Field: R_Z(2 * beta * h_a) for each qubit
  4. Repeat for multi-layer probes
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from ..compression.base import CompressedGraph
from ..compression.quotient import get_edge_list, get_node_weights


def build_cgqp_circuit(
    cg: CompressedGraph,
    probe_params: dict,
    w: np.ndarray | None = None,
) -> QuantumCircuit:
    """Build a CGQP quantum circuit for a compressed graph.

    Parameters
    ----------
    cg : CompressedGraph
        Compressed graph with k supernodes.
    probe_params : dict
        Probe parameters with keys:
        - "gamma": interaction strength (or list for multi-layer)
        - "beta": field strength (or list for multi-layer)
        - "layers": number of layers (default 1)
    w : np.ndarray, shape (4,), optional
        Projection weights for node descriptors. Default: uniform.

    Returns
    -------
    QuantumCircuit
        Circuit with k qubits using AQT-native gates.
    """
    k = cg.k
    qc = QuantumCircuit(k)

    # Encoding: R_y(theta_a) for each qubit
    h = get_node_weights(cg, w)
    theta = np.pi * np.clip(h, -1.0, 1.0)
    for a in range(k):
        qc.ry(float(theta[a]), a)

    # Get edges
    edges = get_edge_list(cg)

    # Probe parameters
    layers = probe_params.get("layers", 1)
    gammas = probe_params.get("gamma", 0.45)
    betas = probe_params.get("beta", 0.60)

    if not isinstance(gammas, list):
        gammas = [gammas] * layers
    if not isinstance(betas, list):
        betas = [betas] * layers

    for layer in range(layers):
        gamma = gammas[min(layer, len(gammas) - 1)]
        beta = betas[min(layer, len(betas) - 1)]

        # Interaction: R_XX(2 * gamma * J_ab)
        for a, b, j_ab in edges:
            angle = 2.0 * gamma * j_ab
            qc.rxx(float(angle), a, b)

        # Field: R_Z(2 * beta * h_a)
        for a in range(k):
            angle = 2.0 * beta * h[a]
            qc.rz(float(angle), a)

    return qc


def build_cgqp_circuits_batch(
    compressed_graphs: list[CompressedGraph],
    probe_params: dict,
    w: np.ndarray | None = None,
) -> list[QuantumCircuit]:
    """Build CGQP circuits for a batch of compressed graphs."""
    return [build_cgqp_circuit(cg, probe_params, w) for cg in compressed_graphs]
