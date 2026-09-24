"""Quantum controls: falsify the quantum signal from within the quantum circuit.

Controls:
  Q0: No entanglement (separable) — same encoding + fields, no RXX gates
  Q1: Original entangled reservoir (reference)
  Q2: Scrambled connectivity — same gate count/depth, random edge structure
  Q3: Scrambled input — permute which qubit gets which node's data
  Q4: Entanglement sweep — scale RXX angles by e ∈ {0, 0.25, 0.5, 0.75, 1.0}
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from ..compression.base import CompressedGraph
from ..compression.quotient import get_edge_list, get_node_weights


def build_separable_circuit(
    cg: CompressedGraph,
    probe_params: dict,
    w: np.ndarray | None = None,
) -> QuantumCircuit:
    """Q0: No entanglement. Same encoding + RZ fields, NO RXX gates.

    U(x) = ⊗_i U_i(x_i)
    """
    k = cg.k
    qc = QuantumCircuit(k)

    h = get_node_weights(cg, w)
    theta = np.pi * np.clip(h, -1.0, 1.0)
    for a in range(k):
        qc.ry(float(theta[a]), a)

    layers = probe_params.get("layers", 1)
    betas = probe_params.get("beta", 0.60)
    if not isinstance(betas, list):
        betas = [betas] * layers

    for layer in range(layers):
        beta = betas[min(layer, len(betas) - 1)]
        for a in range(k):
            qc.rz(float(2.0 * beta * h[a]), a)

    return qc


def build_scrambled_connectivity_circuit(
    cg: CompressedGraph,
    probe_params: dict,
    w: np.ndarray | None = None,
    seed: int = 42,
) -> QuantumCircuit:
    """Q2: Scrambled connectivity. Same gate count/depth, random edge structure.

    Randomizes which qubit pairs interact while preserving the number of
    RXX gates and their angle distribution.
    """
    k = cg.k
    qc = QuantumCircuit(k)

    h = get_node_weights(cg, w)
    theta = np.pi * np.clip(h, -1.0, 1.0)
    for a in range(k):
        qc.ry(float(theta[a]), a)

    edges = get_edge_list(cg)
    rng = np.random.default_rng(seed)

    # Scramble: assign edges to random qubit pairs
    all_pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    if len(all_pairs) < len(edges):
        # Not enough pairs, just use original
        scrambled_edges = edges
    else:
        chosen = rng.choice(len(all_pairs), size=len(edges), replace=False)
        scrambled_edges = []
        for idx, (a_orig, b_orig, j_ab) in zip(chosen, edges):
            a_new, b_new = all_pairs[idx]
            scrambled_edges.append((a_new, b_new, j_ab))

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

        for a, b, j_ab in scrambled_edges:
            qc.rxx(float(2.0 * gamma * j_ab), a, b)

        for a in range(k):
            qc.rz(float(2.0 * beta * h[a]), a)

    return qc


def build_scrambled_input_circuit(
    cg: CompressedGraph,
    probe_params: dict,
    w: np.ndarray | None = None,
    seed: int = 42,
) -> QuantumCircuit:
    """Q3: Scrambled input. Permute which qubit gets which node's encoding.

    Preserves input distribution but destroys structural correspondence
    between graph structure and quantum dynamics.
    """
    k = cg.k
    qc = QuantumCircuit(k)

    h = get_node_weights(cg, w)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(k)
    h_scrambled = h[perm]

    theta = np.pi * np.clip(h_scrambled, -1.0, 1.0)
    for a in range(k):
        qc.ry(float(theta[a]), a)

    edges = get_edge_list(cg)

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

        for a, b, j_ab in edges:
            qc.rxx(float(2.0 * gamma * j_ab), a, b)

        # Use scrambled h for fields too
        for a in range(k):
            qc.rz(float(2.0 * beta * h_scrambled[a]), a)

    return qc


def build_entanglement_sweep_circuit(
    cg: CompressedGraph,
    probe_params: dict,
    w: np.ndarray | None = None,
    entanglement_strength: float = 1.0,
) -> QuantumCircuit:
    """Q4: Entanglement sweep. Scale RXX angles by factor e.

    e=0 → fully separable (same as Q0)
    e=1 → original entangled (same as Q1)
    e=0.5 → half-strength entanglement

    This allows testing ΔR² = f(e) for a structured relationship.
    """
    k = cg.k
    qc = QuantumCircuit(k)

    h = get_node_weights(cg, w)
    theta = np.pi * np.clip(h, -1.0, 1.0)
    for a in range(k):
        qc.ry(float(theta[a]), a)

    edges = get_edge_list(cg)

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

        # Scale entanglement by e
        for a, b, j_ab in edges:
            angle = 2.0 * gamma * j_ab * entanglement_strength
            qc.rxx(float(angle), a, b)

        for a in range(k):
            qc.rz(float(2.0 * beta * h[a]), a)

    return qc


def build_quantum_control_circuit(
    cg: CompressedGraph,
    probe_params: dict,
    control: str = "Q1",
    w: np.ndarray | None = None,
    seed: int = 42,
    entanglement_strength: float = 1.0,
) -> QuantumCircuit:
    """Dispatch to the appropriate quantum control circuit builder.

    Parameters
    ----------
    control : str
        "Q0" = separable (no entanglement)
        "Q1" = original entangled (reference)
        "Q2" = scrambled connectivity
        "Q3" = scrambled input
        "Q4" = entanglement sweep (use entanglement_strength param)
    """
    if control == "Q0":
        return build_separable_circuit(cg, probe_params, w)
    elif control == "Q1":
        from .cgqp import build_cgqp_circuit
        return build_cgqp_circuit(cg, probe_params, w)
    elif control == "Q2":
        return build_scrambled_connectivity_circuit(cg, probe_params, w, seed)
    elif control == "Q3":
        return build_scrambled_input_circuit(cg, probe_params, w, seed)
    elif control == "Q4":
        return build_entanglement_sweep_circuit(cg, probe_params, w, entanglement_strength)
    else:
        raise ValueError(f"Unknown quantum control: {control}")
