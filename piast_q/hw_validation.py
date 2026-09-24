"""Stage 1B: Hardware validation on the PIAST-Q trapped-ion quantum computer.

Validates that the quantum micro-reservoir signal survives on real hardware.
Generates a small set of graphs, compresses them, builds CGQP circuits,
runs them on PIAST-Q (200 shots/circuit), extracts features from measurement
counts, and compares against simulator baselines.

PIAST-Q constraints:
  - 200 shots per circuit execution (SHOTS_MAX)
  - ~250 jobs/hour; a few hours/week total
  - Native gates: {RZ, R(=RY), RXX} — exact match for CGQP circuits
  - Best quality windows (CEST): Mon 13-17, Tue-Thu 10-17

Usage (from piast_q directory):
  PYTHONPATH=../quantum_micro_teacher/quantum_micro_teacher .venv/bin/python hw_validation.py \
      --graphs 100 --k 8 --probe P2 --shots 200 --conditions Q1 Q0

Or dry-run (simulator only, no hardware):
  PYTHONPATH=../quantum_micro_teacher/quantum_micro_teacher .venv/bin/python hw_validation.py \
      --graphs 100 --k 8 --probe P2 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import EstimatorV2
from qiskit.quantum_info import SparsePauliOp

# QMT imports
from qmt.graphs.splits import make_splits, train_val_split
from qmt.graphs.descriptors import compute_descriptors_batch
from qmt.solvers.portfolio import run_portfolio, check_solver_dominance
from qmt.compression.spectral import SpectralCompressor
from qmt.compression.quotient import get_edge_list, get_node_weights
from qmt.quantum.probes import get_probe
from qmt.quantum.observables import extract_observables
from qmt.quantum.quantum_controls import build_quantum_control_circuit
from qmt.diagnostics.classical_controls import (
    control_a_gaussian_projection,
    control_b_rff,
    control_c_random_reservoir,
)
from qmt.diagnostics.proper_stats import cv_conditional_test

# PiastQ imports (optional — only needed for real hardware)
try:
    from piast_common import SHOTS_MAX, get_backend, load_token, login
    PIASTQ_AVAILABLE = True
except ImportError:
    PIASTQ_AVAILABLE = False
    SHOTS_MAX = 200


# ---------------------------------------------------------------------------
# Circuit construction with measurements
# ---------------------------------------------------------------------------

def build_measured_circuit(cg, probe_params, control="Q1", basis="Z",
                           w=None, seed=42, entanglement_strength=1.0):
    """Build a CGQP circuit with measurements in the specified basis.

    Parameters
    ----------
    cg : CompressedGraph
        Compressed graph.
    probe_params : dict
        Probe parameters.
    control : str
        Quantum control: "Q0", "Q1", "Q2", "Q3", "Q4".
    basis : str
        Measurement basis: "Z" (computational) or "X" (Hadamard basis).
    w : np.ndarray, optional
        Node descriptor projection weights.
    seed : int
        Random seed for scrambled controls.
    entanglement_strength : float
        Entanglement sweep parameter for Q4.

    Returns
    -------
    QuantumCircuit
        Circuit with k qubits + k classical bits, measurements appended.
    """
    qc = build_quantum_control_circuit(
        cg, probe_params, control=control, w=w, seed=seed,
        entanglement_strength=entanglement_strength,
    )
    k = qc.num_qubits
    qc_meas = QuantumCircuit(k, k)
    qc_meas.compose(qc, range(k), inplace=True)

    if basis == "X":
        for a in range(k):
            qc_meas.h(a)

    qc_meas.measure(range(k), range(k))
    return qc_meas


# ---------------------------------------------------------------------------
# Feature extraction from hardware counts
# ---------------------------------------------------------------------------

def counts_to_expectations(counts, k):
    """Convert bitstring counts to single-qubit Z expectation values.

    Returns
    -------
    z_vals : np.ndarray, shape (k,)
        <Z_a> for each qubit a.
    zz_matrix : np.ndarray, shape (k, k)
        <Z_a Z_b> for each pair (symmetric, diagonal=1).
    """
    total = sum(counts.values())
    if total == 0:
        return np.zeros(k), np.eye(k)

    z_vals = np.zeros(k)
    zz_matrix = np.zeros((k, k))
    np.fill_diagonal(zz_matrix, 1.0)

    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring.replace(" ", "")]
        # Qiskit bitstring is little-endian: bit[0] = qubit 0
        if len(bits) < k:
            bits = [0] * (k - len(bits)) + bits
        bits = bits[:k]

        for a in range(k):
            z_a = 1.0 if bits[a] == 0 else -1.0
            z_vals[a] += count * z_a / total

        for a in range(k):
            for b in range(a + 1, k):
                z_a = 1.0 if bits[a] == 0 else -1.0
                z_b = 1.0 if bits[b] == 0 else -1.0
                zz_matrix[a, b] += count * z_a * z_b / total
                zz_matrix[b, a] = zz_matrix[a, b]

    return z_vals, zz_matrix


def extract_features_from_counts(z_counts, x_counts, cg, probe_params, w=None):
    """Extract the 8-dim quantum feature vector from Z and X measurement counts.

    Features: [mu_Z, sigma_Z, mu_X, sigma_X, C_XX, C_ZZ, C_contrast, V_H]
    """
    k = cg.k
    edges = get_edge_list(cg)
    h = get_node_weights(cg, w)

    # Z-basis expectations
    z_vals, zz_matrix = counts_to_expectations(z_counts, k)

    # X-basis expectations
    x_vals, xx_matrix = counts_to_expectations(x_counts, k)

    # Pooled statistics
    mu_Z = float(np.mean(z_vals))
    sigma_Z = float(np.std(z_vals))
    mu_X = float(np.mean(x_vals))
    sigma_X = float(np.std(x_vals))

    # Weighted correlations on edges
    edge_weights = np.array([j for _, _, j in edges], dtype=float)
    if len(edge_weights) > 0 and edge_weights.sum() > 1e-10:
        w_norm = np.abs(edge_weights) / (np.abs(edge_weights).sum() + 1e-10)
        c_xx_vals = [xx_matrix[a, b] for a, b, _ in edges]
        c_zz_vals = [zz_matrix[a, b] for a, b, _ in edges]
        C_XX = float(np.sum(w_norm * np.array(c_xx_vals))) if len(c_xx_vals) > 0 else 0.0
        C_ZZ = float(np.sum(w_norm * np.array(c_zz_vals))) if len(c_zz_vals) > 0 else 0.0
    else:
        C_XX = 0.0
        C_ZZ = 0.0

    # Contrast: edges vs non-edges (ZZ)
    if k > 2 and len(edges) > 0:
        edge_set = set()
        for a, b, _ in edges:
            edge_set.add((min(a, b), max(a, b)))
        nonedge_zz = []
        for a in range(k):
            for b in range(a + 1, k):
                if (a, b) not in edge_set:
                    nonedge_zz.append(zz_matrix[a, b])
        if len(nonedge_zz) > 0:
            edge_zz_mean = float(np.mean([zz_matrix[a, b] for a, b, _ in edges]))
            nonedge_zz_mean = float(np.mean(nonedge_zz))
            C_contrast = edge_zz_mean - nonedge_zz_mean
        else:
            C_contrast = 0.0
    else:
        C_contrast = 0.0

    # Quantum variance V_H
    H_mean = 0.0
    if len(edge_weights) > 0:
        xx_edge_vals = np.array([xx_matrix[a, b] for a, b, _ in edges])
        H_mean += float(np.sum(edge_weights * xx_edge_vals))
    H_mean += float(np.sum(h * z_vals))

    var_terms = []
    if len(edge_weights) > 0:
        xx_edge_vals = np.array([xx_matrix[a, b] for a, b, _ in edges])
        if len(xx_edge_vals) > 0:
            var_terms.extend(edge_weights * (xx_edge_vals - np.mean(xx_edge_vals)) ** 2)
    if len(h) > 0:
        var_terms.extend(h * (z_vals - np.mean(z_vals)) ** 2)
    V_H = float(np.sum(var_terms)) if len(var_terms) > 0 else 0.0

    features = np.array([
        mu_Z, sigma_Z, mu_X, sigma_X, C_XX, C_ZZ, C_contrast, V_H,
    ], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Simulator baseline (exact statevector)
# ---------------------------------------------------------------------------

def extract_simulator_features(cg, probe_params, control="Q1", w=None,
                                seed=42, entanglement_strength=1.0):
    """Extract features using the exact statevector simulator (EstimatorV2)."""
    circuit = build_quantum_control_circuit(
        cg, probe_params, control=control, w=w, seed=seed,
        entanglement_strength=entanglement_strength,
    )
    return extract_observables(cg, probe_params, w, circuit=circuit)


# ---------------------------------------------------------------------------
# Hardware job submission
# ---------------------------------------------------------------------------

def run_on_hardware(backend, circuits, shots=200, batch_delay=15.0):
    """Submit circuits to PIAST-Q and collect results.

    Respects the ~250 jobs/hour limit by adding a delay between batches.

    Parameters
    ----------
    backend : PIAST-Q backend object
    circuits : list of QuantumCircuit
        Pre-transpiled circuits with measurements.
    shots : int
        Shots per circuit (max 200).
    batch_delay : float
        Seconds to wait between individual job submissions (14.4s = 250/hour).

    Returns
    -------
    list of dict
        Counts for each circuit.
    """
    results = []
    n = len(circuits)
    for i, qc in enumerate(circuits):
        print(f"  Job {i+1}/{n}...", end=" ", flush=True)
        try:
            job = backend.run(qc, shots=shots)
            counts = job.result().get_counts()
            results.append(counts)
            print(f"OK ({sum(counts.values())} shots)")
        except Exception as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            results.append({})
        if i < n - 1:
            time.sleep(batch_delay)
    return results


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def run_hw_validation(
    n_graphs: int = 100,
    k: int = 8,
    probe_name: str = "P2",
    conditions: list[str] = None,
    shots: int = 200,
    seed: int = 42,
    dry_run: bool = False,
    outdir: str = "hw_results",
    solver_budget: int = 50,
    batch_delay: float = 14.4,
):
    """Run the hardware validation experiment.

    Parameters
    ----------
    n_graphs : int
        Number of graphs to validate on (keep small for hardware budget).
    k : int
        Compression dimension (qubits).
    probe_name : str
        Probe name (P1, P2, P3, P4).
    conditions : list[str]
        Quantum conditions to test: Q0, Q1, Q2, Q3, Q4.
    shots : int
        Shots per circuit (max 200 for PIAST-Q).
    seed : int
        Base random seed.
    dry_run : bool
        If True, only run simulator (no hardware submission).
    outdir : str
        Output directory for results.
    solver_budget : int
        Solver budget per graph.
    batch_delay : float
        Delay between hardware jobs in seconds (14.4s = 250/hour).
    """
    if conditions is None:
        conditions = ["Q1", "Q0"]

    os.makedirs(outdir, exist_ok=True)
    probe_params = get_probe(probe_name)

    print("=" * 70)
    print("Stage 1B: Hardware Validation on PIAST-Q")
    print(f"  Graphs: {n_graphs} | k={k} | Probe={probe_name}")
    print(f"  Conditions: {conditions}")
    print(f"  Shots: {shots} | Dry-run: {dry_run}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate graphs + solve + compress
    # ------------------------------------------------------------------
    print("\n[1] Generating graphs...")
    t0 = time.time()
    splits = make_splits(n_train=n_graphs, n_ood_size=0, n_ood_family=0, seed=seed)
    all_graphs = splits["train"]
    train_graphs, val_graphs = train_val_split(all_graphs, val_frac=0.15, seed=seed)
    all_graphs = train_graphs + val_graphs
    print(f"  Generated {len(all_graphs)} graphs ({time.time()-t0:.1f}s)")

    print("\n[2] Running solver portfolio...")
    t0 = time.time()
    solver_results = []
    for i, G in enumerate(all_graphs):
        sr = run_portfolio(G, budget=solver_budget, seed=seed + i)
        solver_results.append(sr)
    print(f"  Solved {len(all_graphs)} graphs ({time.time()-t0:.1f}s)")

    # Dominance check
    win_fracs = check_solver_dominance(solver_results)
    print(f"  Win fractions: {win_fracs}")
    gate_pass = win_fracs.get("pass", True) if isinstance(win_fracs, dict) else True
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")

    print("\n[3] Computing classical descriptors...")
    t0 = time.time()
    X_classical = compute_descriptors_batch(all_graphs)
    print(f"  X_classical: {X_classical.shape} ({time.time()-t0:.1f}s)")

    print(f"\n[4] Compressing graphs to k={k}...")
    t0 = time.time()
    compressor = SpectralCompressor()
    compressed = []
    for i, G in enumerate(all_graphs):
        cg = compressor.compress(G, k)
        compressed.append(cg)
    print(f"  Compressed {len(compressed)} graphs ({time.time()-t0:.1f}s)")

    # Build target Y (normalized solver performance)
    Y = np.zeros((len(all_graphs), 5), dtype=np.float32)
    for i, sr in enumerate(solver_results):
        Y[i, :] = sr.normalized
    Y_norm = Y
    print(f"  Y: {Y_norm.shape}")

    # ------------------------------------------------------------------
    # Step 2: Simulator features (exact statevector baseline)
    # ------------------------------------------------------------------
    print("\n[5] Extracting SIMULATOR features (exact statevector)...")
    sim_features = {}
    for cond in conditions:
        t0 = time.time()
        feats = []
        for cg in compressed:
            f = extract_simulator_features(cg, probe_params, control=cond, seed=seed)
            feats.append(f)
        sim_features[cond] = np.stack(feats)
        print(f"  {cond}: {sim_features[cond].shape} ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Step 3: Hardware features (if not dry-run)
    # ------------------------------------------------------------------
    hw_features = {}
    if not dry_run:
        if not PIASTQ_AVAILABLE:
            print("\n[ERROR] PiastQ not available. Run from piast_q/ directory.")
            print("  Falling back to dry-run (simulator only).")
        else:
            print("\n[6] Connecting to PIAST-Q...")
            try:
                login(load_token())
                backend = get_backend(direct=True)
                print(f"  Backend: {backend}")
                print(f"  Num qubits: {getattr(backend, 'num_qubits', '?')}")
            except Exception as exc:
                print(f"  [FAIL] Connection failed: {exc}")
                print("  Falling back to dry-run (simulator only).")
                backend = None

            if backend is not None:
                for cond in conditions:
                    print(f"\n[7] Building {cond} circuits for hardware...")
                    circuits_z = []
                    circuits_x = []
                    for cg in compressed:
                        qc_z = build_measured_circuit(
                            cg, probe_params, control=cond, basis="Z", seed=seed)
                        qc_x = build_measured_circuit(
                            cg, probe_params, control=cond, basis="X", seed=seed)
                        circuits_z.append(qc_z)
                        circuits_x.append(qc_x)

                    # Transpile to backend
                    print(f"  Transpiling {len(circuits_z)} Z-circuits...")
                    tqc_z = [transpile(qc, backend) for qc in circuits_z]
                    print(f"  Transpiling {len(circuits_x)} X-circuits...")
                    tqc_x = [transpile(qc, backend) for qc in circuits_x]

                    total_jobs = len(tqc_z) + len(tqc_x)
                    est_time_min = total_jobs * batch_delay / 60
                    print(f"  Total jobs: {total_jobs} (est. {est_time_min:.1f} min)")

                    # Run Z-basis
                    print(f"\n  Running Z-basis circuits for {cond}...")
                    z_results = run_on_hardware(backend, tqc_z, shots=shots,
                                                batch_delay=batch_delay)

                    # Run X-basis
                    print(f"\n  Running X-basis circuits for {cond}...")
                    x_results = run_on_hardware(backend, tqc_x, shots=shots,
                                                batch_delay=batch_delay)

                    # Extract features from counts
                    print(f"\n  Extracting {cond} features from hardware counts...")
                    feats = []
                    for i, cg in enumerate(compressed):
                        f = extract_features_from_counts(
                            z_results[i], x_results[i], cg, probe_params)
                        feats.append(f)
                    hw_features[cond] = np.stack(feats)
                    print(f"  {cond} (hardware): {hw_features[cond].shape}")

                    # Save raw counts
                    counts_file = os.path.join(outdir, f"hw_counts_{cond}.json")
                    with open(counts_file, "w") as f:
                        json.dump({
                            "z": [dict(c) for c in z_results],
                            "x": [dict(c) for c in x_results],
                        }, f, default=str)
                    print(f"  Saved counts to {counts_file}")

    # ------------------------------------------------------------------
    # Step 4: Comparison and conditional test
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Simulator conditional test
    print("\n[8] Simulator conditional test (5-fold CV × 10 seeds)...")
    sim_results = {}
    for cond, feats in sim_features.items():
        result = cv_conditional_test(
            X_classical, feats, Y_norm,
            n_seeds=10, n_folds=5,
        )
        sim_results[cond] = result
        dr2 = result["delta_r2"]
        print(f"  {cond}: ΔR² = {dr2['mean']:.4f} ± {dr2['std']:.4f} "
              f"[{dr2['ci_lo']:.4f}, {dr2['ci_hi']:.4f}]")

    # Hardware conditional test (if available)
    hw_results = {}
    if hw_features:
        print("\n[9] Hardware conditional test (5-fold CV × 10 seeds)...")
        for cond, feats in hw_features.items():
            result = cv_conditional_test(
                X_classical, feats, Y_norm,
                n_seeds=10, n_folds=5,
            )
            hw_results[cond] = result
            dr2 = result["delta_r2"]
            print(f"  {cond} (HW): ΔR² = {dr2['mean']:.4f} ± {dr2['std']:.4f} "
                  f"[{dr2['ci_lo']:.4f}, {dr2['ci_hi']:.4f}]")

    # Feature correlation: simulator vs hardware
    print("\n[10] Simulator vs Hardware feature comparison...")
    correlations = {}
    for cond in hw_features:
        sim_f = sim_features[cond]
        hw_f = hw_features[cond]
        per_feature_corr = []
        for j in range(sim_f.shape[1]):
            if np.std(sim_f[:, j]) > 1e-10 and np.std(hw_f[:, j]) > 1e-10:
                r = float(np.corrcoef(sim_f[:, j], hw_f[:, j])[0, 1])
            else:
                r = 0.0
            per_feature_corr.append(r)
        correlations[cond] = per_feature_corr
        print(f"  {cond} per-feature correlations: {[f'{r:.3f}' for r in per_feature_corr]}")
        mean_r = float(np.nanmean(per_feature_corr))
        print(f"  {cond} mean correlation: {mean_r:.3f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    print("\n[11] Saving results...")

    # Save feature matrices
    for cond, feats in sim_features.items():
        np.save(os.path.join(outdir, f"sim_{cond}.npy"), feats)
    for cond, feats in hw_features.items():
        np.save(os.path.join(outdir, f"hw_{cond}.npy"), feats)
    np.save(os.path.join(outdir, "X_classical.npy"), X_classical)
    np.save(os.path.join(outdir, "Y_normalized.npy"), Y_norm)

    # Save summary JSON
    summary = {
        "config": {
            "n_graphs": n_graphs,
            "k": k,
            "probe": probe_name,
            "conditions": conditions,
            "shots": shots,
            "seed": seed,
            "dry_run": dry_run,
            "solver_budget": solver_budget,
        },
        "simulator_results": sim_results,
        "hardware_results": hw_results,
        "feature_correlations": correlations,
    }
    summary_file = os.path.join(outdir, "hw_validation_results.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved to {summary_file}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HARDWARE VALIDATION REPORT")
    print("=" * 70)
    print(f"\nGraphs: {n_graphs} | k={k} | Probe={probe_name} | Shots={shots}")
    print(f"Conditions: {conditions}")
    print()

    print("Simulator Results:")
    for cond, res in sim_results.items():
        dr2 = res["delta_r2"]
        print(f"  {cond}: ΔR² = {dr2['mean']:.4f} "
              f"[{dr2['ci_lo']:.4f}, {dr2['ci_hi']:.4f}]")

    if hw_results:
        print("\nHardware Results:")
        for cond, res in hw_results.items():
            dr2 = res["delta_r2"]
            print(f"  {cond}: ΔR² = {dr2['mean']:.4f} "
                  f"[{dr2['ci_lo']:.4f}, {dr2['ci_hi']:.4f}]")

        print("\nSimulator vs Hardware Correlation:")
        for cond, corrs in correlations.items():
            mean_r = float(np.nanmean(corrs))
            print(f"  {cond}: mean r = {mean_r:.3f}")

        # Verdict
        print("\nVerdict:")
        for cond in hw_results:
            sim_dr2 = sim_results[cond]["delta_r2"]["mean"]
            hw_dr2 = hw_results[cond]["delta_r2"]["mean"]
            hw_ci_low = hw_results[cond]["delta_r2"]["ci_lo"]
            if hw_dr2 > 0 and hw_ci_low > 0:
                print(f"  {cond}: PASS (HW ΔR² = {hw_dr2:.4f}, CI > 0)")
            elif hw_dr2 > 0:
                print(f"  {cond}: MARGINAL (HW ΔR² = {hw_dr2:.4f}, CI includes 0)")
            else:
                print(f"  {cond}: FAIL (HW ΔR² = {hw_dr2:.4f})")
    else:
        print("\n(No hardware results — dry run)")

    print("\n" + "=" * 70)
    print("Done. Results in", outdir)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1B: Hardware validation on PIAST-Q")
    parser.add_argument("--graphs", type=int, default=100,
                        help="Number of graphs (default 100)")
    parser.add_argument("--k", type=int, default=8,
                        help="Compression dimension / qubits (default 8)")
    parser.add_argument("--probe", type=str, default="P2",
                        help="Probe name (default P2)")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["Q1", "Q0"],
                        help="Quantum conditions to test")
    parser.add_argument("--shots", type=int, default=200,
                        help="Shots per circuit (max 200 for PIAST-Q)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulator only, no hardware submission")
    parser.add_argument("--outdir", type=str, default="hw_results",
                        help="Output directory")
    parser.add_argument("--solver-budget", type=int, default=50,
                        help="Solver budget per graph")
    parser.add_argument("--batch-delay", type=float, default=14.4,
                        help="Delay between hardware jobs in seconds "
                             "(14.4s = 250 jobs/hour)")
    args = parser.parse_args()

    if args.shots > SHOTS_MAX:
        print(f"WARNING: shots={args.shots} exceeds PIAST-Q limit ({SHOTS_MAX}). "
              f"Capping to {SHOTS_MAX}.")
        args.shots = SHOTS_MAX

    run_hw_validation(
        n_graphs=args.graphs,
        k=args.k,
        probe_name=args.probe,
        conditions=args.conditions,
        shots=args.shots,
        seed=args.seed,
        dry_run=args.dry_run,
        outdir=args.outdir,
        solver_budget=args.solver_budget,
        batch_delay=args.batch_delay,
    )


if __name__ == "__main__":
    main()
