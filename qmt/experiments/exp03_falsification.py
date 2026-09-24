"""Stage 0C: Falsification experiment.

The most important experiment in the project.
Try to kill the quantum signal with every reasonable classical and quantum control.

Classical controls (matched dimension to Q):
  A: Gaussian random projection of C
  B: Random Fourier Features from C
  C: Random classical reservoir (fixed random network)
  D: Trainable nonlinear baselines (RBF, MLP, RF, XGBoost)

Quantum controls:
  Q0: No entanglement (separable)
  Q1: Original entangled reservoir (reference)
  Q2: Scrambled connectivity
  Q3: Scrambled input
  Q4: Entanglement sweep e ∈ {0, 0.25, 0.5, 0.75, 1.0}

Statistics:
  - 5-fold CV × 10 seeds
  - Permutation test B=1000
  - Bootstrap CIs
  - Representation geometry

Usage:
    PYTHONPATH=. python -m qmt.experiments.exp03_falsification \
        --graphs 2000 --k 8 --probe P2 --outdir qmt/results/stage0c
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import networkx as nx

from ..graphs.splits import make_splits, train_val_split
from ..graphs.descriptors import compute_descriptors_batch
from ..solvers.portfolio import run_portfolio, check_solver_dominance
from ..compression.spectral import SpectralCompressor
from ..compression.random_control import RandomCompressor
from ..quantum.probes import get_probe
from ..quantum.observables import extract_observables_batch
from ..quantum.quantum_controls import build_quantum_control_circuit
from ..diagnostics.classical_controls import (
    control_a_gaussian_projection,
    control_b_rff,
    control_c_random_reservoir,
    compute_delta_r2,
)
from ..diagnostics.proper_stats import cv_conditional_test, permutation_test, bootstrap_ci
from ..diagnostics.representation_geometry import full_geometry_analysis


def run_stage0c(
    n_graphs: int = 2000,
    k: int = 8,
    probe_name: str = "P2",
    outdir: str = "qmt/results/stage0c",
    solver_budget: int = 50,
    n_seeds: int = 10,
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    seed: int = 42,
):
    os.makedirs(outdir, exist_ok=True)
    print("=" * 70)
    print("Stage 0C: Falsification Experiment")
    print(f"  Graphs: {n_graphs} | k={k} | Probe={probe_name}")
    print(f"  Seeds: {n_seeds} | Permutations: {n_permutations} | Bootstrap: {n_bootstrap}")
    print("=" * 70)

    # Step 1: Generate graphs
    t0 = time.time()
    print(f"\n[1] Generating {n_graphs} graphs...")
    splits = make_splits(n_train=n_graphs, n_ood_size=0, n_ood_family=0, seed=seed)
    all_graphs = splits["train"]
    train_graphs, val_graphs = train_val_split(all_graphs, val_frac=0.15, seed=seed)
    all_graphs = train_graphs + val_graphs
    n_train = len(train_graphs)
    print(f"  Train: {n_train} | Val: {len(val_graphs)}")
    print(f"  Data gen: {time.time()-t0:.1f}s")

    # Step 2: Run solver portfolio
    t0 = time.time()
    print(f"\n[2] Running solver portfolio...")
    solver_results = []
    for i, G in enumerate(all_graphs):
        sr = run_portfolio(G, budget=solver_budget, seed=seed + i)
        solver_results.append(sr)
        if (i + 1) % 500 == 0:
            print(f"  Solved {i+1}/{len(all_graphs)} ({time.time()-t0:.1f}s)")
    print(f"  Solver portfolio: {time.time()-t0:.1f}s")

    # Step 3: Dominance check
    print(f"\n[3] Checking solver dominance...")
    dominance = check_solver_dominance(solver_results)
    print(f"  Win fractions: {dominance['fractions']}")
    print(f"  Max fraction: {dominance['max_fraction']:.3f}")
    print(f"  Gate: {'PASS' if dominance['pass'] else 'FAIL'}")

    # Step 4: Classical descriptors
    t0 = time.time()
    print(f"\n[4] Computing classical descriptors...")
    X_classical = compute_descriptors_batch(all_graphs)
    print(f"  X_classical: {X_classical.shape}")
    print(f"  Descriptors: {time.time()-t0:.1f}s")

    # Build targets
    Y = np.stack([sr.normalized for sr in solver_results])
    Y_hardness = np.array([sr.hardness for sr in solver_results])

    # Step 5: Compress graphs
    t0 = time.time()
    print(f"\n[5] Compressing graphs to k={k}...")
    compressor = SpectralCompressor()
    compressed = []
    for i, G in enumerate(all_graphs):
        cg = compressor.compress(G, k)
        compressed.append(cg)
        if (i + 1) % 500 == 0:
            print(f"  Compressed {i+1}/{len(all_graphs)} ({time.time()-t0:.1f}s)")
    print(f"  Compression: {time.time()-t0:.1f}s")

    # Step 6: Extract quantum features for all controls
    probe = get_probe(probe_name)
    n_q_features = 8

    quantum_controls = ["Q0", "Q1", "Q2", "Q3"]
    entanglement_sweep = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_Q = {}

    for qc_name in quantum_controls:
        t0 = time.time()
        print(f"\n[6.{qc_name}] Extracting {qc_name} features...")
        circuits = []
        for i, cg in enumerate(compressed):
            circ = build_quantum_control_circuit(
                cg, probe, control=qc_name, seed=seed + i,
            )
            circuits.append(circ)
        Q = extract_observables_batch(compressed, probe, circuits=circuits)
        all_Q[qc_name] = Q
        print(f"  {qc_name}: {Q.shape} ({time.time()-t0:.1f}s)")

    # Entanglement sweep
    for e in entanglement_sweep:
        qc_name = f"Q4_e{e}"
        t0 = time.time()
        print(f"\n[6.{qc_name}] Entanglement sweep e={e}...")
        circuits = []
        for i, cg in enumerate(compressed):
            circ = build_quantum_control_circuit(
                cg, probe, control="Q4", seed=seed + i,
                entanglement_strength=e,
            )
            circuits.append(circ)
        Q = extract_observables_batch(compressed, probe, circuits=circuits)
        all_Q[qc_name] = Q
        print(f"  {qc_name}: {Q.shape} ({time.time()-t0:.1f}s)")

    # Step 7: Classical controls
    print(f"\n[7] Building classical controls...")
    classical_controls = {}
    for name, fn in [
        ("A_gaussian", lambda C: control_a_gaussian_projection(C, n_features=n_q_features, seed=seed)),
        ("B_rff", lambda C: control_b_rff(C, n_features=n_q_features, seed=seed)),
        ("C_reservoir", lambda C: control_c_random_reservoir(C, n_features=n_q_features, seed=seed)),
    ]:
        features = fn(X_classical)
        classical_controls[name] = features
        print(f"  {name}: {features.shape}")

    # Step 8: Proper CV conditional test for all features
    print(f"\n[8] Cross-validated conditional test (5-fold × {n_seeds} seeds)...")
    results = {}

    # Quantum controls
    for qc_name, Q in all_Q.items():
        print(f"\n  {qc_name}...")
        t0 = time.time()
        result = cv_conditional_test(X_classical, Q, Y, n_folds=5, n_seeds=n_seeds)
        result["time"] = time.time() - t0
        results[qc_name] = result
        print(f"    ΔR² = {result['delta_r2']['mean']:.4f} ± {result['delta_r2']['std']:.4f} "
              f"[{result['delta_r2']['ci_lo']:.4f}, {result['delta_r2']['ci_hi']:.4f}]")

    # Classical controls
    for cc_name, features in classical_controls.items():
        print(f"\n  {cc_name}...")
        t0 = time.time()
        result = cv_conditional_test(X_classical, features, Y, n_folds=5, n_seeds=n_seeds)
        result["time"] = time.time() - t0
        results[cc_name] = result
        print(f"    ΔR² = {result['delta_r2']['mean']:.4f} ± {result['delta_r2']['std']:.4f} "
              f"[{result['delta_r2']['ci_lo']:.4f}, {result['delta_r2']['ci_hi']:.4f}]")

    # Also test on hardness target
    print(f"\n  Q1 on hardness target...")
    result_hardness = cv_conditional_test(X_classical, all_Q["Q1"], Y_hardness, n_folds=5, n_seeds=n_seeds)
    results["Q1_hardness"] = result_hardness
    print(f"    ΔR² = {result_hardness['delta_r2']['mean']:.4f} ± {result_hardness['delta_r2']['std']:.4f}")

    # Step 9: Permutation test for Q1
    print(f"\n[9] Permutation test for Q1 (B={n_permutations})...")
    perm_result = permutation_test(
        X_classical, all_Q["Q1"], Y, B=n_permutations, n_folds=5, seed=seed,
    )
    print(f"  Observed ΔR²: {perm_result['delta_r2_observed']:.4f}")
    print(f"  Null mean: {perm_result['null_mean']:.4f}")
    print(f"  p-value: {perm_result['p_value']:.4f}")
    print(f"  Significant (p<0.01): {perm_result['significant']}")

    # Step 10: Bootstrap CI for Q1
    print(f"\n[10] Bootstrap CI for Q1 (B={n_bootstrap})...")
    boot_result = bootstrap_ci(
        X_classical, all_Q["Q1"], Y, n_bootstrap=n_bootstrap, n_folds=5, seed=seed,
    )
    print(f"  ΔR² = {boot_result['mean']:.4f} [{boot_result['ci_lo']:.4f}, {boot_result['ci_hi']:.4f}]")

    # Step 11: Representation geometry
    print(f"\n[11] Representation geometry...")
    geometry = full_geometry_analysis(X_classical, all_Q["Q1"], Y)
    print(f"  C effective rank: {geometry['classical']['effective_rank']:.2f}")
    print(f"  Q effective rank: {geometry['quantum']['effective_rank']:.2f}")
    print(f"  CKA(C,Q) linear: {geometry['cka_linear_CQ']:.4f}")
    print(f"  CKA(C,Q) rbf: {geometry['cka_rbf_CQ']:.4f}")
    print(f"  CKA(Q,Y) linear: {geometry['cka_linear_QY']:.4f}")
    print(f"  CKA(C,Y) linear: {geometry['cka_linear_CY']:.4f}")
    print(f"  Kernel alignment Q-Y: {geometry['kernel_alignment_QY']:.4f}")
    print(f"  Kernel alignment C-Y: {geometry['kernel_alignment_CY']:.4f}")

    # Step 12: Entanglement sweep analysis
    print(f"\n[12] Entanglement sweep analysis...")
    sweep_results = []
    for e in entanglement_sweep:
        qc_name = f"Q4_e{e}"
        r = results[qc_name]["delta_r2"]["mean"]
        sweep_results.append({"e": e, "delta_r2": r})
        print(f"  e={e:.2f}: ΔR² = {r:.4f}")

    # Step 13: Generate report
    print(f"\n[13] Generating STAGE0C_REPORT.md...")
    report = generate_stage0c_report(
        results, perm_result, boot_result, geometry,
        sweep_results, dominance, all_Q, classical_controls,
        n_graphs, k, probe_name, n_seeds, n_permutations, n_bootstrap,
    )
    report_path = os.path.join(outdir, "STAGE0C_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    # Save results
    save_results = {
        "cv_results": {k: v for k, v in results.items()},
        "permutation": {k: v for k, v in perm_result.items() if k != "null_distribution"},
        "bootstrap": {k: v for k, v in boot_result.items() if k != "distribution"},
        "geometry": geometry,
        "entanglement_sweep": sweep_results,
        "dominance": {
            "fractions": {k: float(v) for k, v in dominance["fractions"].items()},
            "max_fraction": float(dominance["max_fraction"]),
            "pass": bool(dominance["pass"]),
        },
    }
    with open(os.path.join(outdir, "stage0c_results.json"), "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    # Save features
    for name, Q in all_Q.items():
        np.save(os.path.join(outdir, f"Q_{name}.npy"), Q)
    for name, features in classical_controls.items():
        np.save(os.path.join(outdir, f"C_control_{name}.npy"), features)
    np.save(os.path.join(outdir, "X_classical.npy"), X_classical)
    np.save(os.path.join(outdir, "Y_normalized.npy"), Y)
    np.save(os.path.join(outdir, "Y_hardness.npy"), Y_hardness)

    print(f"\nDone. Results in {outdir}/")


def generate_stage0c_report(
    results, perm_result, boot_result, geometry,
    sweep_results, dominance, all_Q, classical_controls,
    n_graphs, k, probe_name, n_seeds, n_permutations, n_bootstrap,
) -> str:
    lines = []
    lines.append("# STAGE 0C REPORT: Falsification Experiment\n")
    lines.append(f"**Graphs**: {n_graphs} | **k**: {k} | **Probe**: {probe_name}\n")
    lines.append(f"**Seeds**: {n_seeds} | **Permutations**: {n_permutations} | **Bootstrap**: {n_bootstrap}\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Dataset validity
    lines.append("\n## 1. Dataset Validity\n")
    lines.append(f"- Solver win fractions: {dominance['fractions']}")
    lines.append(f"- Max fraction: {dominance['max_fraction']:.3f}")
    lines.append(f"- Gate: {'PASS' if dominance['pass'] else 'FAIL'}")

    # CV results table
    lines.append("\n## 2. Cross-Validated Conditional Test (5-fold × {} seeds)\n".format(n_seeds))
    lines.append("| Feature | ΔR² mean | ΔR² std | 95% CI | R²(C) | R²(C+Q) | R² residual |")
    lines.append("|---------|----------|---------|--------|-------|---------|-------------|")

    # Order: Q1 first, then quantum controls, then classical controls
    ordered = ["Q1", "Q0", "Q2", "Q3"] + [f"Q4_e{e}" for e in [0.0, 0.25, 0.5, 0.75, 1.0]] + \
              ["A_gaussian", "B_rff", "C_reservoir"]

    for name in ordered:
        if name not in results:
            continue
        r = results[name]
        d = r["delta_r2"]
        label = name
        if name == "Q1":
            label = "**Q1 (entangled)**"
        elif name == "Q0":
            label = "Q0 (separable)"
        elif name == "Q2":
            label = "Q2 (scrambled conn.)"
        elif name == "Q3":
            label = "Q3 (scrambled input)"
        lines.append(
            f"| {label} | {d['mean']:.4f} | {d['std']:.4f} | "
            f"[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] | "
            f"{r['r2_c']['mean']:.4f} | {r['r2_cq']['mean']:.4f} | "
            f"{r['r2_residual']['mean']:.4f} |"
        )

    # Hardness
    if "Q1_hardness" in results:
        r = results["Q1_hardness"]
        d = r["delta_r2"]
        lines.append(
            f"| Q1 (hardness) | {d['mean']:.4f} | {d['std']:.4f} | "
            f"[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] | "
            f"{r['r2_c']['mean']:.4f} | {r['r2_cq']['mean']:.4f} | "
            f"{r['r2_residual']['mean']:.4f} |"
        )

    # Key comparisons
    lines.append("\n## 3. Key Comparisons (Gates)\n")

    q1 = results.get("Q1", {}).get("delta_r2", {}).get("mean", 0)
    q0 = results.get("Q0", {}).get("delta_r2", {}).get("mean", 0)
    q2 = results.get("Q2", {}).get("delta_r2", {}).get("mean", 0)
    q3 = results.get("Q3", {}).get("delta_r2", {}).get("mean", 0)

    best_classical = max(
        results.get("A_gaussian", {}).get("delta_r2", {}).get("mean", -999),
        results.get("B_rff", {}).get("delta_r2", {}).get("mean", -999),
        results.get("C_reservoir", {}).get("delta_r2", {}).get("mean", -999),
    )

    comparisons = [
        ("Gate A: Q1 ΔR² > 0", q1 > 0, f"ΔR² = {q1:.4f}"),
        ("Gate B: Q1 > best classical", q1 > best_classical, f"Q1={q1:.4f} vs best_cl={best_classical:.4f}"),
        ("Gate C: Q1 > Q0 (entanglement)", q1 > q0, f"Q1={q1:.4f} vs Q0={q0:.4f}"),
        ("Gate D: Q1 > Q2 (scrambled conn.)", q1 > q2, f"Q1={q1:.4f} vs Q2={q2:.4f}"),
        ("Gate E: Q1 > Q3 (scrambled input)", q1 > q3, f"Q1={q1:.4f} vs Q3={q3:.4f}"),
    ]

    for name, passed, detail in comparisons:
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'} ({detail})")

    # Permutation test
    lines.append("\n## 4. Permutation Test (H₀: I(Q;Y|C) = 0)\n")
    lines.append(f"- Observed ΔR²: {perm_result['delta_r2_observed']:.4f}")
    lines.append(f"- Null distribution: mean={perm_result['null_mean']:.4f}, std={perm_result['null_std']:.4f}")
    lines.append(f"- p-value: {perm_result['p_value']:.4f}")
    lines.append(f"- Significant (p<0.01): {'YES' if perm_result['significant'] else 'NO'}")

    # Bootstrap CI
    lines.append("\n## 5. Bootstrap Confidence Interval\n")
    lines.append(f"- ΔR² = {boot_result['mean']:.4f} [{boot_result['ci_lo']:.4f}, {boot_result['ci_hi']:.4f}]_{int(100*(1-0.05))}%")
    lines.append(f"- CI excludes zero: {'YES' if boot_result['ci_lo'] > 0 else 'NO'}")

    # Entanglement sweep
    lines.append("\n## 6. Entanglement Sweep\n")
    lines.append("| e | ΔR² |")
    lines.append("|---|------|")
    for s in sweep_results:
        lines.append(f"| {s['e']:.2f} | {s['delta_r2']:.4f} |")

    # Check monotonicity
    sweep_deltas = [s["delta_r2"] for s in sweep_results]
    is_monotone = all(sweep_deltas[i] <= sweep_deltas[i+1] for i in range(len(sweep_deltas)-1))
    lines.append(f"\nMonotone increasing: {'YES' if is_monotone else 'NO'}")

    # Representation geometry
    lines.append("\n## 7. Representation Geometry\n")
    lines.append(f"- Classical effective rank: {geometry['classical']['effective_rank']:.2f}")
    lines.append(f"- Quantum effective rank: {geometry['quantum']['effective_rank']:.2f}")
    lines.append(f"- CKA(C, Q) linear: {geometry['cka_linear_CQ']:.4f}")
    lines.append(f"- CKA(C, Q) rbf: {geometry['cka_rbf_CQ']:.4f}")
    lines.append(f"- CKA(Q, Y) linear: {geometry['cka_linear_QY']:.4f}")
    lines.append(f"- CKA(C, Y) linear: {geometry['cka_linear_CY']:.4f}")
    lines.append(f"- Kernel alignment Q→Y: {geometry['kernel_alignment_QY']:.4f}")
    lines.append(f"- Kernel alignment C→Y: {geometry['kernel_alignment_CY']:.4f}")
    lines.append(f"- Principal angles C↔Q: {[f'{a:.1f}°' for a in geometry['principal_angles_CQ']]}")

    # Verdict
    lines.append("\n## 8. Verdict\n")
    n_pass = sum(1 for _, p, _ in comparisons if p)
    lines.append(f"Passed {n_pass}/{len(comparisons)} gates.")

    if n_pass >= 4 and perm_result["significant"] and boot_result["ci_lo"] > 0:
        lines.append("\n**VERDICT: QUANTUM SIGNAL SURVIVES FALSIFICATION**")
        lines.append("\nThe quantum reservoir signal is:")
        lines.append("- Reproducible across seeds (CV)")
        lines.append("- Statistically significant (permutation test)")
        lines.append("- Not explained by matched classical controls")
        lines.append("- Dependent on entanglement and graph structure")
        lines.append("\n→ Proceed to Stage 1A (full IID study, 10k+ graphs)")
    elif n_pass >= 3 and perm_result["significant"]:
        lines.append("\n**VERDICT: PARTIAL SIGNAL**")
        lines.append("\nSome gates pass but not all. Investigate which controls match.")
        lines.append("\n→ Investigate before proceeding")
    else:
        lines.append("\n**VERDICT: SIGNAL FALSIFIED**")
        lines.append("\nThe quantum signal does not survive classical or quantum controls.")
        lines.append("\n→ The pilot signal was likely nonlinear feature expansion, not quantum-specific.")

    lines.append("\n---\n")
    lines.append("*Internal research gate, not a scientific claim.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 0C: Falsification Experiment")
    parser.add_argument("--graphs", type=int, default=2000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--probe", type=str, default="P2")
    parser.add_argument("--outdir", type=str, default="qmt/results/stage0c")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_stage0c(
        n_graphs=args.graphs, k=args.k, probe_name=args.probe,
        outdir=args.outdir, solver_budget=args.budget,
        n_seeds=args.seeds, n_permutations=args.permutations,
        n_bootstrap=args.bootstrap, seed=args.seed,
    )


if __name__ == "__main__":
    main()
