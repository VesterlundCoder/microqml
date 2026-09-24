"""Experiment 0A: Global Quantum Micro-Reservoir pilot.

Pipeline:
1. Generate 2,000 MaxCut graphs (training families, N=30-80)
2. Run solver portfolio on all graphs -> y(G) vectors
3. Compute 12 cheap classical descriptors x_G
4. Spectral compression to k=8 and k=12
5. Build CGQP circuits, extract quantum features q(G) using one probe (P2)
6. Run conditional residual test: I(Q;Y|C)
7. Run oracle diagnostic
8. Check dataset validity (solver dominance)
9. Output: PILOT_REPORT.md with go/no-go recommendation

Usage:
    PYTHONPATH=. python -m qmt.experiments.exp01_global_reservoir \
        --graphs 2000 --k 8 12 --probe P2 --outdir qmt/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import networkx as nx

from ..graphs.splits import make_splits, train_val_split
from ..graphs.descriptors import compute_descriptors_batch
from ..solvers.portfolio import run_portfolio, check_solver_dominance, SOLVER_NAMES
from ..compression.spectral import SpectralCompressor
from ..compression.random_control import RandomCompressor
from ..quantum.probes import get_probe
from ..quantum.observables import extract_observables_batch
from ..diagnostics.conditional_info import conditional_residual_test, feature_variance_check
from ..diagnostics.oracle_models import oracle_test


def run_stage0(
    n_graphs: int = 2000,
    k_values: list[int] | None = None,
    probe_name: str = "P2",
    outdir: str = "qmt/results",
    solver_budget: int = 500,
    seed: int = 42,
):
    if k_values is None:
        k_values = [8, 12]

    os.makedirs(outdir, exist_ok=True)
    print(f"=" * 60)
    print(f"Experiment 0A: Global Quantum Micro-Reservoir Pilot")
    print(f"  Graphs: {n_graphs} | k={k_values} | Probe={probe_name}")
    print(f"=" * 60)

    # Step 1: Generate graphs
    t0 = time.time()
    print(f"\n[1] Generating {n_graphs} graphs...")
    splits = make_splits(
        n_train=n_graphs, n_ood_size=0, n_ood_family=0, seed=seed,
    )
    all_graphs = splits["train"]
    train_graphs, val_graphs = train_val_split(all_graphs, val_frac=0.15, seed=seed)
    print(f"  Train: {len(train_graphs)} | Val: {len(val_graphs)}")
    print(f"  Data gen: {time.time()-t0:.1f}s")

    # Step 2: Run solver portfolio
    t0 = time.time()
    print(f"\n[2] Running solver portfolio on all graphs...")
    all_graphs = train_graphs + val_graphs
    n_train = len(train_graphs)
    solver_results = []
    for i, G in enumerate(all_graphs):
        sr = run_portfolio(G, budget=solver_budget, seed=seed + i)
        sr.graph_id = f"g{i}"
        solver_results.append(sr)
        if (i + 1) % 500 == 0:
            print(f"  Solved {i+1}/{len(all_graphs)} graphs ({time.time()-t0:.1f}s)")
    print(f"  Solver portfolio: {time.time()-t0:.1f}s")

    # Step 3: Dataset validity gate
    print(f"\n[3] Checking solver dominance...")
    dominance = check_solver_dominance(solver_results)
    print(f"  Solver win fractions: {dominance['fractions']}")
    print(f"  Max fraction: {dominance['max_fraction']:.3f}")
    print(f"  Gate: {'PASS' if dominance['pass'] else 'FAIL'}")

    # Step 4: Compute classical descriptors
    t0 = time.time()
    print(f"\n[4] Computing classical descriptors (x_G)...")
    X_classical = compute_descriptors_batch(all_graphs)
    print(f"  X_classical shape: {X_classical.shape}")
    print(f"  Descriptors: {time.time()-t0:.1f}s")

    # Build target matrix Y (normalized solver performance vector)
    Y = np.stack([sr.normalized for sr in solver_results])  # (N, 5)
    Y_hardness = np.array([sr.hardness for sr in solver_results])  # (N,)
    Y_best = np.array([sr.best_solver for sr in solver_results])  # (N,)

    # Split into train/val
    Xc_tr = X_classical[:n_train]
    Xc_va = X_classical[n_train:]
    Y_tr = Y[:n_train]
    Y_va = Y[n_train:]

    # Step 5: Compress graphs
    results = {}
    for k in k_values:
        t0 = time.time()
        print(f"\n[5] Compressing graphs to k={k} (spectral)...")
        compressor = SpectralCompressor()
        compressed = []
        for i, G in enumerate(all_graphs):
            cg = compressor.compress(G, k)
            compressed.append(cg)
            if (i + 1) % 500 == 0:
                print(f"  Compressed {i+1}/{len(all_graphs)} ({time.time()-t0:.1f}s)")
        print(f"  Compression: {time.time()-t0:.1f}s")

        # Step 6: Extract quantum features
        t0 = time.time()
        print(f"\n[6] Extracting quantum features (probe={probe_name})...")
        probe = get_probe(probe_name)
        Q = extract_observables_batch(compressed, probe)
        print(f"  Q shape: {Q.shape}")
        print(f"  Quantum extraction: {time.time()-t0:.1f}s")

        # Step 7: Feature variance check
        print(f"\n[7] Checking quantum feature variance...")
        var_check = feature_variance_check(Q)
        print(f"  Effective rank: {var_check['effective_rank']:.2f}")
        print(f"  Constant features: {var_check['n_constant']}/{var_check['n_features']}")

        # Step 8: Conditional residual test
        print(f"\n[8] Conditional residual test: I(Q;Y|C)...")
        cond_result = conditional_residual_test(X_classical, Q, Y)
        print(f"  R^2 classical:     {cond_result['r2_classical']:.4f}")
        print(f"  R^2 residual:      {cond_result['r2_residual']:.4f}")
        print(f"  Delta rho:          {cond_result['delta_rho']:.4f}")
        print(f"  R^2 combined:      {cond_result['r2_combined']:.4f}")
        print(f"  R^2 quantum only:  {cond_result['r2_quantum_only']:.4f}")
        print(f"  Gate: {'PASS' if cond_result['pass_gate'] else 'FAIL'}")

        # Step 9: Oracle test
        print(f"\n[9] Oracle diagnostic test...")
        oracle_result = oracle_test(X_classical, Q, Y, seed=seed)
        print(f"  Oracle R^2 (Q):    {oracle_result['oracle_r2_q']:.4f}")
        print(f"  Oracle R^2 (C):    {oracle_result['oracle_r2_c']:.4f}")
        print(f"  Oracle R^2 (C+Q):   {oracle_result['oracle_r2_cq']:.4f}")
        print(f"  Incremental (oracle): {oracle_result['incremental_oracle']:.4f}")
        print(f"  Incremental (ridge):  {oracle_result['incremental_ridge']:.4f}")

        # Also test on hardness target
        print(f"\n[8b] Conditional residual on hardness...")
        cond_hardness = conditional_residual_test(X_classical, Q, Y_hardness)
        print(f"  R^2 classical:     {cond_hardness['r2_classical']:.4f}")
        print(f"  R^2 residual:      {cond_hardness['r2_residual']:.4f}")
        print(f"  Delta rho:          {cond_hardness['delta_rho']:.4f}")
        print(f"  Gate: {'PASS' if cond_hardness['pass_gate'] else 'FAIL'}")

        results[f"k={k}"] = {
            "k": k,
            "compression": "spectral",
            "probe": probe_name,
            "n_graphs": len(all_graphs),
            "solver_dominance": dominance,
            "feature_variance": var_check,
            "conditional_residual": cond_result,
            "conditional_residual_hardness": cond_hardness,
            "oracle": oracle_result,
            "Q_shape": list(Q.shape),
        }

        # Save quantum features
        np.save(os.path.join(outdir, f"Q_k{k}_{probe_name}.npy"), Q)
        np.save(os.path.join(outdir, "X_classical.npy"), X_classical)
        np.save(os.path.join(outdir, "Y_normalized.npy"), Y)
        np.save(os.path.join(outdir, "Y_hardness.npy"), Y_hardness)
        np.save(os.path.join(outdir, "Y_best_solver.npy"), Y_best)

    # Step 10: Generate PILOT_REPORT.md
    print(f"\n[10] Generating PILOT_REPORT.md...")
    report = generate_pilot_report(results, n_graphs, k_values, probe_name, dominance)
    report_path = os.path.join(outdir, "PILOT_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    # Save raw results
    with open(os.path.join(outdir, "exp0a_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDone. Results in {outdir}/")


def generate_pilot_report(
    results: dict, n_graphs: int, k_values: list[int],
    probe_name: str, dominance: dict,
) -> str:
    """Generate PILOT_REPORT.md with go/no-go recommendation."""
    lines = []
    lines.append("# PILOT REPORT: Experiment 0A — Global Quantum Micro-Reservoir\n")
    lines.append(f"**Graphs**: {n_graphs} | **k values**: {k_values} | **Probe**: {probe_name}\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Solver dominance
    lines.append("\n## 1. Dataset Validity\n")
    lines.append(f"- Solver win fractions: {dominance['fractions']}")
    lines.append(f"- Max fraction: {dominance['max_fraction']:.3f}")
    lines.append(f"- Gate: {'PASS' if dominance['pass'] else 'FAIL'}")

    # Per-k results
    any_pass = False
    for key, r in results.items():
        lines.append(f"\n## Results for {key}\n")
        lines.append(f"- Compression: {r['compression']}")
        lines.append(f"- Q shape: {r['Q_shape']}")
        lines.append(f"- Effective rank: {r['feature_variance']['effective_rank']:.2f}")
        lines.append(f"- Constant features: {r['feature_variance']['n_constant']}")
        lines.append(f"\n### Conditional Residual Test (I(Q;Y|C))")
        cr = r["conditional_residual"]
        lines.append(f"- R^2 classical: {cr['r2_classical']:.4f}")
        lines.append(f"- R^2 residual: {cr['r2_residual']:.4f}")
        lines.append(f"- Delta rho: {cr['delta_rho']:.4f}")
        lines.append(f"- R^2 combined: {cr['r2_combined']:.4f}")
        lines.append(f"- R^2 quantum only: {cr['r2_quantum_only']:.4f}")
        lines.append(f"- Gate: {'PASS' if cr['pass_gate'] else 'FAIL'}")
        lines.append(f"\n### Conditional Residual on Hardness")
        ch = r["conditional_residual_hardness"]
        lines.append(f"- R^2 classical: {ch['r2_classical']:.4f}")
        lines.append(f"- R^2 residual: {ch['r2_residual']:.4f}")
        lines.append(f"- Delta rho: {ch['delta_rho']:.4f}")
        lines.append(f"- Gate: {'PASS' if ch['pass_gate'] else 'FAIL'}")
        lines.append(f"\n### Oracle Diagnostic")
        o = r["oracle"]
        lines.append(f"- Oracle R^2 (Q): {o['oracle_r2_q']:.4f}")
        lines.append(f"- Oracle R^2 (C): {o['oracle_r2_c']:.4f}")
        lines.append(f"- Oracle R^2 (C+Q): {o['oracle_r2_cq']:.4f}")
        lines.append(f"- Incremental (oracle): {o['incremental_oracle']:.4f}")
        lines.append(f"- Incremental (ridge): {o['incremental_ridge']:.4f}")

        if cr["pass_gate"] or ch["pass_gate"]:
            any_pass = True

    # Recommendation
    lines.append("\n## Recommendation\n")
    if any_pass:
        lines.append("**PROMOTE_TO_FULL_STAGE_1 = YES**")
        lines.append("\nAt least one k value shows incremental quantum signal (R^2_residual >= 0.02 or delta_rho >= 0.03).")
        lines.append("Proceed to full Stage 1 with all conditions (M0-M6), multiple compressors, and full parameter sweep.")
    else:
        lines.append("**PROMOTE_TO_FULL_STAGE_1 = NO**")
        lines.append("\nNo k value showed sufficient incremental quantum signal.")
        lines.append("The quantum representation may not contain task-relevant information beyond cheap classical descriptors.")
        lines.append("Consider:")
        lines.append("- Different compression methods (community, hyperbolic)")
        lines.append("- Different probe parameters")
        lines.append("- Different observable sets")
        lines.append("- The possibility that spectral compression + XX dynamics is insufficient for this task")

    lines.append("\n---\n")
    lines.append("*This is an internal pilot gate, not a scientific claim.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Experiment 0A: Global Quantum Micro-Reservoir Pilot")
    parser.add_argument("--graphs", type=int, default=2000, help="Number of graphs to generate")
    parser.add_argument("--k", type=int, nargs="+", default=[8, 12], help="Compression sizes")
    parser.add_argument("--probe", type=str, default="P2", help="Probe name (P1-P4)")
    parser.add_argument("--outdir", type=str, default="qmt/results", help="Output directory")
    parser.add_argument("--budget", type=int, default=500, help="Solver budget per graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_stage0(
        n_graphs=args.graphs,
        k_values=args.k,
        probe_name=args.probe,
        outdir=args.outdir,
        solver_budget=args.budget,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
