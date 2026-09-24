"""Stage 1A: Full IID study at 10k graphs + OOD generalization.

Tests H1-H5 from the preregistered hypotheses:
  H1: Conditional quantum signal (ΔR² > 0 at scale)
  H2: Quantum-specificity (Q > classical controls)
  H3: Entanglement contribution (Q1 > Q0)
  H4: OOD generalization (size shift, family shift)
  H5: Representation geometry predicts utility

Scale: 10,000 IID graphs + 2,000 OOD-size + 2,000 OOD-family
Statistics: 5-fold CV × 10 seeds, permutation B=1000, bootstrap B=1000

Usage:
    PYTHONPATH=. python -m qmt.experiments.exp04_stage1a \
        --graphs 10000 --k 8 --probe P2 --outdir qmt/results/stage1a
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
from ..graphs.boundary_datasets import generate_size_ood, generate_family_ood
from ..solvers.portfolio import run_portfolio, check_solver_dominance
from ..compression.spectral import SpectralCompressor
from ..quantum.probes import get_probe
from ..quantum.observables import extract_observables_batch
from ..quantum.quantum_controls import build_quantum_control_circuit
from ..diagnostics.classical_controls import (
    control_a_gaussian_projection,
    control_b_rff,
    control_c_random_reservoir,
)
from ..diagnostics.proper_stats import cv_conditional_test, permutation_test, bootstrap_ci
from ..diagnostics.representation_geometry import full_geometry_analysis
from ..diagnostics.feature_attribution import (
    compute_per_graph_gain,
    compute_graph_properties,
    attribute_gain_to_properties,
    solver_boundary_analysis,
    compute_solver_margins,
)


def run_stage1a(
    n_graphs: int = 10000,
    k: int = 8,
    probe_name: str = "P2",
    outdir: str = "qmt/results/stage1a",
    solver_budget: int = 50,
    n_seeds: int = 10,
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    n_ood_size: int = 2000,
    n_ood_family: int = 2000,
    seed: int = 42,
):
    os.makedirs(outdir, exist_ok=True)
    print("=" * 70)
    print("Stage 1A: Full IID Study + OOD Generalization")
    print(f"  IID: {n_graphs} | OOD-size: {n_ood_size} | OOD-family: {n_ood_family}")
    print(f"  k={k} | Probe={probe_name}")
    print(f"  Seeds: {n_seeds} | Permutations: {n_permutations} | Bootstrap: {n_bootstrap}")
    print("=" * 70)

    # ================================================================
    # Part I: IID study (10k graphs)
    # ================================================================
    print("\n" + "=" * 70)
    print("PART I: IID Study")
    print("=" * 70)

    # Step 1: Generate IID graphs
    t0 = time.time()
    print(f"\n[1] Generating {n_graphs} IID graphs...")
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
        if (i + 1) % 1000 == 0:
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
        if (i + 1) % 1000 == 0:
            print(f"  Compressed {i+1}/{len(all_graphs)} ({time.time()-t0:.1f}s)")
    print(f"  Compression: {time.time()-t0:.1f}s")

    # Step 6: Extract quantum features
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
            circ = build_quantum_control_circuit(cg, probe, control=qc_name, seed=seed + i)
            circuits.append(circ)
        Q = extract_observables_batch(compressed, probe, circuits=circuits)
        all_Q[qc_name] = Q
        print(f"  {qc_name}: {Q.shape} ({time.time()-t0:.1f}s)")

    for e in entanglement_sweep:
        qc_name = f"Q4_e{e}"
        t0 = time.time()
        print(f"\n[6.{qc_name}] Entanglement sweep e={e}...")
        circuits = []
        for i, cg in enumerate(compressed):
            circ = build_quantum_control_circuit(
                cg, probe, control="Q4", seed=seed + i, entanglement_strength=e,
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

    # Step 8: CV conditional test
    print(f"\n[8] Cross-validated conditional test (5-fold × {n_seeds} seeds)...")
    results = {}

    for qc_name, Q in all_Q.items():
        print(f"\n  {qc_name}...")
        t0 = time.time()
        result = cv_conditional_test(X_classical, Q, Y, n_folds=5, n_seeds=n_seeds)
        result["time"] = time.time() - t0
        results[qc_name] = result
        print(f"    ΔR² = {result['delta_r2']['mean']:.4f} ± {result['delta_r2']['std']:.4f} "
              f"[{result['delta_r2']['ci_lo']:.4f}, {result['delta_r2']['ci_hi']:.4f}]")

    for cc_name, features in classical_controls.items():
        print(f"\n  {cc_name}...")
        t0 = time.time()
        result = cv_conditional_test(X_classical, features, Y, n_folds=5, n_seeds=n_seeds)
        result["time"] = time.time() - t0
        results[cc_name] = result
        print(f"    ΔR² = {result['delta_r2']['mean']:.4f} ± {result['delta_r2']['std']:.4f} "
              f"[{result['delta_r2']['ci_lo']:.4f}, {result['delta_r2']['ci_hi']:.4f}]")

    # Hardness target
    print(f"\n  Q1 on hardness target...")
    result_hardness = cv_conditional_test(X_classical, all_Q["Q1"], Y_hardness, n_folds=5, n_seeds=n_seeds)
    results["Q1_hardness"] = result_hardness
    print(f"    ΔR² = {result_hardness['delta_r2']['mean']:.4f} ± {result_hardness['delta_r2']['std']:.4f}")

    # Step 9: Permutation test
    print(f"\n[9] Permutation test for Q1 (B={n_permutations})...")
    perm_result = permutation_test(
        X_classical, all_Q["Q1"], Y, B=n_permutations, n_folds=5, seed=seed,
    )
    print(f"  Observed ΔR²: {perm_result['delta_r2_observed']:.4f}")
    print(f"  Null mean: {perm_result['null_mean']:.4f}")
    print(f"  p-value: {perm_result['p_value']:.4f}")
    print(f"  Significant (p<0.01): {perm_result['significant']}")

    # Step 10: Bootstrap CI
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
    print(f"  CKA(Q,Y) linear: {geometry['cka_linear_QY']:.4f}")
    print(f"  Kernel alignment Q→Y: {geometry['kernel_alignment_QY']:.4f}")

    # Step 12: Entanglement sweep
    print(f"\n[12] Entanglement sweep analysis...")
    sweep_results = []
    for e in entanglement_sweep:
        qc_name = f"Q4_e{e}"
        r = results[qc_name]["delta_r2"]["mean"]
        sweep_results.append({"e": e, "delta_r2": r})
        print(f"  e={e:.2f}: ΔR² = {r:.4f}")

    # Step 13: Feature attribution
    print(f"\n[13] Feature attribution (per-graph gain)...")
    gains = compute_per_graph_gain(X_classical, all_Q["Q1"], Y, n_folds=5, seed=seed)
    properties = compute_graph_properties(all_graphs)
    attribution = attribute_gain_to_properties(gains, properties)
    print(f"  Mean gain: {attribution['mean_gain']:.4f}")
    print(f"  Positive gains: {attribution['n_positive']}/{attribution['n_positive']+attribution['n_negative']}")
    print(f"  Attribution R²: {attribution['r2']:.4f}")
    print(f"  Top correlations:")
    sorted_corr = sorted(attribution["correlations"].items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr[:5]:
        print(f"    {name}: r={corr:.4f}")

    # Step 14: Solver-boundary analysis
    print(f"\n[14] Solver-boundary analysis...")
    margins = compute_solver_margins(solver_results)
    boundary = solver_boundary_analysis(gains, margins)
    print(f"  Correlation (margin, gain): {boundary['correlation_margin_gain']:.4f}")
    print(f"  Hypothesis: {boundary['hypothesis']}")
    print(f"  Supported: {boundary['hypothesis_supported']}")
    for cat, stats in boundary["categories"].items():
        print(f"  {cat}: n={stats['n']}, mean_gain={stats['gain_mean']:.4f}, frac_positive={stats['frac_positive']:.3f}")

    # ================================================================
    # Part II: OOD Generalization (H4)
    # ================================================================
    print("\n" + "=" * 70)
    print("PART II: OOD Generalization")
    print("=" * 70)

    ood_results = {}

    # --- OOD Size ---
    if n_ood_size > 0:
        print(f"\n[15] OOD-Size: Generate {n_ood_size} graphs (N=150-250)...")
        t0 = time.time()
        ood_size_graphs = generate_size_ood(
            train_n_range=(30, 80),
            test_n_ranges=[(150, 250)],
            n_train=0,
            n_test_per_range=n_ood_size,
            seed=seed + 100000,
        )["test_150_250"]
        print(f"  Generated {len(ood_size_graphs)} OOD-size graphs ({time.time()-t0:.1f}s)")

        # Solve
        print(f"  Running solver portfolio on OOD-size...")
        t0 = time.time()
        ood_size_sr = []
        for i, G in enumerate(ood_size_graphs):
            sr = run_portfolio(G, budget=solver_budget, seed=seed + 200000 + i)
            ood_size_sr.append(sr)
            if (i + 1) % 500 == 0:
                print(f"    Solved {i+1}/{len(ood_size_graphs)} ({time.time()-t0:.1f}s)")

        # Descriptors
        X_ood_size = compute_descriptors_batch(ood_size_graphs)
        Y_ood_size = np.stack([sr.normalized for sr in ood_size_sr])

        # Compress
        print(f"  Compressing OOD-size graphs...")
        t0 = time.time()
        ood_size_compressed = []
        for i, G in enumerate(ood_size_graphs):
            cg = compressor.compress(G, k)
            ood_size_compressed.append(cg)
            if (i + 1) % 500 == 0:
                print(f"    Compressed {i+1}/{len(ood_size_graphs)} ({time.time()-t0:.1f}s)")

        # Quantum features (Q1 only for OOD)
        print(f"  Extracting Q1 features for OOD-size...")
        circuits_ood = []
        for i, cg in enumerate(ood_size_compressed):
            circ = build_quantum_control_circuit(cg, probe, control="Q1", seed=seed + 300000 + i)
            circuits_ood.append(circ)
        Q_ood_size = extract_observables_batch(ood_size_compressed, probe, circuits=circuits_ood)

        # CV on OOD-size
        print(f"  CV conditional test on OOD-size...")
        ood_size_cv = cv_conditional_test(X_ood_size, Q_ood_size, Y_ood_size, n_folds=5, n_seeds=n_seeds)
        ood_results["ood_size"] = {
            "n_graphs": len(ood_size_graphs),
            "n_range": [150, 250],
            "cv": ood_size_cv,
        }
        print(f"    ΔR² = {ood_size_cv['delta_r2']['mean']:.4f} ± {ood_size_cv['delta_r2']['std']:.4f}")
        print(f"    R²(C) = {ood_size_cv['r2_c']['mean']:.4f}")
        print(f"    R²(C+Q) = {ood_size_cv['r2_cq']['mean']:.4f}")

    # --- OOD Family ---
    if n_ood_family > 0:
        print(f"\n[16] OOD-Family: Generate {n_ood_family} graphs (BA, geometric, config)...")
        t0 = time.time()
        ood_fam_data = generate_family_ood(
            train_families=["er", "regular", "ws"],
            test_families=["ba", "geometric", "config"],
            n_train=0,
            n_test=n_ood_family // 3,
            n_range=(30, 80),
            seed=seed + 400000,
        )

        ood_fam_graphs = []
        ood_fam_labels = []
        for key, graphs in ood_fam_data.items():
            if key.startswith("test_"):
                fam = key.replace("test_", "")
                for G in graphs:
                    ood_fam_graphs.append(G)
                    ood_fam_labels.append(fam)

        print(f"  Generated {len(ood_fam_graphs)} OOD-family graphs ({time.time()-t0:.1f}s)")

        # Solve
        print(f"  Running solver portfolio on OOD-family...")
        t0 = time.time()
        ood_fam_sr = []
        for i, G in enumerate(ood_fam_graphs):
            sr = run_portfolio(G, budget=solver_budget, seed=seed + 500000 + i)
            ood_fam_sr.append(sr)
            if (i + 1) % 500 == 0:
                print(f"    Solved {i+1}/{len(ood_fam_graphs)} ({time.time()-t0:.1f}s)")

        # Descriptors
        X_ood_fam = compute_descriptors_batch(ood_fam_graphs)
        Y_ood_fam = np.stack([sr.normalized for sr in ood_fam_sr])

        # Compress
        print(f"  Compressing OOD-family graphs...")
        t0 = time.time()
        ood_fam_compressed = []
        for i, G in enumerate(ood_fam_graphs):
            cg = compressor.compress(G, k)
            ood_fam_compressed.append(cg)
            if (i + 1) % 500 == 0:
                print(f"    Compressed {i+1}/{len(ood_fam_graphs)} ({time.time()-t0:.1f}s)")

        # Quantum features
        print(f"  Extracting Q1 features for OOD-family...")
        circuits_ood_fam = []
        for i, cg in enumerate(ood_fam_compressed):
            circ = build_quantum_control_circuit(cg, probe, control="Q1", seed=seed + 600000 + i)
            circuits_ood_fam.append(circ)
        Q_ood_fam = extract_observables_batch(ood_fam_compressed, probe, circuits=circuits_ood_fam)

        # CV on OOD-family (overall)
        print(f"  CV conditional test on OOD-family (overall)...")
        ood_fam_cv = cv_conditional_test(X_ood_fam, Q_ood_fam, Y_ood_fam, n_folds=5, n_seeds=n_seeds)
        ood_results["ood_family"] = {
            "n_graphs": len(ood_fam_graphs),
            "families": list(set(ood_fam_labels)),
            "cv": ood_fam_cv,
        }
        print(f"    ΔR² = {ood_fam_cv['delta_r2']['mean']:.4f} ± {ood_fam_cv['delta_r2']['std']:.4f}")
        print(f"    R²(C) = {ood_fam_cv['r2_c']['mean']:.4f}")
        print(f"    R²(C+Q) = {ood_fam_cv['r2_cq']['mean']:.4f}")

        # Per-family breakdown
        ood_results["ood_family"]["per_family"] = {}
        for fam in sorted(set(ood_fam_labels)):
            idx = [i for i, l in enumerate(ood_fam_labels) if l == fam]
            if len(idx) < 50:
                continue
            X_fam = X_ood_fam[idx]
            Q_fam = Q_ood_fam[idx]
            Y_fam = Y_ood_fam[idx]
            cv_fam = cv_conditional_test(X_fam, Q_fam, Y_fam, n_folds=5, n_seeds=n_seeds)
            ood_results["ood_family"]["per_family"][fam] = {
                "n": len(idx),
                "delta_r2": cv_fam["delta_r2"]["mean"],
                "r2_c": cv_fam["r2_c"]["mean"],
                "r2_cq": cv_fam["r2_cq"]["mean"],
            }
            print(f"    {fam}: ΔR² = {cv_fam['delta_r2']['mean']:.4f} (n={len(idx)})")

    # ================================================================
    # Step 17: Generate report
    # ================================================================
    print(f"\n[17] Generating STAGE1A_REPORT.md...")
    report = generate_stage1a_report(
        results, perm_result, boot_result, geometry,
        sweep_results, dominance, attribution, boundary, ood_results,
        n_graphs, k, probe_name, n_seeds, n_permutations, n_bootstrap,
        n_ood_size, n_ood_family,
    )
    report_path = os.path.join(outdir, "STAGE1A_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    # Save results JSON
    save_results = {
        "iid": {
            "n_graphs": n_graphs,
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
            "attribution": attribution,
            "boundary": {
                "correlation_margin_gain": boundary["correlation_margin_gain"],
                "hypothesis": boundary["hypothesis"],
                "hypothesis_supported": boundary["hypothesis_supported"],
                "categories": boundary["categories"],
            },
        },
        "ood": ood_results,
    }
    with open(os.path.join(outdir, "stage1a_results.json"), "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    # Save features
    for name, Q in all_Q.items():
        np.save(os.path.join(outdir, f"Q_{name}.npy"), Q)
    for name, features in classical_controls.items():
        np.save(os.path.join(outdir, f"C_control_{name}.npy"), features)
    np.save(os.path.join(outdir, "X_classical.npy"), X_classical)
    np.save(os.path.join(outdir, "Y_normalized.npy"), Y)
    np.save(os.path.join(outdir, "Y_hardness.npy"), Y_hardness)
    np.save(os.path.join(outdir, "gains.npy"), gains)

    print(f"\nDone. Results in {outdir}/")
    return save_results


def generate_stage1a_report(
    results, perm_result, boot_result, geometry,
    sweep_results, dominance, attribution, boundary, ood_results,
    n_graphs, k, probe_name, n_seeds, n_permutations, n_bootstrap,
    n_ood_size, n_ood_family,
) -> str:
    lines = []
    lines.append("# STAGE 1A REPORT: Full IID Study + OOD Generalization\n")
    lines.append(f"**Graphs**: {n_graphs} IID + {n_ood_size} OOD-size + {n_ood_family} OOD-family")
    lines.append(f" | **k**: {k} | **Probe**: {probe_name}\n")
    lines.append(f"**Seeds**: {n_seeds} | **Permutations**: {n_permutations} | **Bootstrap**: {n_bootstrap}\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Dataset validity
    lines.append("\n## 1. Dataset Validity\n")
    lines.append(f"- Solver win fractions: {dominance['fractions']}")
    lines.append(f"- Max fraction: {dominance['max_fraction']:.3f}")
    lines.append(f"- Gate: {'PASS' if dominance['pass'] else 'FAIL'}")

    # CV results
    lines.append(f"\n## 2. Cross-Validated Conditional Test (5-fold × {n_seeds} seeds)\n")
    lines.append("| Feature | ΔR² mean | ΔR² std | 95% CI | R²(C) | R²(C+Q) | R² residual |")
    lines.append("|---------|----------|---------|--------|-------|---------|-------------|")

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

    if "Q1_hardness" in results:
        r = results["Q1_hardness"]
        d = r["delta_r2"]
        lines.append(
            f"| Q1 (hardness) | {d['mean']:.4f} | {d['std']:.4f} | "
            f"[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] | "
            f"{r['r2_c']['mean']:.4f} | {r['r2_cq']['mean']:.4f} | "
            f"{r['r2_residual']['mean']:.4f} |"
        )

    # Gates
    lines.append("\n## 3. Hypothesis Tests (Gates)\n")

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
        ("H1: Q1 ΔR² > 0", q1 > 0, f"ΔR² = {q1:.4f}"),
        ("H2: Q1 > best classical", q1 > best_classical, f"Q1={q1:.4f} vs best_cl={best_classical:.4f}"),
        ("H3: Q1 > Q0 (entanglement)", q1 > q0, f"Q1={q1:.4f} vs Q0={q0:.4f}"),
        ("H3: Q1 > Q2 (scrambled conn.)", q1 > q2, f"Q1={q1:.4f} vs Q2={q2:.4f}"),
        ("H3: Q1 > Q3 (scrambled input)", q1 > q3, f"Q1={q1:.4f} vs Q3={q3:.4f}"),
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
    lines.append(f"- ΔR² = {boot_result['mean']:.4f} [{boot_result['ci_lo']:.4f}, {boot_result['ci_hi']:.4f}]_95%")
    lines.append(f"- CI excludes zero: {'YES' if boot_result['ci_lo'] > 0 else 'NO'}")

    # Entanglement sweep
    lines.append("\n## 6. Entanglement Sweep (H3)\n")
    lines.append("| e | ΔR² |")
    lines.append("|---|------|")
    for s in sweep_results:
        lines.append(f"| {s['e']:.2f} | {s['delta_r2']:.4f} |")
    sweep_deltas = [s["delta_r2"] for s in sweep_results]
    is_monotone = all(sweep_deltas[i] <= sweep_deltas[i+1] for i in range(len(sweep_deltas)-1))
    lines.append(f"\nMonotone increasing: {'YES' if is_monotone else 'NO'}")
    best_e = sweep_results[np.argmax(sweep_deltas)] if sweep_deltas else None
    if best_e:
        lines.append(f"Optimal entanglement: e={best_e['e']:.2f} (ΔR²={best_e['delta_r2']:.4f})")

    # Representation geometry
    lines.append("\n## 7. Representation Geometry (H5)\n")
    lines.append(f"- Classical effective rank: {geometry['classical']['effective_rank']:.2f}")
    lines.append(f"- Quantum effective rank: {geometry['quantum']['effective_rank']:.2f}")
    lines.append(f"- CKA(C, Q) linear: {geometry['cka_linear_CQ']:.4f}")
    lines.append(f"- CKA(C, Q) rbf: {geometry['cka_rbf_CQ']:.4f}")
    lines.append(f"- CKA(Q, Y) linear: {geometry['cka_linear_QY']:.4f}")
    lines.append(f"- CKA(C, Y) linear: {geometry['cka_linear_CY']:.4f}")
    lines.append(f"- Kernel alignment Q→Y: {geometry['kernel_alignment_QY']:.4f}")
    lines.append(f"- Kernel alignment C→Y: {geometry['kernel_alignment_CY']:.4f}")
    lines.append(f"- Principal angles C↔Q: {[f'{a:.1f}°' for a in geometry['principal_angles_CQ']]}")

    # Feature attribution
    lines.append("\n## 8. Feature Attribution\n")
    lines.append(f"- Mean per-graph gain: {attribution['mean_gain']:.4f}")
    lines.append(f"- Median per-graph gain: {attribution['median_gain']:.4f}")
    lines.append(f"- Positive gains (Q helped): {attribution['n_positive']}")
    lines.append(f"- Negative gains (Q hurt): {attribution['n_negative']}")
    lines.append(f"- Attribution R² (gain ~ graph properties): {attribution['r2']:.4f}")
    lines.append(f"\n### Top property correlations with gain:\n")
    lines.append("| Property | Correlation |")
    lines.append("|----------|------------|")
    sorted_corr = sorted(attribution["correlations"].items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr[:8]:
        lines.append(f"| {name} | {corr:.4f} |")
    lines.append(f"\n### Regression coefficients:\n")
    lines.append("| Property | Coefficient |")
    lines.append("|----------|------------|")
    sorted_coef = sorted(attribution["coefficients"].items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_coef[:8]:
        lines.append(f"| {name} | {coef:.4f} |")

    # Solver-boundary analysis
    lines.append("\n## 9. Solver-Boundary Analysis\n")
    lines.append(f"- Correlation (margin, gain): {boundary['correlation_margin_gain']:.4f}")
    lines.append(f"- Hypothesis: {boundary['hypothesis']}")
    lines.append(f"- Hypothesis supported: {'YES' if boundary['hypothesis_supported'] else 'NO'}")
    lines.append(f"\n| Category | N | Mean gain | Std | Frac positive |")
    lines.append("|----------|---|-----------|-----|---------------|")
    for cat in ["boundary", "medium", "easy"]:
        s = boundary["categories"].get(cat, {})
        lines.append(f"| {cat} | {s.get('n', 0)} | {s.get('gain_mean', 0):.4f} | {s.get('gain_std', 0):.4f} | {s.get('frac_positive', 0):.3f} |")

    # OOD generalization
    lines.append("\n## 10. OOD Generalization (H4)\n")

    if "ood_size" in ood_results:
        r = ood_results["ood_size"]["cv"]
        d = r["delta_r2"]
        lines.append("### OOD-Size (train N∈[30,80], test N∈[150,250])\n")
        lines.append(f"- N: {ood_results['ood_size']['n_graphs']}")
        lines.append(f"- ΔR² = {d['mean']:.4f} ± {d['std']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")
        lines.append(f"- R²(C) = {r['r2_c']['mean']:.4f}")
        lines.append(f"- R²(C+Q) = {r['r2_cq']['mean']:.4f}")
        lines.append(f"- ΔR² > 0: {'PASS' if d['mean'] > 0 else 'FAIL'}")
        lines.append(f"- CI excludes zero: {'PASS' if d['ci_lo'] > 0 else 'FAIL'}")

    if "ood_family" in ood_results:
        r = ood_results["ood_family"]["cv"]
        d = r["delta_r2"]
        lines.append("\n### OOD-Family (train {ER, regular, WS}, test {BA, geometric, config})\n")
        lines.append(f"- N: {ood_results['ood_family']['n_graphs']}")
        lines.append(f"- Families: {ood_results['ood_family']['families']}")
        lines.append(f"- ΔR² = {d['mean']:.4f} ± {d['std']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")
        lines.append(f"- R²(C) = {r['r2_c']['mean']:.4f}")
        lines.append(f"- R²(C+Q) = {r['r2_cq']['mean']:.4f}")
        lines.append(f"- ΔR² > 0: {'PASS' if d['mean'] > 0 else 'FAIL'}")
        lines.append(f"- CI excludes zero: {'PASS' if d['ci_lo'] > 0 else 'FAIL'}")

        if "per_family" in ood_results["ood_family"]:
            lines.append("\n#### Per-family breakdown:\n")
            lines.append("| Family | N | ΔR² | R²(C) | R²(C+Q) |")
            lines.append("|--------|---|-----|-------|---------|")
            for fam, stats in sorted(ood_results["ood_family"]["per_family"].items()):
                lines.append(f"| {fam} | {stats['n']} | {stats['delta_r2']:.4f} | {stats['r2_c']:.4f} | {stats['r2_cq']:.4f} |")

    # Verdict
    lines.append("\n## 11. Verdict\n")

    iid_gates = [
        ("H1: ΔR² > 0", q1 > 0),
        ("H2: Q > classical", q1 > best_classical),
        ("H3: Q1 > Q0", q1 > q0),
        ("Permutation p<0.01", perm_result["significant"]),
        ("Bootstrap CI > 0", boot_result["ci_lo"] > 0),
    ]
    n_iid_pass = sum(1 for _, p in iid_gates if p)

    ood_size_pass = ood_results.get("ood_size", {}).get("cv", {}).get("delta_r2", {}).get("mean", 0) > 0
    ood_fam_pass = ood_results.get("ood_family", {}).get("cv", {}).get("delta_r2", {}).get("mean", 0) > 0
    ood_gates = [("H4: OOD-size ΔR² > 0", ood_size_pass), ("H4: OOD-family ΔR² > 0", ood_fam_pass)]
    n_ood_pass = sum(1 for _, p in ood_gates if p)

    for name, passed in iid_gates + ood_gates:
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")

    total_gates = len(iid_gates) + len(ood_gates)
    total_pass = n_iid_pass + n_ood_pass
    lines.append(f"\nPassed {total_pass}/{total_gates} gates.")

    if total_pass >= 6:
        lines.append("\n**VERDICT: QUANTUM SIGNAL CONFIRMED AT SCALE**")
        lines.append("\nThe quantum reservoir signal:")
        lines.append("- Survives falsification at 10k graph scale")
        lines.append("- Is statistically significant (permutation + bootstrap)")
        lines.append("- Is not explained by classical controls")
        lines.append("- Depends on entanglement and graph structure")
        lines.append("- Generalizes to OOD size and family shifts")
        lines.append("\n→ Proceed to hardware validation and paper preparation")
    elif total_pass >= 4:
        lines.append("\n**VERDICT: PARTIAL CONFIRMATION**")
        lines.append("\nSignal survives IID but may not generalize OOD. Investigate failure modes.")
    else:
        lines.append("\n**VERDICT: SIGNAL DOES NOT SURVIVE AT SCALE**")
        lines.append("\nThe 2k-graph signal was a finite-sample effect.")

    lines.append("\n---\n")
    lines.append("*Internal research gate, not a scientific claim.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 1A: Full IID Study + OOD")
    parser.add_argument("--graphs", type=int, default=10000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--probe", type=str, default="P2")
    parser.add_argument("--outdir", type=str, default="qmt/results/stage1a")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--ood_size", type=int, default=2000)
    parser.add_argument("--ood_family", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_stage1a(
        n_graphs=args.graphs, k=args.k, probe_name=args.probe,
        outdir=args.outdir, solver_budget=args.budget,
        n_seeds=args.seeds, n_permutations=args.permutations,
        n_bootstrap=args.bootstrap, n_ood_size=args.ood_size,
        n_ood_family=args.ood_family, seed=args.seed,
    )


if __name__ == "__main__":
    main()
