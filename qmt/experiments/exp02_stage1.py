"""Experiment 0B / Stage 1: Full condition matrix M0-M6.

Conditions:
  M0: Classical baseline (12 descriptors → micro-student)
  M1: Quantum distillation (spectral Q → teacher → student)
  M2: Shuffled control (shuffled Q → teacher → student)
  M3: Random compression control (random Q → teacher → student)
  M4: Classical compressed teacher (compressed classical features → teacher)
  M5: Direct quantum features (Q as input, no distillation)
  M6: Oracle (large model on C+Q)

Sweep: λ_q ∈ {0, 0.1, 0.3, 1.0, 3.0} × hidden ∈ {2, 3, 4, 5}

Usage:
    PYTHONPATH=. python -m qmt.experiments.exp02_stage1 \
        --graphs 2000 --k 8 --probe P2 --outdir qmt/results/stage1 --budget 50
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
from ..solvers.portfolio import run_portfolio, check_solver_dominance, SOLVER_NAMES
from ..compression.spectral import SpectralCompressor
from ..compression.random_control import RandomCompressor
from ..quantum.probes import get_probe
from ..quantum.observables import extract_observables_batch
from ..classical.compressed_features import extract_classical_compressed_batch
from ..distillation.trainer import run_condition, fit_teacher, predict_teacher


def run_stage1(
    n_graphs: int = 2000,
    k: int = 8,
    probe_name: str = "P2",
    outdir: str = "qmt/results/stage1",
    solver_budget: int = 50,
    seed: int = 42,
    epochs: int = 300,
    lambda_qs: list[float] | None = None,
    hiddens: list[int] | None = None,
):
    if lambda_qs is None:
        lambda_qs = [0.0, 0.1, 0.3, 1.0, 3.0]
    if hiddens is None:
        hiddens = [2, 3, 4, 5]

    os.makedirs(outdir, exist_ok=True)
    print(f"=" * 70)
    print(f"Stage 1: Full Condition Matrix (M0-M6)")
    print(f"  Graphs: {n_graphs} | k={k} | Probe={probe_name}")
    print(f"  λ_q sweep: {lambda_qs}")
    print(f"  Hidden sweep: {hiddens}")
    print(f"=" * 70)

    # Step 1: Generate graphs
    t0 = time.time()
    print(f"\n[1] Generating {n_graphs} graphs...")
    splits = make_splits(n_train=n_graphs, n_ood_size=0, n_ood_family=0, seed=seed)
    all_graphs = splits["train"]
    train_graphs, val_graphs = train_val_split(all_graphs, val_frac=0.15, seed=seed)
    n_train = len(train_graphs)
    all_graphs = train_graphs + val_graphs
    print(f"  Train: {n_train} | Val: {len(val_graphs)}")
    print(f"  Data gen: {time.time()-t0:.1f}s")

    # Step 2: Run solver portfolio
    t0 = time.time()
    print(f"\n[2] Running solver portfolio...")
    solver_results = []
    for i, G in enumerate(all_graphs):
        sr = run_portfolio(G, budget=solver_budget, seed=seed + i)
        sr.graph_id = f"g{i}"
        solver_results.append(sr)
        if (i + 1) % 500 == 0:
            print(f"  Solved {i+1}/{len(all_graphs)} ({time.time()-t0:.1f}s)")
    print(f"  Solver portfolio: {time.time()-t0:.1f}s")

    # Step 3: Dataset validity gate
    print(f"\n[3] Checking solver dominance...")
    dominance = check_solver_dominance(solver_results)
    print(f"  Win fractions: {dominance['fractions']}")
    print(f"  Max fraction: {dominance['max_fraction']:.3f}")
    print(f"  Gate: {'PASS' if dominance['pass'] else 'FAIL'}")

    # Step 4: Compute classical descriptors
    t0 = time.time()
    print(f"\n[4] Computing classical descriptors...")
    X_classical = compute_descriptors_batch(all_graphs)
    X_classical = np.nan_to_num(X_classical, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  X_classical: {X_classical.shape}")

    # Build targets
    Y = np.stack([sr.normalized for sr in solver_results])
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    Y_hardness = np.array([sr.hardness for sr in solver_results])
    Y_hardness = np.nan_to_num(Y_hardness, nan=0.0, posinf=0.0, neginf=0.0)

    # Split
    Xc_tr, Xc_va = X_classical[:n_train], X_classical[n_train:]
    Y_tr, Y_va = Y[:n_train], Y[n_train:]
    Yh_tr, Yh_va = Y_hardness[:n_train], Y_hardness[n_train:]

    # Step 5: Compress graphs (spectral + random)
    t0 = time.time()
    print(f"\n[5] Compressing graphs to k={k}...")
    spectral_compressor = SpectralCompressor()
    random_compressor = RandomCompressor(seed=seed)

    spectral_compressed = []
    random_compressed = []
    for i, G in enumerate(all_graphs):
        spectral_compressed.append(spectral_compressor.compress(G, k))
        random_compressed.append(random_compressor.compress(G, k))
        if (i + 1) % 500 == 0:
            print(f"  Compressed {i+1}/{len(all_graphs)} ({time.time()-t0:.1f}s)")
    print(f"  Compression: {time.time()-t0:.1f}s")

    # Step 6: Extract quantum features
    t0 = time.time()
    print(f"\n[6] Extracting quantum features (probe={probe_name})...")
    probe = get_probe(probe_name)
    Q_spectral = extract_observables_batch(spectral_compressed, probe)
    Q_spectral = np.nan_to_num(Q_spectral, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Q_spectral: {Q_spectral.shape}")

    Q_random = extract_observables_batch(random_compressed, probe)
    Q_random = np.nan_to_num(Q_random, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Q_random: {Q_random.shape}")
    print(f"  Quantum extraction: {time.time()-t0:.1f}s")

    # Step 7: Extract classical compressed features (M4 teacher)
    t0 = time.time()
    print(f"\n[7] Extracting classical compressed features (M4)...")
    C_compressed = extract_classical_compressed_batch(spectral_compressed)
    C_compressed = np.nan_to_num(C_compressed, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  C_compressed: {C_compressed.shape}")
    print(f"  Classical compressed: {time.time()-t0:.1f}s")

    # Split quantum and compressed features
    Qs_tr, Qs_va = Q_spectral[:n_train], Q_spectral[n_train:]
    Qr_tr, Qr_va = Q_random[:n_train], Q_random[n_train:]
    Cc_tr, Cc_va = C_compressed[:n_train], C_compressed[n_train:]

    # Step 8: Run all conditions
    print(f"\n[8] Running conditions M0-M6 with sweep...")

    all_results = []

    # --- M0: Classical baseline (no sweep, just hidden) ---
    print(f"\n  M0: Classical baseline")
    for h in hiddens:
        r = run_condition("M0", Xc_tr, Y_tr, Xc_va, Y_va, hidden=h, epochs=epochs, seed=seed)
        all_results.append(r)
        print(f"    h={h}: val_r2={r.val_r2:.4f} params={r.n_params}")

    # --- M1: Quantum distillation (sweep λ_q × hidden) ---
    print(f"\n  M1: Quantum distillation (spectral)")
    for lq in lambda_qs:
        for h in hiddens:
            r = run_condition("M1", Xc_tr, Y_tr, Xc_va, Y_va,
                              Qs_tr, Qs_va, hidden=h, lambda_q=lq, epochs=epochs, seed=seed)
            all_results.append(r)
            print(f"    λ_q={lq} h={h}: val_r2={r.val_r2:.4f} params={r.n_params}")

    # --- M2: Shuffled control (sweep λ_q × hidden) ---
    print(f"\n  M2: Shuffled control")
    for lq in lambda_qs:
        for h in hiddens:
            r = run_condition("M2", Xc_tr, Y_tr, Xc_va, Y_va,
                              Qs_tr, Qs_va, hidden=h, lambda_q=lq, epochs=epochs, seed=seed)
            all_results.append(r)
            print(f"    λ_q={lq} h={h}: val_r2={r.val_r2:.4f} params={r.n_params}")

    # --- M3: Random compression control (sweep λ_q × hidden) ---
    print(f"\n  M3: Random compression control")
    for lq in lambda_qs:
        for h in hiddens:
            r = run_condition("M3", Xc_tr, Y_tr, Xc_va, Y_va,
                              Qr_tr, Qr_va, hidden=h, lambda_q=lq, epochs=epochs, seed=seed)
            all_results.append(r)
            print(f"    λ_q={lq} h={h}: val_r2={r.val_r2:.4f} params={r.n_params}")

    # --- M4: Classical compressed teacher (sweep λ_q × hidden) ---
    print(f"\n  M4: Classical compressed teacher")
    for lq in lambda_qs:
        for h in hiddens:
            r = run_condition("M4", Xc_tr, Y_tr, Xc_va, Y_va,
                              Cc_tr, Cc_va, hidden=h, lambda_q=lq, epochs=epochs, seed=seed)
            all_results.append(r)
            print(f"    λ_q={lq} h={h}: val_r2={r.val_r2:.4f} params={r.n_params}")

    # --- M5: Direct quantum features (no sweep, just hidden) ---
    print(f"\n  M5: Direct quantum features")
    for h in hiddens:
        r = run_condition("M5", Xc_tr, Y_tr, Xc_va, Y_va,
                          Qs_tr, Qs_va, hidden=h, epochs=epochs, seed=seed)
        all_results.append(r)
        print(f"    h={h}: val_r2={r.val_r2:.4f} params={r.n_params}")

    # --- M6: Oracle ---
    print(f"\n  M6: Oracle (C+Q)")
    r = run_condition("M6", Xc_tr, Y_tr, Xc_va, Y_va,
                      Qs_tr, Qs_va, epochs=min(epochs, 200), seed=seed)
    all_results.append(r)
    print(f"    val_r2={r.val_r2:.4f}")

    # Step 9: Also run on hardness target
    print(f"\n[9] Running M0-M1 on hardness target...")
    hardness_results = []

    r_m0h = run_condition("M0", Xc_tr, Yh_tr.reshape(-1, 1), Xc_va, Yh_va.reshape(-1, 1),
                          hidden=4, epochs=epochs, seed=seed, task_dim=1)
    hardness_results.append(r_m0h)
    print(f"  M0 hardness: val_r2={r_m0h.val_r2:.4f}")

    for lq in [0.3, 1.0]:
        r_m1h = run_condition("M1", Xc_tr, Yh_tr.reshape(-1, 1), Xc_va, Yh_va.reshape(-1, 1),
                              Qs_tr, Qs_va, hidden=4, lambda_q=lq, epochs=epochs, seed=seed, task_dim=1)
        hardness_results.append(r_m1h)
        print(f"  M1 hardness λ_q={lq}: val_r2={r_m1h.val_r2:.4f}")

    # Step 10: Generate report
    print(f"\n[10] Generating STAGE1_REPORT.md...")
    report = generate_stage1_report(
        all_results, hardness_results, dominance,
        n_graphs, k, probe_name, lambda_qs, hiddens,
    )
    report_path = os.path.join(outdir, "STAGE1_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    # Save raw results
    results_json = []
    for r in all_results + hardness_results:
        results_json.append({
            "condition": r.condition,
            "train_r2": r.train_r2,
            "val_r2": r.val_r2,
            "n_params": r.n_params,
            "lambda_q": r.lambda_q,
            "hidden": r.hidden,
        })
    with open(os.path.join(outdir, "stage1_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # Save feature matrices
    np.save(os.path.join(outdir, "X_classical.npy"), X_classical)
    np.save(os.path.join(outdir, "Y_normalized.npy"), Y)
    np.save(os.path.join(outdir, "Y_hardness.npy"), Y_hardness)
    np.save(os.path.join(outdir, f"Q_spectral_k{k}_{probe_name}.npy"), Q_spectral)
    np.save(os.path.join(outdir, f"Q_random_k{k}_{probe_name}.npy"), Q_random)
    np.save(os.path.join(outdir, f"C_compressed_k{k}.npy"), C_compressed)

    print(f"\nDone. Results in {outdir}/")
    return all_results


def generate_stage1_report(
    all_results: list,
    hardness_results: list,
    dominance: dict,
    n_graphs: int,
    k: int,
    probe_name: str,
    lambda_qs: list,
    hiddens: list,
) -> str:
    """Generate STAGE1_REPORT.md."""
    lines = []
    lines.append("# STAGE 1 REPORT: Full Condition Matrix (M0-M6)\n")
    lines.append(f"**Graphs**: {n_graphs} | **k**: {k} | **Probe**: {probe_name}\n")
    lines.append(f"**λ_q sweep**: {lambda_qs} | **Hidden sweep**: {hiddens}\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Dataset validity
    lines.append("\n## 1. Dataset Validity\n")
    lines.append(f"- Solver win fractions: {dominance['fractions']}")
    lines.append(f"- Max fraction: {dominance['max_fraction']:.3f}")
    lines.append(f"- Gate: {'PASS' if dominance['pass'] else 'FAIL'}")

    # Group results by condition
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in all_results:
        by_cond[r.condition].append(r)

    # Summary table
    lines.append("\n## 2. Condition Summary (best val R² per condition)\n")
    lines.append("| Condition | Description | Best val R² | Best λ_q | Best h | Params |")
    lines.append("|-----------|-------------|-------------|----------|--------|--------|")

    cond_desc = {
        "M0": "Classical baseline (no quantum)",
        "M1": "Quantum distillation (spectral Q)",
        "M2": "Shuffled control (shuffled Q)",
        "M3": "Random compression control",
        "M4": "Classical compressed teacher",
        "M5": "Direct quantum features",
        "M6": "Oracle (C+Q, large model)",
    }

    best_per_cond = {}
    for cond in ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]:
        if cond in by_cond:
            results = by_cond[cond]
            best = max(results, key=lambda r: r.val_r2)
            best_per_cond[cond] = best
            lq_str = f"{best.lambda_q}" if best.lambda_q > 0 else "N/A"
            lines.append(f"| {cond} | {cond_desc[cond]} | {best.val_r2:.4f} | {lq_str} | {best.hidden} | {best.n_params} |")

    # Also compute best at λ_q > 0 for distillation conditions
    best_lq_pos = {}
    for cond in ["M1", "M2", "M3", "M4"]:
        if cond in by_cond:
            results_pos = [r for r in by_cond[cond] if r.lambda_q > 0]
            if results_pos:
                best_lq_pos[cond] = max(results_pos, key=lambda r: r.val_r2)

    lines.append("\n### Best at λ_q > 0 (distillation active)\n")
    lines.append("| Condition | Best val R² | Best λ_q | Best h |")
    lines.append("|-----------|-------------|----------|--------|")
    for cond in ["M1", "M2", "M3", "M4"]:
        if cond in best_lq_pos:
            r = best_lq_pos[cond]
            lines.append(f"| {cond} | {r.val_r2:.4f} | {r.lambda_q} | {r.hidden} |")

    # Detailed sweep tables
    lines.append("\n## 3. Detailed Sweep Results\n")

    for cond in ["M1", "M2", "M3", "M4"]:
        if cond not in by_cond:
            continue
        lines.append(f"\n### {cond}: {cond_desc[cond]}\n")
        lines.append("| λ_q \\ h | " + " | ".join(str(h) for h in hiddens) + " |")
        lines.append("|---------" * (len(hiddens) + 1) + "|")
        for lq in lambda_qs:
            row = f"| {lq} "
            for h in hiddens:
                matching = [r for r in by_cond[cond] if r.lambda_q == lq and r.hidden == h]
                if matching:
                    row += f"| {matching[0].val_r2:.4f} "
                else:
                    row += "| - "
            row += "|"
            lines.append(row)

    # M0 and M5 (no λ_q sweep)
    for cond in ["M0", "M5"]:
        if cond not in by_cond:
            continue
        lines.append(f"\n### {cond}: {cond_desc[cond]}\n")
        lines.append("| hidden | val R² | params |")
        lines.append("|--------|--------|--------|")
        for r in by_cond[cond]:
            lines.append(f"| {r.hidden} | {r.val_r2:.4f} | {r.n_params} |")

    # Key comparisons — use best at λ_q > 0 for distillation conditions
    lines.append("\n## 4. Key Comparisons (at λ_q > 0, distillation active)\n")

    m1_best = best_lq_pos.get("M1", best_per_cond.get("M1"))
    m2_best = best_lq_pos.get("M2", best_per_cond.get("M2"))
    m3_best = best_lq_pos.get("M3", best_per_cond.get("M3"))
    m4_best = best_lq_pos.get("M4", best_per_cond.get("M4"))

    if "M0" in best_per_cond and m1_best:
        delta = m1_best.val_r2 - best_per_cond["M0"].val_r2
        lines.append(f"- **M1 vs M0** (quantum distillation vs classical): ΔR² = {delta:+.4f} {'✓' if delta > 0 else '✗'}")

    if m1_best and m2_best:
        delta = m1_best.val_r2 - m2_best.val_r2
        lines.append(f"- **M1 vs M2** (real vs shuffled Q): ΔR² = {delta:+.4f} {'✓' if delta > 0 else '✗'}")

    if m1_best and m3_best:
        delta = m1_best.val_r2 - m3_best.val_r2
        lines.append(f"- **M1 vs M3** (spectral vs random compression): ΔR² = {delta:+.4f} {'✓' if delta > 0 else '✗'}")

    if m1_best and m4_best:
        delta = m1_best.val_r2 - m4_best.val_r2
        lines.append(f"- **M1 vs M4** (quantum vs classical compressed teacher): ΔR² = {delta:+.4f} {'✓' if delta > 0 else '✗'}")

    if "M5" in best_per_cond and "M0" in best_per_cond:
        delta = best_per_cond["M5"].val_r2 - best_per_cond["M0"].val_r2
        lines.append(f"- **M5 vs M0** (direct Q vs classical): ΔR² = {delta:+.4f} {'✓' if delta > 0 else '✗'}")

    if "M6" in best_per_cond and "M0" in best_per_cond:
        delta = best_per_cond["M6"].val_r2 - best_per_cond["M0"].val_r2
        lines.append(f"- **M6 vs M0** (oracle vs classical): ΔR² = {delta:+.4f}")

    # Hardness results
    if hardness_results:
        lines.append("\n## 5. Hardness Target\n")
        lines.append("| Condition | λ_q | val R² |")
        lines.append("|-----------|-----|--------|")
        for r in hardness_results:
            lines.append(f"| {r.condition} | {r.lambda_q} | {r.val_r2:.4f} |")

    # Verdict
    lines.append("\n## 6. Verdict\n")

    m1_beats_m0 = m1_best and m1_best.val_r2 > best_per_cond.get("M0", type('', (), {'val_r2': -999})).val_r2
    m1_beats_m2 = m1_best and m2_best and m1_best.val_r2 > m2_best.val_r2
    m1_beats_m3 = m1_best and m3_best and m1_best.val_r2 > m3_best.val_r2
    m1_beats_m4 = m1_best and m4_best and m1_best.val_r2 > m4_best.val_r2

    gates = [m1_beats_m0, m1_beats_m2, m1_beats_m3, m1_beats_m4]
    gate_names = ["M1 > M0", "M1 > M2", "M1 > M3", "M1 > M4"]

    for name, passed in zip(gate_names, gates):
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")

    n_pass = sum(gates)
    if n_pass >= 3:
        lines.append(f"\n**VERDICT: PROMOTE_TO_STAGE_2 = YES** ({n_pass}/4 gates passed)")
        lines.append("\nQuantum distillation shows genuine incremental signal. Proceed to Stage 2 with OOD evaluation.")
    elif n_pass >= 2:
        lines.append(f"\n**VERDICT: PROMOTE_TO_STAGE_2 = CONDITIONAL** ({n_pass}/4 gates passed)")
        lines.append("\nSome signal present but not all controls pass. Investigate before proceeding.")
    else:
        lines.append(f"\n**VERDICT: PROMOTE_TO_STAGE_2 = NO** ({n_pass}/4 gates passed)")
        lines.append("\nQuantum distillation does not show consistent incremental signal beyond controls.")

    lines.append("\n---\n")
    lines.append("*Internal research gate, not a scientific claim.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Full Condition Matrix")
    parser.add_argument("--graphs", type=int, default=2000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--probe", type=str, default="P2")
    parser.add_argument("--outdir", type=str, default="qmt/results/stage1")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lambda_qs", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--hiddens", type=int, nargs="+", default=[2, 3, 4, 5])
    args = parser.parse_args()

    run_stage1(
        n_graphs=args.graphs, k=args.k, probe_name=args.probe,
        outdir=args.outdir, solver_budget=args.budget, seed=args.seed,
        epochs=args.epochs, lambda_qs=args.lambda_qs, hiddens=args.hiddens,
    )


if __name__ == "__main__":
    main()
