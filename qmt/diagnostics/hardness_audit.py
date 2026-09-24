"""Hardness target audit: diagnose R²_C = -3734.83.

Checks:
- Var(Y_train), Var(Y_test) per fold
- min/max/median/std of targets
- Extreme outliers
- NaN/Inf
- Naive baseline R² (predict mean)
- Train/test scaling consistency
- Log numerator/denominator of R²
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import json
import os


def audit_hardness(
    X_classical: np.ndarray,
    Y_hardness: np.ndarray,
    Q: np.ndarray | None = None,
    n_folds: int = 5,
) -> dict:
    """Full audit of the hardness target pipeline.

    Returns
    -------
    dict with all audit results
    """
    Y = np.asarray(Y_hardness, dtype=float).ravel()
    X = np.nan_to_num(np.asarray(X_classical, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    audit = {}

    # 1. Basic statistics
    audit["basic_stats"] = {
        "n": len(Y),
        "min": float(np.min(Y)),
        "max": float(np.max(Y)),
        "mean": float(np.mean(Y)),
        "median": float(np.median(Y)),
        "std": float(np.std(Y)),
        "var": float(np.var(Y)),
        "n_nan": int(np.sum(~np.isfinite(Y_hardness.ravel()))),
        "n_zero": int(np.sum(Y == 0)),
        "n_near_zero": int(np.sum(np.abs(Y) < 1e-6)),
        "n_extreme": int(np.sum(np.abs(Y) > np.median(np.abs(Y)) * 100)),
        "percentiles": {
            "1%": float(np.percentile(Y, 1)),
            "5%": float(np.percentile(Y, 5)),
            "25%": float(np.percentile(Y, 25)),
            "75%": float(np.percentile(Y, 75)),
            "95%": float(np.percentile(Y, 95)),
            "99%": float(np.percentile(Y, 99)),
        },
    }

    # 2. Naive baseline: predict mean
    y_mean = np.mean(Y)
    ss_tot = np.sum((Y - y_mean) ** 2)
    ss_res_naive = np.sum((Y - y_mean) ** 2)  # = ss_tot
    r2_naive = 1.0 - ss_res_naive / max(ss_tot, 1e-20)
    audit["naive_baseline"] = {
        "r2": float(r2_naive),  # Should be exactly 0
        "ss_tot": float(ss_tot),
        "ss_res": float(ss_res_naive),
        "y_mean": float(y_mean),
    }

    # 3. In-sample Ridge (replicates the broken test)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = Ridge(alpha=1.0)
    model.fit(Xs, Y)
    y_pred = model.predict(Xs)
    ss_res_ridge = np.sum((Y - y_pred) ** 2)
    r2_insample = 1.0 - ss_res_ridge / max(ss_tot, 1e-20)
    audit["in_sample_ridge"] = {
        "r2": float(r2_insample),
        "ss_tot": float(ss_tot),
        "ss_res": float(ss_res_ridge),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "coef_norm": float(np.linalg.norm(model.coef_)),
    }

    # 4. Cross-validated Ridge
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        sc = StandardScaler().fit(X_tr)
        Xtr_s = sc.transform(X_tr)
        Xva_s = sc.transform(X_va)

        m = Ridge(alpha=1.0)
        m.fit(Xtr_s, Y_tr)
        y_pred_va = m.predict(Xva_s)

        ss_tot_fold = np.sum((Y_va - Y_va.mean()) ** 2)
        ss_res_fold = np.sum((Y_va - y_pred_va) ** 2)
        r2_fold = 1.0 - ss_res_fold / max(ss_tot_fold, 1e-20)

        # Naive fold baseline
        y_pred_naive = np.full_like(Y_va, Y_tr.mean())
        ss_res_naive_fold = np.sum((Y_va - y_pred_naive) ** 2)
        r2_naive_fold = 1.0 - ss_res_naive_fold / max(ss_tot_fold, 1e-20)

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "y_train_mean": float(Y_tr.mean()),
            "y_train_var": float(np.var(Y_tr)),
            "y_val_mean": float(Y_va.mean()),
            "y_val_var": float(np.var(Y_va)),
            "y_val_min": float(np.min(Y_va)),
            "y_val_max": float(np.max(Y_va)),
            "r2_ridge": float(r2_fold),
            "r2_naive": float(r2_naive_fold),
            "ss_tot_fold": float(ss_tot_fold),
            "ss_res_fold": float(ss_res_fold),
            "pred_min": float(np.min(y_pred_va)),
            "pred_max": float(np.max(y_pred_va)),
        })

    audit["cv_folds"] = fold_results
    audit["cv_mean_r2"] = float(np.mean([f["r2_ridge"] for f in fold_results]))
    audit["cv_mean_r2_naive"] = float(np.mean([f["r2_naive"] for f in fold_results]))

    # 5. Diagnosis
    diagnosis = []
    if audit["basic_stats"]["var"] < 1e-6:
        diagnosis.append("Near-zero target variance — R² is numerically unstable")
    if audit["basic_stats"]["n_near_zero"] > len(Y) * 0.3:
        diagnosis.append(f"{audit['basic_stats']['n_near_zero']} near-zero targets — degenerate distribution")
    if audit["in_sample_ridge"]["pred_max"] > audit["basic_stats"]["max"] * 10:
        diagnosis.append("Ridge predictions explode — model is extrapolating wildly")
    if audit["in_sample_ridge"]["pred_min"] < audit["basic_stats"]["min"] * 10:
        diagnosis.append("Ridge predictions have extreme negative values")
    if any(f["y_val_var"] < 1e-8 for f in fold_results):
        diagnosis.append("Some folds have near-zero val variance — fold degeneracy")
    if audit["in_sample_ridge"]["r2"] < -1:
        diagnosis.append(f"In-sample R² = {audit['in_sample_ridge']:.2f} — Ridge is catastrophically overfitting")

    # Check if the issue is target scaling
    Y_log = np.log1p(Y)
    scaler2 = StandardScaler().fit(X)
    Xs2 = scaler2.transform(X)
    m2 = Ridge(alpha=1.0)
    m2.fit(Xs2, Y_log)
    y_pred_log = m2.predict(Xs2)
    ss_res_log = np.sum((Y_log - y_pred_log) ** 2)
    ss_tot_log = np.sum((Y_log - Y_log.mean()) ** 2)
    r2_log = 1.0 - ss_res_log / max(ss_tot_log, 1e-20)
    audit["log_transformed_r2"] = float(r2_log)

    if r2_log > 0 and audit["in_sample_ridge"]["r2"] < -1:
        diagnosis.append(f"Log-transform fixes R² to {r2_log:.4f} — target needs log1p transform")

    audit["diagnosis"] = diagnosis

    return audit


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="qmt/results")
    parser.add_argument("--outdir", type=str, default="qmt/results/audit")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X = np.load(os.path.join(args.results_dir, "X_classical.npy"))
    Y = np.load(os.path.join(args.results_dir, "Y_hardness.npy"))
    Q = np.load(os.path.join(args.results_dir, "Q_k8_P2.npy"))

    audit = audit_hardness(X, Y, Q)

    print("=" * 60)
    print("HARDNESS TARGET AUDIT")
    print("=" * 60)
    print(f"\nBasic stats:")
    for k, v in audit["basic_stats"].items():
        if k != "percentiles":
            print(f"  {k}: {v}")
    print(f"\nPercentiles:")
    for k, v in audit["basic_stats"]["percentiles"].items():
        print(f"  {k}: {v:.6f}")

    print(f"\nNaive baseline R²: {audit['naive_baseline']['r2']:.6f} (should be 0)")
    print(f"In-sample Ridge R²: {audit['in_sample_ridge']['r2']:.4f}")
    print(f"  ss_tot: {audit['in_sample_ridge']['ss_tot']:.6f}")
    print(f"  ss_res: {audit['in_sample_ridge']['ss_res']:.6f}")
    print(f"  pred range: [{audit['in_sample_ridge']['pred_min']:.4f}, {audit['in_sample_ridge']['pred_max']:.4f}]")
    print(f"  coef_norm: {audit['in_sample_ridge']['coef_norm']:.4f}")

    print(f"\nCV Ridge R² (mean): {audit['cv_mean_r2']:.4f}")
    print(f"CV Naive R² (mean): {audit['cv_mean_r2_naive']:.4f}")
    for f in audit["cv_folds"]:
        print(f"  Fold {f['fold']}: r2={f['r2_ridge']:.4f} naive={f['r2_naive']:.4f} "
              f"val_var={f['y_val_var']:.6f} pred_range=[{f['pred_min']:.4f}, {f['pred_max']:.4f}]")

    print(f"\nLog-transformed R²: {audit['log_transformed_r2']:.4f}")

    print(f"\nDiagnosis:")
    for d in audit["diagnosis"]:
        print(f"  - {d}")

    with open(os.path.join(args.outdir, "hardness_audit.json"), "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nFull audit saved to {args.outdir}/hardness_audit.json")


if __name__ == "__main__":
    main()
