"""Proper statistics for the conditional information test.

Implements:
  - 5-fold CV × N seeds with all transformations inside fold
  - Permutation test (B=1000) for H_0: I(Q;Y|C) = 0
  - Bootstrap confidence intervals
  - Proper residual test (fit on train, evaluate on test)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import Callable


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² for possibly multi-output targets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-20))


def cv_conditional_test(
    X_classical: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    n_seeds: int = 10,
    alpha: float = 1.0,
) -> dict:
    """Cross-validated conditional information test.

    For each seed and fold:
      1. Fit classical model on train: y_C = f(C_train)
      2. Predict on test: y_pred_C = f(C_test)
      3. Compute residuals on test: r = Y_test - y_pred_C
      4. Fit quantum model on train residuals: g(Q_train) → r_train
      5. Predict on test: r_pred = g(Q_test)
      6. R²_residual = R²(r_test, r_pred)
      7. Also fit combined model (C+Q) and compute ΔR²

    All scaling/fitting happens inside the fold.

    Returns
    -------
    dict with per-seed, per-fold results + summary statistics
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    X_classical = np.nan_to_num(np.asarray(X_classical, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Q = np.nan_to_num(np.asarray(Q, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    all_r2_c = []
    all_r2_cq = []
    all_r2_res = []
    all_delta_r2 = []

    for seed in range(n_seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_classical)):
            Xc_tr, Xc_va = X_classical[tr_idx], X_classical[va_idx]
            Q_tr, Q_va = Q[tr_idx], Q[va_idx]
            Y_tr, Y_va = Y[tr_idx], Y[va_idx]

            # Classical model
            sc_c = StandardScaler().fit(Xc_tr)
            Xc_tr_s = sc_c.transform(Xc_tr)
            Xc_va_s = sc_c.transform(Xc_va)

            m_c = Ridge(alpha=alpha)
            m_c.fit(Xc_tr_s, Y_tr)
            pred_c = m_c.predict(Xc_va_s)
            pred_c_tr = m_c.predict(Xc_tr_s)
            # Ensure 2D shape matches Y
            if pred_c.ndim == 1:
                pred_c = pred_c.reshape(-1, 1)
            if pred_c_tr.ndim == 1:
                pred_c_tr = pred_c_tr.reshape(-1, 1)
            r2_c = _r2(Y_va, pred_c)

            # Residuals on train
            resid_tr = Y_tr - pred_c_tr

            # Quantum model on train residuals
            sc_q = StandardScaler().fit(Q_tr)
            Q_tr_s = sc_q.transform(Q_tr)
            Q_va_s = sc_q.transform(Q_va)

            m_q = Ridge(alpha=alpha)
            m_q.fit(Q_tr_s, resid_tr)
            res_pred = m_q.predict(Q_va_s)
            if res_pred.ndim == 1:
                res_pred = res_pred.reshape(-1, 1)

            # R² residual: how much of the test residual does Q explain?
            resid_va = Y_va - pred_c
            r2_res = _r2(resid_va, res_pred)

            # Combined model
            X_tr_combined = np.hstack([Xc_tr_s, Q_tr_s])
            X_va_combined = np.hstack([Xc_va_s, Q_va_s])
            m_cq = Ridge(alpha=alpha)
            m_cq.fit(X_tr_combined, Y_tr)
            pred_cq = m_cq.predict(X_va_combined)
            if pred_cq.ndim == 1:
                pred_cq = pred_cq.reshape(-1, 1)
            r2_cq = _r2(Y_va, pred_cq)

            delta_r2 = r2_cq - r2_c

            all_r2_c.append(r2_c)
            all_r2_cq.append(r2_cq)
            all_r2_res.append(r2_res)
            all_delta_r2.append(delta_r2)

    all_r2_c = np.array(all_r2_c)
    all_r2_cq = np.array(all_r2_cq)
    all_r2_res = np.array(all_r2_res)
    all_delta_r2 = np.array(all_delta_r2)

    return {
        "r2_c": {
            "mean": float(np.mean(all_r2_c)),
            "std": float(np.std(all_r2_c)),
            "median": float(np.median(all_r2_c)),
            "ci_lo": float(np.percentile(all_r2_c, 2.5)),
            "ci_hi": float(np.percentile(all_r2_c, 97.5)),
        },
        "r2_cq": {
            "mean": float(np.mean(all_r2_cq)),
            "std": float(np.std(all_r2_cq)),
            "median": float(np.median(all_r2_cq)),
            "ci_lo": float(np.percentile(all_r2_cq, 2.5)),
            "ci_hi": float(np.percentile(all_r2_cq, 97.5)),
        },
        "r2_residual": {
            "mean": float(np.mean(all_r2_res)),
            "std": float(np.std(all_r2_res)),
            "median": float(np.median(all_r2_res)),
            "ci_lo": float(np.percentile(all_r2_res, 2.5)),
            "ci_hi": float(np.percentile(all_r2_res, 97.5)),
        },
        "delta_r2": {
            "mean": float(np.mean(all_delta_r2)),
            "std": float(np.std(all_delta_r2)),
            "median": float(np.median(all_delta_r2)),
            "ci_lo": float(np.percentile(all_delta_r2, 2.5)),
            "ci_hi": float(np.percentile(all_delta_r2, 97.5)),
        },
        "n_evaluations": len(all_delta_r2),
        "n_folds": n_folds,
        "n_seeds": n_seeds,
    }


def permutation_test(
    X_classical: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    B: int = 1000,
    n_folds: int = 5,
    seed: int = 42,
    permute: str = "Q",
) -> dict:
    """Permutation test for H_0: I(Q;Y|C) = 0.

    Permutes Q (or Y) across instances and recomputes ΔR² each time.
    The observed ΔR² is compared against the null distribution.

    Parameters
    ----------
    B : int
        Number of permutations.
    permute : str
        "Q" = permute quantum features
        "Y" = permute targets
    """
    # Observed ΔR²
    observed = cv_conditional_test(X_classical, Q, Y, n_folds=n_folds, n_seeds=1)
    delta_obs = observed["delta_r2"]["mean"]

    # Null distribution
    rng = np.random.default_rng(seed)
    null_deltas = []

    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    Q_arr = np.asarray(Q, dtype=float)

    for b in range(B):
        perm = rng.permutation(Q_arr.shape[0])
        if permute == "Q":
            Q_perm = Q_arr[perm]
            result = cv_conditional_test(X_classical, Q_perm, Y_arr, n_folds=n_folds, n_seeds=1)
        else:
            Y_perm = Y_arr[perm]
            result = cv_conditional_test(X_classical, Q_arr, Y_perm, n_folds=n_folds, n_seeds=1)
        null_deltas.append(result["delta_r2"]["mean"])

        if (b + 1) % 100 == 0:
            print(f"  Permutation {b+1}/{B}")

    null_deltas = np.array(null_deltas)

    # p-value: fraction of null >= observed
    p_value = float(np.mean(null_deltas >= delta_obs))

    return {
        "delta_r2_observed": float(delta_obs),
        "null_mean": float(np.mean(null_deltas)),
        "null_std": float(np.std(null_deltas)),
        "null_median": float(np.median(null_deltas)),
        "p_value": p_value,
        "n_permutations": B,
        "null_distribution": null_deltas.tolist(),
        "significant": p_value < 0.01,
    }


def bootstrap_ci(
    X_classical: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 1000,
    n_folds: int = 5,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for ΔR².

    Resamples graph instances with replacement and recomputes ΔR².
    """
    n = X_classical.shape[0]
    rng = np.random.default_rng(seed)

    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)

    bootstrap_deltas = []

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        result = cv_conditional_test(
            X_classical[idx], Q[idx], Y_arr[idx],
            n_folds=n_folds, n_seeds=1,
        )
        bootstrap_deltas.append(result["delta_r2"]["mean"])

        if (b + 1) % 100 == 0:
            print(f"  Bootstrap {b+1}/{n_bootstrap}")

    bootstrap_deltas = np.array(bootstrap_deltas)

    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
    ci_hi = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))

    return {
        "mean": float(np.mean(bootstrap_deltas)),
        "std": float(np.std(bootstrap_deltas)),
        "median": float(np.median(bootstrap_deltas)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "distribution": bootstrap_deltas.tolist(),
    }
