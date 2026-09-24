"""Conditional information test: I(Q;Y|C).

The core kill-gate for Experiment 0A.
1. Fit classical predictor: y_C = f(x_G)
2. Compute residual: e = y - y_C
3. Fit quantum predictor: q(G) -> e
4. Measure R^2_residual and delta_rho
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


def conditional_residual_test(
    X_classical: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    classical_model: str = "ridge",
) -> dict:
    """Test whether quantum features contain incremental information about Y
    beyond classical descriptors.

    Parameters
    ----------
    X_classical : np.ndarray, shape (N, n_classical)
        Cheap classical descriptors (e.g., 12 graph features).
    Q : np.ndarray, shape (N, n_quantum)
        Quantum features from CGQP.
    Y : np.ndarray, shape (N,) or (N, n_targets)
        Target values (e.g., solver performance vector).
    classical_model : str
        "ridge" or "rf" for the classical predictor.

    Returns
    -------
    dict
        "r2_classical": R^2 of classical model on Y
        "r2_residual": R^2 of quantum model on residuals
        "delta_rho": Spearman improvement from adding Q
        "r2_combined": R^2 of (C,Q) combined model
        "r2_quantum_only": R^2 of Q-only model on Y
        "pass_gate": bool
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    X_classical = np.nan_to_num(np.asarray(X_classical, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Q = np.nan_to_num(np.asarray(Q, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1: Fit classical predictor
    if classical_model == "rf":
        clf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    else:
        clf = Ridge(alpha=1.0)

    scaler_c = StandardScaler().fit(X_classical)
    Xc = scaler_c.transform(X_classical)

    clf.fit(Xc, Y)
    y_pred_c = clf.predict(Xc)
    residual = Y - y_pred_c

    # R^2 of classical model
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    ss_res_c = np.sum(residual ** 2)
    r2_classical = 1.0 - ss_res_c / max(ss_tot, 1e-10)

    # Step 2: Fit quantum predictor on residuals
    scaler_q = StandardScaler().fit(Q)
    Xq = scaler_q.transform(Q)

    q_model = Ridge(alpha=1.0)
    q_model.fit(Xq, residual)
    res_pred = q_model.predict(Xq)
    ss_res_q = np.sum((residual - res_pred) ** 2)
    r2_residual = 1.0 - ss_res_q / max(ss_res_c, 1e-10)

    # Step 3: Combined model (C + Q)
    X_combined = np.hstack([Xc, Xq])
    combined_model = Ridge(alpha=1.0)
    combined_model.fit(X_combined, Y)
    y_pred_combined = combined_model.predict(X_combined)
    ss_res_combined = np.sum((Y - y_pred_combined) ** 2)
    r2_combined = 1.0 - ss_res_combined / max(ss_tot, 1e-10)

    # Step 4: Q-only model
    q_only_model = Ridge(alpha=1.0)
    q_only_model.fit(Xq, Y)
    y_pred_q = q_only_model.predict(Xq)
    ss_res_q_only = np.sum((Y - y_pred_q) ** 2)
    r2_quantum_only = 1.0 - ss_res_q_only / max(ss_tot, 1e-10)

    # Delta rho: Spearman improvement
    # For multi-output, use mean of per-output Spearman
    def _safe_spearman(a, b):
        r = spearmanr(a, b)[0]
        return 0.0 if not np.isfinite(r) else r

    if Y.shape[1] == 1:
        rho_c = _safe_spearman(y_pred_c.ravel(), Y.ravel())
        rho_combined = _safe_spearman(y_pred_combined.ravel(), Y.ravel())
    else:
        rho_c = np.mean([_safe_spearman(y_pred_c[:, j], Y[:, j]) for j in range(Y.shape[1])])
        rho_combined = np.mean([_safe_spearman(y_pred_combined[:, j], Y[:, j]) for j in range(Y.shape[1])])
    delta_rho = rho_combined - rho_c
    if not np.isfinite(delta_rho):
        delta_rho = 0.0

    # Gate: pass if r2_residual >= 0.02 or delta_rho >= 0.03
    pass_gate = (r2_residual >= 0.02) or (delta_rho >= 0.03)

    return {
        "r2_classical": float(r2_classical),
        "r2_residual": float(r2_residual),
        "delta_rho": float(delta_rho),
        "r2_combined": float(r2_combined),
        "r2_quantum_only": float(r2_quantum_only),
        "rho_classical": float(rho_c),
        "rho_combined": float(rho_combined),
        "pass_gate": bool(pass_gate),
    }


def feature_variance_check(Q: np.ndarray) -> dict:
    """Check that quantum features are non-constant and have variance.

    Returns
    -------
    dict
        "variances": per-feature variance
        "effective_rank": PCA effective rank
        "n_constant": number of constant features
    """
    variances = Q.var(axis=0)
    n_constant = int(np.sum(variances < 1e-10))

    # Effective rank via PCA
    scaler = StandardScaler().fit(Q)
    Qs = scaler.transform(Q)
    cov = np.cov(Qs.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) > 0 and eigvals.sum() > 0:
        p = eigvals / eigvals.sum()
        effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-20))))
    else:
        effective_rank = 0.0

    return {
        "variances": variances.tolist(),
        "effective_rank": effective_rank,
        "n_constant": n_constant,
        "n_features": Q.shape[1],
    }
