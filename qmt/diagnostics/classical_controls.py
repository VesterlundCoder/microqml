"""Classical falsification controls: matched-dimension classical feature transforms.

Controls:
  A: Gaussian random projection of C
  B: Random Fourier Features (RFF) from C
  C: Random classical reservoir (fixed random network)
  D: Trainable nonlinear baselines (RBF kernel, MLP, XGBoost, RF)

All controls produce features with the SAME dimension as Q (default 8).
"""
from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def control_a_gaussian_projection(
    C: np.ndarray, n_features: int = 8, seed: int = 42,
) -> np.ndarray:
    """Control A: Gaussian random projection of C.

    R_{ij} ~ N(0, sigma^2), output = R @ C
    Tests: does the gain come from just adding a random projection?
    """
    rng = np.random.default_rng(seed)
    n, d = C.shape
    R = rng.standard_normal((d, n_features)) / np.sqrt(d)
    features = C @ R
    return features.astype(np.float32)


def control_b_rff(
    C: np.ndarray, n_features: int = 8, seed: int = 42, gamma: float = 1.0,
) -> np.ndarray:
    """Control B: Random Fourier Features from C.

    z(x) = sqrt(2/D) * cos(Wx + b)
    Tests: does the gain come from nonlinear feature expansion?
    """
    scaler = StandardScaler().fit(C)
    Cs = scaler.transform(C)
    sampler = RBFSampler(gamma=gamma, n_components=n_features, random_state=seed)
    features = sampler.fit_transform(Cs)
    return features.astype(np.float32)


def control_c_random_reservoir(
    C: np.ndarray, n_features: int = 8, seed: int = 42, activation: str = "tanh",
) -> np.ndarray:
    """Control C: Random classical reservoir (fixed random network).

    h = sigma(W @ C + b), output = W_out @ h
    W and b are fixed (not trained). Matches parameter budget of quantum circuit.
    """
    rng = np.random.default_rng(seed)
    n, d = C.shape

    # Input scaling
    scaler = StandardScaler().fit(C)
    Cs = scaler.transform(C)

    # Random hidden layer (matched to quantum circuit depth)
    W = rng.standard_normal((d, n_features)) / np.sqrt(d)
    b = rng.standard_normal(n_features) * 0.1

    h = Cs @ W + b
    if activation == "tanh":
        h = np.tanh(h)
    elif activation == "relu":
        h = np.maximum(h, 0)
    elif activation == "sin":
        h = np.sin(h)

    return h.astype(np.float32)


def control_d_trainable(
    C: np.ndarray,
    Y: np.ndarray,
    C_val: np.ndarray,
    Y_val: np.ndarray,
    method: str = "rbf",
    n_features: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Control D: Trainable nonlinear baseline.

    Returns
    -------
    (train_pred, val_pred, info)
    """
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    scaler = StandardScaler().fit(C)
    Xtr = scaler.transform(C)
    Xva = scaler.transform(C_val)

    info = {"method": method}

    if method == "rbf":
        if Y.ndim == 1:
            model = SVR(kernel="rbf", C=1.0, gamma="scale")
            model.fit(Xtr, Y)
            train_pred = model.predict(Xtr).reshape(-1, 1)
            val_pred = model.predict(Xva).reshape(-1, 1)
        else:
            model = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, gamma="scale"))
            model.fit(Xtr, Y)
            train_pred = model.predict(Xtr)
            val_pred = model.predict(Xva)

    elif method == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=seed, early_stopping=True,
        )
        model.fit(Xtr, Y)
        train_pred = model.predict(Xtr)
        val_pred = model.predict(Xva)

    elif method == "xgboost":
        from xgboost import XGBRegressor
        if Y.ndim == 1:
            model = XGBRegressor(n_estimators=100, max_depth=4, random_state=seed)
            model.fit(Xtr, Y)
            train_pred = model.predict(Xtr).reshape(-1, 1)
            val_pred = model.predict(Xva).reshape(-1, 1)
        else:
            models = []
            train_preds = []
            val_preds = []
            for j in range(Y.shape[1]):
                m = XGBRegressor(n_estimators=100, max_depth=4, random_state=seed + j)
                m.fit(Xtr, Y[:, j])
                models.append(m)
                train_preds.append(m.predict(Xtr))
                val_preds.append(m.predict(Xva))
            train_pred = np.stack(train_preds, axis=1)
            val_pred = np.stack(val_preds, axis=1)

    elif method == "rf":
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=seed,
        )
        model.fit(Xtr, Y)
        train_pred = model.predict(Xtr)
        val_pred = model.predict(Xva)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute R²
    ss_tot_val = np.sum((Y_val - Y_val.mean(axis=0)) ** 2)
    ss_res_val = np.sum((Y_val - val_pred) ** 2)
    info["r2_val"] = float(1.0 - ss_res_val / max(ss_tot_val, 1e-10))

    return train_pred, val_pred, info


def compute_delta_r2(
    X_classical: np.ndarray,
    extra_features: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Compute ΔR² = R²(C+extra) - R²(C) using proper CV.

    All transformations (scaling, etc.) are fitted inside each fold.
    """
    from sklearn.model_selection import KFold

    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    X_classical = np.nan_to_num(np.asarray(X_classical, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    extra_features = np.nan_to_num(np.asarray(extra_features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r2_c_folds = []
    r2_cq_folds = []

    for tr_idx, va_idx in kf.split(X_classical):
        Xc_tr, Xc_va = X_classical[tr_idx], X_classical[va_idx]
        Xe_tr, Xe_va = extra_features[tr_idx], extra_features[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        # Classical only
        sc_c = StandardScaler().fit(Xc_tr)
        Xc_tr_s = sc_c.transform(Xc_tr)
        Xc_va_s = sc_c.transform(Xc_va)

        m_c = Ridge(alpha=1.0)
        m_c.fit(Xc_tr_s, Y_tr)
        pred_c = m_c.predict(Xc_va_s)
        ss_tot = np.sum((Y_va - Y_va.mean(axis=0)) ** 2)
        ss_res_c = np.sum((Y_va - pred_c) ** 2)
        r2_c = 1.0 - ss_res_c / max(ss_tot, 1e-10)
        r2_c_folds.append(r2_c)

        # Combined
        X_tr_combined = np.hstack([Xc_tr_s, StandardScaler().fit(Xe_tr).transform(Xe_tr)])
        X_va_combined = np.hstack([Xc_va_s, StandardScaler().fit(Xe_tr).transform(Xe_va)])

        m_cq = Ridge(alpha=1.0)
        m_cq.fit(X_tr_combined, Y_tr)
        pred_cq = m_cq.predict(X_va_combined)
        ss_res_cq = np.sum((Y_va - pred_cq) ** 2)
        r2_cq = 1.0 - ss_res_cq / max(ss_tot, 1e-10)
        r2_cq_folds.append(r2_cq)

    delta_r2 = np.array(r2_cq_folds) - np.array(r2_c_folds)

    return {
        "r2_c_mean": float(np.mean(r2_c_folds)),
        "r2_c_std": float(np.std(r2_c_folds)),
        "r2_cq_mean": float(np.mean(r2_cq_folds)),
        "r2_cq_std": float(np.std(r2_cq_folds)),
        "delta_r2_mean": float(np.mean(delta_r2)),
        "delta_r2_std": float(np.std(delta_r2)),
        "delta_r2_per_fold": delta_r2.tolist(),
        "r2_c_per_fold": [float(x) for x in r2_c_folds],
        "r2_cq_per_fold": [float(x) for x in r2_cq_folds],
    }
