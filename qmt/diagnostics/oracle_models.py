"""Oracle diagnostic models: oversized models to test if signal exists at all.

If a 24->128->128->Y model cannot extract task information from quantum features,
the micro-model is not the bottleneck.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class OracleMLP:
    """Large diagnostic MLP: input_dim -> 128 -> 128 -> output_dim."""

    def __init__(self, input_dim: int, output_dim: int = 1, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = None
        self._build()

    def _build(self):
        torch.manual_seed(self.seed)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def fit(self, X, Y, epochs=300, batch_size=64):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Xs = self.scaler.fit_transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)

        n = len(Xt)
        for epoch in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                self.opt.zero_grad()
                pred = self.model(Xt[idx])
                loss = nn.MSELoss()(pred, Yt[idx])
                loss.backward()
                self.opt.step()

    def predict(self, X):
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        with torch.no_grad():
            pred = self.model(torch.tensor(Xs, dtype=torch.float32))
        return pred.numpy()

    def score_r2(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        pred = self.predict(X)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        ss_res = np.sum((Y - pred) ** 2)
        return float(1.0 - ss_res / max(ss_tot, 1e-10))


def oracle_test(
    X_classical: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run oracle diagnostic: can a large model extract signal from Q?

    Tests:
    - Q -> Y (quantum only)
    - (C, Q) -> Y (combined)
    - C -> Y (classical only, for comparison)

    Returns
    -------
    dict
        R^2 for each setting.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    output_dim = Y.shape[1]

    # Q -> Y
    oracle_q = OracleMLP(Q.shape[1], output_dim, seed=seed)
    oracle_q.fit(Q, Y, epochs=300)
    r2_q = oracle_q.score_r2(Q, Y)

    # C -> Y
    oracle_c = OracleMLP(X_classical.shape[1], output_dim, seed=seed)
    oracle_c.fit(X_classical, Y, epochs=300)
    r2_c = oracle_c.score_r2(X_classical, Y)

    # (C, Q) -> Y
    X_combined = np.hstack([X_classical, Q])
    oracle_cq = OracleMLP(X_combined.shape[1], output_dim, seed=seed)
    oracle_cq.fit(X_combined, Y, epochs=300)
    r2_cq = oracle_cq.score_r2(X_combined, Y)

    # Ridge baselines for comparison
    ridge_q = Ridge(alpha=1.0).fit(Q, Y)
    r2_ridge_q = float(ridge_q.score(Q, Y))

    ridge_c = Ridge(alpha=1.0).fit(X_classical, Y)
    r2_ridge_c = float(ridge_c.score(X_classical, Y))

    ridge_cq = Ridge(alpha=1.0).fit(X_combined, Y)
    r2_ridge_cq = float(ridge_cq.score(X_combined, Y))

    return {
        "oracle_r2_q": r2_q,
        "oracle_r2_c": r2_c,
        "oracle_r2_cq": r2_cq,
        "ridge_r2_q": r2_ridge_q,
        "ridge_r2_c": r2_ridge_c,
        "ridge_r2_cq": r2_ridge_cq,
        "incremental_oracle": r2_cq - r2_c,
        "incremental_ridge": r2_ridge_cq - r2_ridge_c,
    }
