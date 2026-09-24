"""Representation geometry analysis.

Computes:
  - Singular value spectrum
  - Effective rank (entropic)
  - Participation ratio
  - CKA (Centered Kernel Alignment) between representations
  - Principal angles between subspaces
  - Kernel alignment (K_Q vs K_Y)
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def singular_value_spectrum(X: np.ndarray) -> dict:
    """Compute singular value spectrum of a feature matrix."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    sv = np.linalg.svd(Xs, compute_uv=False)
    sv = np.maximum(sv, 0)
    return {
        "singular_values": sv.tolist(),
        "n_nonzero": int(np.sum(sv > 1e-10)),
        "condition_number": float(sv[0] / max(sv[-1], 1e-20)) if len(sv) > 0 else float("inf"),
    }


def effective_rank(X: np.ndarray) -> float:
    """Entropic effective rank: r_eff = exp(-sum(p_i * log(p_i))).

    where p_i = sigma_i / sum(sigma_j).
    """
    spec = singular_value_spectrum(X)
    sv = np.array(spec["singular_values"])
    sv = sv[sv > 1e-10]
    if len(sv) == 0 or sv.sum() == 0:
        return 0.0
    p = sv / sv.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-20))))


def participation_ratio(X: np.ndarray) -> float:
    """Participation ratio: PR = (sum(sigma_i))^2 / sum(sigma_i^2)."""
    spec = singular_value_spectrum(X)
    sv = np.array(spec["singular_values"])
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0
    return float(sv.sum() ** 2 / np.sum(sv ** 2))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Centered Kernel Alignment) between two representations.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Measures similarity of representations regardless of basis.
    """
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(np.asarray(Y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    KX = X @ X.T
    KY = Y @ Y.T

    # HSIC
    hsic = np.sum(KX * KY)

    # Norms
    norm_x = np.sqrt(np.sum(KX * KX))
    norm_y = np.sqrt(np.sum(KY * KY))

    if norm_x < 1e-20 or norm_y < 1e-20:
        return 0.0

    return float(hsic / (norm_x * norm_y))


def rbf_cka(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """RBF kernel CKA between two representations."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(np.asarray(Y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.metrics.pairwise import rbf_kernel
    if gamma is None:
        # Median heuristic
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X)
        gamma = 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2 + 1e-20)

    KX = rbf_kernel(X, gamma=gamma)
    KY = rbf_kernel(Y, gamma=gamma)

    # Center kernels
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KX = H @ KX @ H
    KY = H @ KY @ H

    hsic = np.sum(KX * KY)
    norm_x = np.sqrt(np.sum(KX * KX))
    norm_y = np.sqrt(np.sum(KY * KY))

    if norm_x < 1e-20 or norm_y < 1e-20:
        return 0.0

    return float(hsic / (norm_x * norm_y))


def principal_angles(X: np.ndarray, Y: np.ndarray) -> list[float]:
    """Principal angles between subspaces spanned by X and Y columns.

    Returns angles in degrees, sorted from smallest to largest.
    """
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(np.asarray(Y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    # Orthonormalize via QR
    Qx, _ = np.linalg.qr(X - X.mean(axis=0))
    Qy, _ = np.linalg.qr(Y - Y.mean(axis=0))

    # SVD of Qx^T Qy gives cosines of principal angles
    U, s, Vt = np.linalg.svd(Qx.T @ Qy)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s) * 180.0 / np.pi

    return angles.tolist()


def kernel_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """Kernel alignment between feature kernel K_X = X X^T and target kernel K_Y = Y Y^T.

    A(X, Y) = <K_X, K_Y> / (||K_X|| * ||K_Y||)
    """
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(np.asarray(Y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    KX = X @ X.T
    KY = Y @ Y.T

    # Center
    KX = KX - KX.mean()
    KY = KY - KY.mean()

    num = np.sum(KX * KY)
    den = np.sqrt(np.sum(KX * KX) * np.sum(KY * KY))

    if den < 1e-20:
        return 0.0

    return float(num / den)


def full_geometry_analysis(
    C: np.ndarray, Q: np.ndarray, Y: np.ndarray,
) -> dict:
    """Complete representation geometry analysis.

    Parameters
    ----------
    C : classical features (N, d_c)
    Q : quantum features (N, d_q)
    Y : targets (N, d_y)

    Returns
    -------
    dict with all geometry metrics
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    return {
        "classical": {
            "effective_rank": effective_rank(C),
            "participation_ratio": participation_ratio(C),
            "singular_values": singular_value_spectrum(C)["singular_values"],
        },
        "quantum": {
            "effective_rank": effective_rank(Q),
            "participation_ratio": participation_ratio(Q),
            "singular_values": singular_value_spectrum(Q)["singular_values"],
        },
        "cka_linear_CQ": linear_cka(C, Q),
        "cka_rbf_CQ": rbf_cka(C, Q),
        "cka_linear_QY": linear_cka(Q, Y),
        "cka_linear_CY": linear_cka(C, Y),
        "kernel_alignment_QY": kernel_alignment(Q, Y),
        "kernel_alignment_CY": kernel_alignment(C, Y),
        "principal_angles_CQ": principal_angles(C, Q),
    }
