"""Distillation trainer for quantum micro-teacher.

Trains a micro-student (12 → h → 5) with optional quantum distillation:
  L = L_task + λ_q * L_distill

  L_task   = MSE(student(x_c), y)
  L_distill = MSE(student(x_c), teacher(q))

The teacher is a Ridge regression from quantum features Q → Y.
At inference, the student only uses classical descriptors x_c.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainResult:
    """Result of training one micro-student configuration."""
    train_r2: float
    val_r2: float
    train_loss: float
    val_loss: float
    n_params: int
    lambda_q: float
    hidden: int
    condition: str
    val_pred: np.ndarray = None


def fit_teacher(Q: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> tuple:
    """Fit a Ridge teacher from quantum features to targets.

    Returns
    -------
    (scaler, model) : tuple
        Fitted StandardScaler and Ridge model.
    """
    scaler = StandardScaler().fit(Q)
    Qs = scaler.transform(Q)
    model = Ridge(alpha=alpha)
    model.fit(Qs, Y)
    return scaler, model


def predict_teacher(scaler, model, Q: np.ndarray) -> np.ndarray:
    """Predict using fitted teacher."""
    return model.predict(scaler.transform(Q))


def train_micro_student(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    hidden: int = 4,
    lambda_q: float = 0.0,
    teacher_pred_train: Optional[np.ndarray] = None,
    teacher_pred_val: Optional[np.ndarray] = None,
    input_dim: int = 12,
    task_dim: int = 5,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    condition: str = "",
) -> TrainResult:
    """Train a micro-student with optional distillation.

    Parameters
    ----------
    X_train, Y_train : training data (classical descriptors, targets)
    X_val, Y_val : validation data
    hidden : hidden layer width
    lambda_q : distillation weight (0 = no distillation)
    teacher_pred_train : teacher predictions on training data (for distillation)
    teacher_pred_val : teacher predictions on val data (for evaluation)
    input_dim : number of input features
    task_dim : number of output targets
    epochs, lr, weight_decay : training hyperparameters
    seed : random seed
    condition : name of the experimental condition

    Returns
    -------
    TrainResult
    """
    from ..classical.micro_models import MicroStudent, count_params

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize inputs
    scaler_x = StandardScaler().fit(X_train)
    Xtr = torch.tensor(scaler_x.transform(X_train), dtype=torch.float32)
    Xva = torch.tensor(scaler_x.transform(X_val), dtype=torch.float32)

    # Standardize targets
    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0) + 1e-8
    Ytr = torch.tensor((Y_train - y_mean) / y_std, dtype=torch.float32)
    Yva = torch.tensor((Y_val - y_mean) / y_std, dtype=torch.float32)

    # Teacher predictions (standardized)
    if teacher_pred_train is not None and lambda_q > 0:
        Ttr = torch.tensor((teacher_pred_train - y_mean) / y_std, dtype=torch.float32)
    else:
        Ttr = None
        lambda_q = 0.0

    # Build model
    model = MicroStudent(input_dim=input_dim, hidden=hidden, task_dim=task_dim)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    best_val_loss = float("inf")
    best_val_pred = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred, _aux = model(Xtr)
        loss_task = nn.MSELoss()(pred, Ytr)

        if Ttr is not None and lambda_q > 0:
            loss_distill = nn.MSELoss()(pred, Ttr)
            loss = loss_task + lambda_q * loss_distill
        else:
            loss = loss_task

        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_pred, _aux = model(Xva)
                val_loss = nn.MSELoss()(val_pred, Yva).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_pred = val_pred.numpy()

    # Compute R^2
    train_pred, _ = model(Xtr)
    train_pred = train_pred.detach().numpy()
    train_r2 = _r2_score(Ytr.numpy(), train_pred)
    val_r2 = _r2_score(Yva.numpy(), best_val_pred)

    # Unstandardize val predictions for downstream use
    val_pred_unstd = best_val_pred * y_std + y_mean

    return TrainResult(
        train_r2=train_r2,
        val_r2=val_r2,
        train_loss=loss.item(),
        val_loss=best_val_loss,
        n_params=n_params,
        lambda_q=lambda_q,
        hidden=hidden,
        condition=condition,
        val_pred=val_pred_unstd,
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 (coefficient of determination)."""
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-10))


def run_condition(
    condition: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    Q_train: np.ndarray = None,
    Q_val: np.ndarray = None,
    hidden: int = 4,
    lambda_q: float = 0.3,
    epochs: int = 300,
    seed: int = 42,
    task_dim: int = 5,
) -> TrainResult:
    """Run one experimental condition.

    Conditions:
    - M0: Classical baseline (no quantum)
    - M1: Quantum distillation (spectral Q → teacher → student)
    - M2: Shuffled control (shuffled Q)
    - M3: Random compression control (random Q → teacher → student)
    - M4: Classical compressed teacher (compressed classical features → teacher)
    - M5: Direct quantum features (Q as input, no distillation)
    - M6: Oracle (large model on C+Q)
    """
    if condition == "M0":
        # Classical baseline: no distillation
        return train_micro_student(
            X_train, Y_train, X_val, Y_val,
            hidden=hidden, lambda_q=0.0,
            input_dim=X_train.shape[1], task_dim=task_dim,
            epochs=epochs, seed=seed, condition=condition,
        )

    elif condition == "M1":
        # Quantum distillation: teacher(Q) → student(X_c)
        scaler_t, model_t = fit_teacher(Q_train, Y_train)
        teacher_train = predict_teacher(scaler_t, model_t, Q_train)
        teacher_val = predict_teacher(scaler_t, model_t, Q_val)
        return train_micro_student(
            X_train, Y_train, X_val, Y_val,
            hidden=hidden, lambda_q=lambda_q,
            teacher_pred_train=teacher_train,
            teacher_pred_val=teacher_val,
            input_dim=X_train.shape[1], task_dim=task_dim,
            epochs=epochs, seed=seed, condition=condition,
        )

    elif condition == "M2":
        # Shuffled control: shuffle Q across graphs
        rng = np.random.default_rng(seed)
        perm = rng.permutation(Q_train.shape[0])
        Q_shuffled_train = Q_train[perm]
        Q_shuffled_val = Q_val[rng.permutation(Q_val.shape[0])]
        scaler_t, model_t = fit_teacher(Q_shuffled_train, Y_train)
        teacher_train = predict_teacher(scaler_t, model_t, Q_shuffled_train)
        teacher_val = predict_teacher(scaler_t, model_t, Q_shuffled_val)
        return train_micro_student(
            X_train, Y_train, X_val, Y_val,
            hidden=hidden, lambda_q=lambda_q,
            teacher_pred_train=teacher_train,
            teacher_pred_val=teacher_val,
            input_dim=X_train.shape[1], task_dim=task_dim,
            epochs=epochs, seed=seed, condition=condition,
        )

    elif condition == "M3":
        # Random compression control: same as M1 but with random Q
        # (Q_random passed in Q_train/Q_val)
        scaler_t, model_t = fit_teacher(Q_train, Y_train)
        teacher_train = predict_teacher(scaler_t, model_t, Q_train)
        teacher_val = predict_teacher(scaler_t, model_t, Q_val)
        return train_micro_student(
            X_train, Y_train, X_val, Y_val,
            hidden=hidden, lambda_q=lambda_q,
            teacher_pred_train=teacher_train,
            teacher_pred_val=teacher_val,
            input_dim=X_train.shape[1], task_dim=task_dim,
            epochs=epochs, seed=seed, condition=condition,
        )

    elif condition == "M4":
        # Classical compressed teacher: compressed classical features → teacher
        # (compressed features passed in Q_train/Q_val)
        scaler_t, model_t = fit_teacher(Q_train, Y_train)
        teacher_train = predict_teacher(scaler_t, model_t, Q_train)
        teacher_val = predict_teacher(scaler_t, model_t, Q_val)
        return train_micro_student(
            X_train, Y_train, X_val, Y_val,
            hidden=hidden, lambda_q=lambda_q,
            teacher_pred_train=teacher_train,
            teacher_pred_val=teacher_val,
            input_dim=X_train.shape[1], task_dim=task_dim,
            epochs=epochs, seed=seed, condition=condition,
        )

    elif condition == "M5":
        # Direct quantum features: Q as input to micro-student
        return train_micro_student(
            Q_train, Y_train, Q_val, Y_val,
            hidden=hidden, lambda_q=0.0,
            input_dim=Q_train.shape[1], task_dim=task_dim,
            epochs=epochs, seed=seed, condition=condition,
        )

    elif condition == "M6":
        # Oracle: large model on (C + Q)
        from ..diagnostics.oracle_models import OracleMLP
        X_combined_train = np.hstack([X_train, Q_train])
        X_combined_val = np.hstack([X_val, Q_val])
        oracle = OracleMLP(X_combined_train.shape[1], Y_train.shape[1], seed=seed)
        oracle.fit(X_combined_train, Y_train, epochs=epochs)
        r2_train = oracle.score_r2(X_combined_train, Y_train)
        r2_val = oracle.score_r2(X_combined_val, Y_val)
        return TrainResult(
            train_r2=r2_train, val_r2=r2_val,
            train_loss=0.0, val_loss=0.0,
            n_params=-1, lambda_q=0.0, hidden=128,
            condition=condition,
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")
