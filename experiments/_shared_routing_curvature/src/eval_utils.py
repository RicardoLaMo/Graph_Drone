"""
eval_utils.py — Shared evaluation helpers for both datasets.
"""

import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, log_loss)


def eval_regression(name, y_true, y_pred):
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"  [{name}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def eval_classification(name, y_true, y_pred, y_proba=None):
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ll = float(log_loss(y_true, y_proba)) if y_proba is not None else float("nan")
    print(f"  [{name}]  Acc={acc:.4f}  F1={f1:.4f}  LogLoss={ll:.4f}")
    return {"model": name, "accuracy": acc, "macro_f1": f1, "log_loss": ll}
