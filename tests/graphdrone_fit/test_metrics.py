from __future__ import annotations

import numpy as np

from src.graphdrone_fit.metrics import classification_metrics, regression_metrics


def test_regression_metrics_keys() -> None:
    y_true = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    y_pred = np.array([0.1, 0.9, 2.2], dtype=np.float32)
    metrics = regression_metrics(y_true, y_pred)
    assert set(metrics) == {"rmse", "mae", "r2"}


def test_classification_metrics_binary() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 1, 1], dtype=np.int64)
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    metrics = classification_metrics(y_true, y_pred, y_proba, class_labels=(0, 1))
    assert {"accuracy", "f1_macro", "f1_weighted", "log_loss", "roc_auc", "pr_auc", "primary_metric"} <= set(metrics)
    assert metrics["accuracy"] == 1.0


def test_classification_metrics_multiclass() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 0, 1, 1], dtype=np.int64)
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.5, 0.3],
        ],
        dtype=np.float32,
    )
    metrics = classification_metrics(y_true, y_pred, y_proba, class_labels=(0, 1, 2))
    assert {"accuracy", "f1_macro", "log_loss", "roc_auc_ovr_macro", "pr_auc_ovr_macro", "primary_metric"} <= set(metrics)
    assert 0.0 <= metrics["roc_auc_ovr_macro"] <= 1.0
