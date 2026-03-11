from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred_or_proba: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    class_labels: tuple[int, ...],
) -> dict[str, float]:
    labels = np.asarray(class_labels, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if y_proba is None:
        y_proba = np.asarray(y_pred_or_proba, dtype=np.float32)
        if y_proba.ndim != 2 or y_proba.shape[0] != y_true.shape[0] or y_proba.shape[1] != len(labels):
            raise ValueError(
                f"Expected y_proba shape [{y_true.shape[0]}, {len(labels)}], got {tuple(y_proba.shape)}"
            )
        y_pred = labels[np.argmax(y_proba, axis=1)]
    else:
        y_pred = np.asarray(y_pred_or_proba, dtype=np.int64).reshape(-1)
        y_proba = np.asarray(y_proba, dtype=np.float32)
    if y_proba.ndim != 2 or y_proba.shape[0] != y_true.shape[0] or y_proba.shape[1] != len(labels):
        raise ValueError(
            f"Expected y_proba shape [{y_true.shape[0]}, {len(labels)}], got {tuple(y_proba.shape)}"
        )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "log_loss": float(log_loss(y_true, y_proba, labels=list(labels))),
    }

    if len(labels) == 2:
        positive_index = 1
        positive_score = y_proba[:, positive_index]
        positive_true = (y_true == labels[positive_index]).astype(np.int64)
        metrics["roc_auc"] = float(roc_auc_score(positive_true, positive_score))
        metrics["pr_auc"] = float(average_precision_score(positive_true, positive_score))
        metrics["primary_metric"] = metrics["roc_auc"]
        return metrics

    y_true_bin = label_binarize(y_true, classes=list(labels))
    metrics["roc_auc"] = float(
        roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    )
    metrics["pr_auc"] = float(
        average_precision_score(y_true_bin, y_proba, average="macro")
    )
    metrics["roc_auc_ovr_macro"] = metrics["roc_auc"]
    metrics["pr_auc_ovr_macro"] = metrics["pr_auc"]
    metrics["primary_metric"] = metrics["macro_f1"]
    return metrics
