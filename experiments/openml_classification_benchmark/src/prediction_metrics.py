from __future__ import annotations

import numpy as np
from scipy.special import expit, softmax

from src.graphdrone_fit.metrics import classification_metrics


def probabilities_from_raw_prediction(
    raw_prediction: np.ndarray,
    *,
    prediction_type: str,
    n_classes: int,
) -> np.ndarray:
    raw = np.asarray(raw_prediction)
    if prediction_type == "probs":
        if raw.ndim == 1:
            if n_classes != 2:
                raise ValueError(f"1D probability predictions require n_classes=2, got {n_classes}")
            pos = raw.astype(np.float32)
            proba = np.column_stack([1.0 - pos, pos]).astype(np.float32)
        elif raw.ndim == 2:
            proba = raw.astype(np.float32)
        else:
            raise ValueError(f"Unsupported probability prediction shape {tuple(raw.shape)}")
    elif prediction_type == "logits":
        if raw.ndim == 1:
            if n_classes != 2:
                raise ValueError(f"1D logits require n_classes=2, got {n_classes}")
            pos = expit(raw).astype(np.float32)
            proba = np.column_stack([1.0 - pos, pos]).astype(np.float32)
        elif raw.ndim == 2:
            proba = softmax(raw.astype(np.float32), axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported logits prediction shape {tuple(raw.shape)}")
    elif prediction_type == "labels":
        labels = raw.reshape(-1).astype(np.int64)
        proba = np.zeros((len(labels), n_classes), dtype=np.float32)
        proba[np.arange(len(labels)), labels] = 1.0
    else:
        raise ValueError(f"Unsupported prediction_type={prediction_type!r}")
    proba = np.clip(proba, 1e-8, 1.0)
    proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-8)
    return proba.astype(np.float32)


def classification_metrics_from_raw_prediction(
    y_true: np.ndarray,
    raw_prediction: np.ndarray,
    *,
    prediction_type: str,
    class_labels: tuple[str, ...],
) -> dict[str, float]:
    proba = probabilities_from_raw_prediction(
        raw_prediction,
        prediction_type=prediction_type,
        n_classes=len(class_labels),
    )
    return classification_metrics(
        y_true,
        proba,
        class_labels=tuple(range(len(class_labels))),
    )
