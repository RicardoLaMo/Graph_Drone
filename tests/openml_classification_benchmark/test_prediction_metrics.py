from __future__ import annotations

import numpy as np

from experiments.openml_classification_benchmark.src.prediction_metrics import (
    classification_metrics_from_raw_prediction,
    probabilities_from_raw_prediction,
)


def test_probabilities_from_raw_prediction_handles_binary_logits() -> None:
    logits = np.array([0.0, 2.0, -2.0], dtype=np.float32)
    proba = probabilities_from_raw_prediction(logits, prediction_type="logits", n_classes=2)
    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classification_metrics_from_raw_prediction_handles_multiclass_probs() -> None:
    y_true = np.array([0, 1, 2], dtype=np.int64)
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )
    metrics = classification_metrics_from_raw_prediction(
        y_true,
        probs,
        prediction_type="probs",
        class_labels=("a", "b", "c"),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
