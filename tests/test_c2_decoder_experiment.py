import numpy as np

from experiments.tab_foundation_compare.src.c2_decoder_experiment import regression_metrics


def test_regression_metrics_are_exact_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    metrics = regression_metrics(y, y.copy())
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
