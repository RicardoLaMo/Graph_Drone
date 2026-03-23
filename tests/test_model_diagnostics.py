import numpy as np
import torch

from graphdrone_fit.model import GraphDrone


def test_regression_residual_usefulness_diagnostics_reports_positive_mass():
    expert_predictions = torch.tensor(
        [
            [10.0, 9.0, 11.0],
            [0.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor([9.0, 2.0], dtype=torch.float32)
    specialist_weights = torch.tensor(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=torch.float32,
    )
    defer_prob = torch.ones((2, 1), dtype=torch.float32)

    diagnostics = GraphDrone._regression_residual_usefulness_diagnostics(
        expert_predictions=expert_predictions,
        y_true=y_true,
        specialist_weights=specialist_weights,
        defer_prob=defer_prob,
        full_index=0,
    )

    assert diagnostics["validation_best_specialist_advantage_score"] > 0.99
    assert diagnostics["validation_weighted_specialist_advantage_score"] > 0.6
    assert diagnostics["validation_defer_weighted_specialist_advantage_score"] > 0.6
    assert diagnostics["validation_top_specialist_advantage_score"] > 0.99
    assert np.isclose(diagnostics["validation_positive_specialist_mass"], 0.8819444, atol=1e-6)
    assert diagnostics["validation_top_specialist_positive_rate"] == 1.0
