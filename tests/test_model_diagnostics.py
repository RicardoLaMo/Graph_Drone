import numpy as np
import torch

from graphdrone_fit.config import GraphDroneConfig
from graphdrone_fit.expert_factory import ExpertPredictionBatch
from graphdrone_fit.model import GraphDrone
from graphdrone_fit.view_descriptor import ViewDescriptor


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


def test_regression_prediction_falls_back_to_anchor_when_training_nonfinite():
    gd = GraphDrone(GraphDroneConfig())
    gd._problem_type = "regression"
    gd._router_training_force_anchor_only = True
    gd._router_fit_diagnostics = {"validation_router_training_nonfinite_flag": 1.0}

    batch = ExpertPredictionBatch(
        expert_ids=("FULL", "SUB0"),
        descriptors=(
            ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="Full",
                is_anchor=True,
                input_dim=2,
                input_indices=(0, 1),
            ),
            ViewDescriptor(
                expert_id="SUB0",
                family="structural_subspace",
                view_name="Sub",
                input_dim=1,
                input_indices=(0,),
            ),
        ),
        predictions=np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        full_expert_id="FULL",
        full_index=0,
        quality_scores=None,
    )

    preds, diagnostics = gd._regression_predictions(np.zeros((2, 2), dtype=np.float32), batch)
    assert np.allclose(preds, np.array([1.0, 2.0], dtype=np.float32))
    assert diagnostics["router_kind"] == "router_training_nonfinite_anchor_only"
    assert diagnostics["router_nonfinite_fallback"] is True
    assert diagnostics["effective_defer_rate"] == 0.0
    assert diagnostics["validation_router_training_nonfinite_flag"] == 1.0
