import numpy as np
import torch

from graphdrone_fit.config import GraphDroneConfig, LegitimacyGateConfig
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
    assert diagnostics["validation_positive_specialist_opportunity_score"] > 0.99
    assert np.isclose(diagnostics["validation_residual_usefulness_gap"], 0.1499999, atol=1e-6)
    assert np.isclose(diagnostics["validation_positive_specialist_mass"], 0.8819444, atol=1e-6)
    assert diagnostics["validation_top_specialist_positive_rate"] == 1.0


def test_regression_residual_usefulness_gap_is_positive_when_router_misses_available_gain():
    expert_predictions = torch.tensor(
        [
            [10.0, 9.0, 20.0],
            [5.0, 4.0, 20.0],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor([9.0, 4.0], dtype=torch.float32)
    specialist_weights = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    defer_prob = torch.full((2, 1), 0.05, dtype=torch.float32)

    stats = GraphDrone._regression_residual_usefulness_tensors(
        expert_predictions=expert_predictions,
        y_true=y_true,
        specialist_weights=specialist_weights,
        defer_prob=defer_prob,
        full_index=0,
    )

    active = stats["active_mask"]
    assert bool(active.all().item()) is True
    assert float(stats["best_advantage"][active].mean().item()) > 0.99
    assert float(stats["realized_advantage"][active].mean().item()) < 0.1
    assert float(stats["residual_usefulness_gap"][active].mean().item()) > 0.9


def test_regression_allocation_usefulness_prefers_positive_mass_on_helpful_specialists():
    expert_predictions = torch.tensor(
        [
            [10.0, 9.0, 20.0],
            [5.0, 4.0, 20.0],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor([9.0, 4.0], dtype=torch.float32)
    defer_prob = torch.ones((2, 1), dtype=torch.float32)

    good_weights = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    bad_weights = torch.tensor(
        [
            [0.1, 0.0, 0.9],
            [0.1, 0.0, 0.9],
        ],
        dtype=torch.float32,
    )

    good_stats = GraphDrone._regression_residual_usefulness_tensors(
        expert_predictions=expert_predictions,
        y_true=y_true,
        specialist_weights=good_weights,
        defer_prob=defer_prob,
        full_index=0,
    )
    bad_stats = GraphDrone._regression_residual_usefulness_tensors(
        expert_predictions=expert_predictions,
        y_true=y_true,
        specialist_weights=bad_weights,
        defer_prob=defer_prob,
        full_index=0,
    )

    good_score = GraphDrone._regression_allocation_usefulness_from_stats(good_stats)
    bad_score = GraphDrone._regression_allocation_usefulness_from_stats(bad_stats)
    assert float(good_score.item()) > float(bad_score.item())


def test_regression_conservative_allocation_penalty_prefers_positive_mass_on_confident_rows():
    expert_predictions = torch.tensor(
        [
            [10.0, 9.0, 20.0],
            [5.0, 4.0, 20.0],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor([9.0, 4.0], dtype=torch.float32)
    defer_prob = torch.ones((2, 1), dtype=torch.float32)

    good_weights = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    bad_weights = torch.tensor(
        [
            [0.1, 0.0, 0.9],
            [0.1, 0.0, 0.9],
        ],
        dtype=torch.float32,
    )

    good_stats = GraphDrone._regression_residual_usefulness_tensors(
        expert_predictions=expert_predictions,
        y_true=y_true,
        specialist_weights=good_weights,
        defer_prob=defer_prob,
        full_index=0,
    )
    bad_stats = GraphDrone._regression_residual_usefulness_tensors(
        expert_predictions=expert_predictions,
        y_true=y_true,
        specialist_weights=bad_weights,
        defer_prob=defer_prob,
        full_index=0,
    )

    good_penalty = GraphDrone._regression_conservative_allocation_penalty_from_stats(good_stats)
    bad_penalty = GraphDrone._regression_conservative_allocation_penalty_from_stats(bad_stats)
    assert float(good_penalty.item()) < float(bad_penalty.item())


def test_regression_robust_allocation_usefulness_prefers_consistent_positive_mass():
    expert_predictions = torch.tensor(
        [
            [10.0, 9.0, 20.0],
            [5.0, 4.0, 20.0],
            [8.0, 7.0, 20.0],
            [6.0, 5.0, 20.0],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor([9.0, 4.0, 7.0, 5.0], dtype=torch.float32)
    defer_prob = torch.ones((4, 1), dtype=torch.float32)

    consistent_weights = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    inconsistent_weights = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.1, 0.0, 0.9],
            [0.1, 0.9, 0.0],
            [0.1, 0.0, 0.9],
        ],
        dtype=torch.float32,
    )

    consistent_stats = GraphDrone._regression_residual_usefulness_tensors(
        expert_predictions=expert_predictions,
        y_true=y_true,
        specialist_weights=consistent_weights,
        defer_prob=defer_prob,
        full_index=0,
    )
    inconsistent_stats = GraphDrone._regression_residual_usefulness_tensors(
        expert_predictions=expert_predictions,
        y_true=y_true,
        specialist_weights=inconsistent_weights,
        defer_prob=defer_prob,
        full_index=0,
    )

    consistent_score = GraphDrone._regression_robust_allocation_usefulness_from_stats(consistent_stats)
    inconsistent_score = GraphDrone._regression_robust_allocation_usefulness_from_stats(inconsistent_stats)
    assert float(consistent_score.item()) > float(inconsistent_score.item())


def test_regression_prediction_falls_back_to_anchor_when_training_nonfinite():
    gd = GraphDrone(GraphDroneConfig())
    gd._problem_type = "regression"
    gd._router_training_force_anchor_only = True
    gd._router_fit_diagnostics = {
        "validation_router_training_nonfinite_flag": 1.0,
        "regression_router_fallback_stage": "train_loss",
        "regression_router_fallback_reason": "nonfinite_loss",
    }

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
    assert diagnostics["regression_router_fallback_stage"] == "train_loss"
    assert diagnostics["regression_router_fallback_reason"] == "nonfinite_loss"


def test_regression_legitimacy_early_exit_preserves_router_fit_diagnostics():
    gd = GraphDrone(
        GraphDroneConfig(
            legitimacy_gate=LegitimacyGateConfig(
                enabled=True,
                regression_enabled=True,
                binary_enabled=False,
                multiclass_enabled=False,
                regression_variance_threshold=1.0,
            )
        )
    )
    gd._problem_type = "regression"
    gd._router_fit_diagnostics = {
        "validation_weighted_specialist_advantage_score": 0.25,
        "validation_allocation_usefulness_score": 0.5,
        "validation_robust_allocation_usefulness_score": 0.4,
        "validation_conservative_allocation_penalty": 0.125,
    }

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
        quality_scores=np.zeros((2, 2, 1), dtype=np.float32),
    )

    preds, diagnostics = gd._regression_predictions(np.zeros((2, 2), dtype=np.float32), batch)
    assert np.allclose(preds, np.array([1.0, 2.0], dtype=np.float32))
    assert diagnostics["router_kind"] == "legitimacy_gate_anchor_only"
    assert diagnostics["validation_weighted_specialist_advantage_score"] == 0.25
    assert diagnostics["validation_allocation_usefulness_score"] == 0.5
    assert diagnostics["validation_robust_allocation_usefulness_score"] == 0.4
    assert diagnostics["validation_conservative_allocation_penalty"] == 0.125
