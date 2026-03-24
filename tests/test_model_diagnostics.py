import numpy as np
import torch
from types import SimpleNamespace

from graphdrone_fit.config import GraphDroneConfig, LegitimacyGateConfig, SetRouterConfig
from graphdrone_fit.expert_factory import ExpertPredictionBatch
from graphdrone_fit.model import GraphDrone
from graphdrone_fit.set_router import RouterOutputs
from graphdrone_fit.set_router import ContextualTransformerRouter, TaskConditionedRouter
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


def test_task_conditioned_router_routing_bias_mode_emits_bias_diagnostics() -> None:
    base_router = ContextualTransformerRouter(token_dim=4)
    router = TaskConditionedRouter(
        token_dim=4,
        prior_dim=3,
        base_router=base_router,
        strength=0.5,
        mode="routing_bias",
        router_kind="contextual_transformer_router_task_prior",
    )
    router.set_task_prior_context(torch.tensor([1.0, 0.5, -0.25], dtype=torch.float32))
    tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    outputs = router(tokens, full_index=0)
    diagnostics = outputs.extra_diagnostics or {}
    assert diagnostics["task_prior_enabled"] == 1.0
    assert diagnostics["task_prior_mode"] == "routing_bias"
    assert "task_prior_routing_bias_mean" in diagnostics
    assert "task_prior_routing_bias_std" in diagnostics
    assert "task_prior_defer_bias_mean" in diagnostics


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


def test_regression_predictions_include_task_prior_diagnostics_on_clean_route():
    class DummyRouter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router_kind = "contextual_transformer_router_task_prior"

        def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
            batch, experts, _ = tokens.shape
            weights = torch.full((batch, experts), 1.0 / experts, dtype=torch.float32)
            return RouterOutputs(
                defer_prob=torch.zeros((batch, 1), dtype=torch.float32),
                specialist_weights=weights,
                full_index=full_index,
                router_kind=self.router_kind,
                extra_diagnostics={"task_prior_enabled": 1.0},
            )

    gd = GraphDrone(
        GraphDroneConfig(
            legitimacy_gate=LegitimacyGateConfig(
                enabled=False,
                regression_enabled=False,
                binary_enabled=False,
                multiclass_enabled=False,
            )
        )
    )
    gd._problem_type = "regression"
    gd._router = DummyRouter()
    gd._task_prior_diagnostics = {
        "task_prior_query_dataset": "cpu_act",
        "task_prior_top_neighbor": "elevators",
        "task_prior_exact_reuse_used": True,
    }
    gd._router_fit_diagnostics = {}
    gd._attention_diagnostics = lambda **_: {}
    gd._build_regression_tokens = lambda matrix, batch: SimpleNamespace(
        tokens=torch.zeros((matrix.shape[0], len(batch.expert_ids), 4), dtype=torch.float32)
    )

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
        predictions=np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32),
        full_expert_id="FULL",
        full_index=0,
        quality_scores=None,
    )

    preds, diagnostics = gd._regression_predictions(np.zeros((2, 2), dtype=np.float32), batch)
    assert preds.shape == (2,)
    assert diagnostics["router_kind"] == "contextual_transformer_router_task_prior"
    assert diagnostics["task_prior_enabled"] == 1.0
    assert diagnostics["task_prior_query_dataset"] == "cpu_act"
    assert diagnostics["task_prior_top_neighbor"] == "elevators"
    assert diagnostics["task_prior_exact_reuse_used"] is True


def test_fit_regression_router_attaches_task_prior_wrapper(monkeypatch):
    class DummyFactory:
        def predict_all(self, X: np.ndarray) -> ExpertPredictionBatch:
            return ExpertPredictionBatch(
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
                predictions=np.tile(np.array([[1.0, 1.2]], dtype=np.float32), (len(X), 1)),
                full_expert_id="FULL",
                full_index=0,
                quality_scores=None,
            )

    class DummyRouter(torch.nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            self.router_kind = kind
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
            batch, experts, _ = tokens.shape
            weights = torch.full((batch, experts), 1.0 / experts, dtype=torch.float32, device=tokens.device)
            defer = torch.zeros((batch, 1), dtype=torch.float32, device=tokens.device)
            return RouterOutputs(
                defer_prob=defer,
                specialist_weights=weights,
                full_index=full_index,
                router_kind=self.router_kind,
            )

    gd = GraphDrone(
        GraphDroneConfig(
            router=SetRouterConfig(
                kind="contextual_transformer",
                task_prior_bank_dir="/tmp/task-bank",
                task_prior_encoder_kind="transformer",
                task_prior_dataset_key="cpu_act",
                task_prior_strength=0.5,
            )
        )
    )
    gd._problem_type = "regression"
    gd._expert_factory = DummyFactory()
    gd._train_views = {}
    gd._router_fit_diagnostics = {}

    monkeypatch.setattr("graphdrone_fit.model.build_set_router", lambda config, token_dim, n_experts: DummyRouter(config.kind))
    monkeypatch.setattr(gd, "_seed_router_training", lambda: None)
    monkeypatch.setattr(gd, "_sample_aux_rows", lambda X: X[: min(len(X), 4)])
    monkeypatch.setattr(gd, "_build_regression_tokens", lambda X, batch: SimpleNamespace(tokens=torch.zeros((len(X), len(batch.expert_ids), 4), dtype=torch.float32)))
    monkeypatch.setattr(gd, "_fit_router_auxiliary_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(gd, "_optimize_regression_router_module", lambda *args, **kwargs: None)

    attach_calls = []

    def fake_attach(*, router, router_cfg, tokens, descriptors, task_type):
        attach_calls.append(
            {
                "router_kind": getattr(router, "router_kind", ""),
                "dataset_key": router_cfg.task_prior_dataset_key,
                "task_type": task_type,
                "token_shape": tuple(tokens.shape),
            }
        )
        wrapped = DummyRouter(f"{router.router_kind}_task_prior")
        gd._task_prior_diagnostics = {"task_prior_query_dataset": router_cfg.task_prior_dataset_key}
        return wrapped

    monkeypatch.setattr(gd, "_maybe_attach_task_prior_router", fake_attach)

    X = np.random.RandomState(0).randn(32, 2).astype(np.float32)
    y = np.random.RandomState(1).randn(32).astype(np.float32)
    gd._fit_regression_router(X, y)

    assert len(attach_calls) == 1
    assert attach_calls[0]["task_type"] == "regression"
    assert attach_calls[0]["dataset_key"] == "cpu_act"
    assert gd._router.router_kind == "contextual_transformer_task_prior"


def test_normalize_regression_router_targets_stabilizes_large_scale_targets():
    expert_predictions = torch.tensor(
        [
            [200000.0, 205000.0, 198000.0],
            [180000.0, 175000.0, 182500.0],
            [220000.0, 218000.0, 225000.0],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor([210000.0, 170000.0, 230000.0], dtype=torch.float32)

    norm_preds, norm_targets, diagnostics = GraphDrone._normalize_regression_router_targets(
        expert_predictions=expert_predictions,
        y_true=y_true,
    )

    assert diagnostics["validation_router_target_scale"] > 1.0
    assert torch.isfinite(norm_preds).all()
    assert torch.isfinite(norm_targets).all()
    assert float(norm_targets.abs().mean().item()) < float(y_true.abs().mean().item())
    assert float(norm_preds.abs().mean().item()) < float(expert_predictions.abs().mean().item())


def test_normalize_regression_token_predictions_stabilizes_anchor_scale():
    predictions = np.array(
        [
            [200000.0, 205000.0, 198000.0],
            [180000.0, 175000.0, 182500.0],
            [220000.0, 218000.0, 225000.0],
        ],
        dtype=np.float32,
    )

    normalized, diagnostics = GraphDrone._normalize_regression_token_predictions(
        predictions,
        full_index=0,
    )

    assert diagnostics["regression_token_prediction_scale"] > 1.0
    assert np.isfinite(normalized).all()
    assert float(np.mean(np.abs(normalized[:, 0]))) < float(np.mean(np.abs(predictions[:, 0])))
