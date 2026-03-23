from __future__ import annotations

import pandas as pd
import torch

from graphdrone_fit.champion_challenger import (
    build_dataset_summary,
    build_paired_task_table,
    evaluate_promotion,
)
from graphdrone_fit.model import _seed_torch_generators
from graphdrone_fit.presets import build_graphdrone_config_from_preset
from graphdrone_fit.set_router import build_set_router


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_v120_champion_preset_disables_afc_features():
    config = build_graphdrone_config_from_preset(
        preset="v1_20_champion",
        n_classes=2,
        default_router_kind="noise_gate_router",
    )
    assert config.router.kind == "noise_gate_router"
    assert config.router.router_seed == 42
    assert config.legitimacy_gate.enabled is False
    assert config.hyperbolic_descriptors.enabled is False


def test_afc_preset_respects_router_seed_env(monkeypatch):
    monkeypatch.setenv("GRAPHDRONE_ROUTER_SEED", "123")
    monkeypatch.setenv("GRAPHDRONE_FREEZE_BASE_ROUTER", "1")
    config = build_graphdrone_config_from_preset(
        preset="afc_candidate",
        n_classes=1,
        default_router_kind="contextual_transformer",
    )
    assert config.router.router_seed == 123
    assert config.router.freeze_base_router is True


def test_router_seed_makes_initialization_reproducible():
    _seed_torch_generators(7)
    router_a = build_set_router(build_graphdrone_config_from_preset(
        preset="afc_candidate",
        n_classes=1,
        default_router_kind="contextual_transformer",
    ).router, token_dim=8, n_experts=4)
    params_a = [param.detach().clone() for param in router_a.parameters()]

    _seed_torch_generators(7)
    router_b = build_set_router(build_graphdrone_config_from_preset(
        preset="afc_candidate",
        n_classes=1,
        default_router_kind="contextual_transformer",
    ).router, token_dim=8, n_experts=4)
    params_b = [param.detach().clone() for param in router_b.parameters()]

    assert len(params_a) == len(params_b)
    for left, right in zip(params_a, params_b):
        assert torch.allclose(left, right)


def test_build_paired_task_table_computes_expected_signs():
    champion = _frame(
        [
            {
                "dataset": "california",
                "fold": 0,
                "task_type": "regression",
                "status": "ok",
                "rmse": 1.0,
                "mae": 0.5,
                "r2": 0.7,
                "elapsed": 10.0,
                "mean_attention_FULL": 0.8,
            },
            {
                "dataset": "segment",
                "fold": 0,
                "task_type": "classification",
                "status": "ok",
                "f1_macro": 0.90,
                "log_loss": 0.30,
                "elapsed": 12.0,
            },
        ]
    )
    challenger = _frame(
        [
            {
                "dataset": "california",
                "fold": 0,
                "task_type": "regression",
                "status": "ok",
                "rmse": 0.95,
                "mae": 0.45,
                "r2": 0.72,
                "elapsed": 8.0,
                "mean_attention_FULL": 0.6,
            },
            {
                "dataset": "segment",
                "fold": 0,
                "task_type": "classification",
                "status": "ok",
                "f1_macro": 0.91,
                "log_loss": 0.27,
                "elapsed": 10.0,
            },
        ]
    )

    paired = build_paired_task_table(champion, challenger)
    reg_row = paired[paired["task_type"] == "regression"].iloc[0]
    clf_row = paired[paired["task_type"] == "classification"].iloc[0]
    assert reg_row["rmse_rel_improvement"] > 0
    assert reg_row["r2_delta"] > 0
    assert reg_row["latency_improvement"] > 0
    assert reg_row["mean_attention_FULL_delta"] < 0
    assert clf_row["f1_delta"] > 0
    assert clf_row["log_loss_delta"] < 0
    assert clf_row["log_loss_rel_improvement"] > 0


def test_regression_promotion_passes_when_guardrails_clear():
    champion = _frame(
        [
            {"dataset": "california", "fold": 0, "task_type": "regression", "status": "ok", "rmse": 1.00, "mae": 0.50, "r2": 0.70, "elapsed": 10.0},
            {"dataset": "cpu_act", "fold": 0, "task_type": "regression", "status": "ok", "rmse": 2.00, "mae": 1.00, "r2": 0.60, "elapsed": 10.0},
        ]
    )
    challenger = _frame(
        [
            {"dataset": "california", "fold": 0, "task_type": "regression", "status": "ok", "rmse": 0.98, "mae": 0.48, "r2": 0.705, "elapsed": 9.0},
            {"dataset": "cpu_act", "fold": 0, "task_type": "regression", "status": "ok", "rmse": 1.96, "mae": 0.97, "r2": 0.602, "elapsed": 9.2},
        ]
    )
    paired = build_paired_task_table(champion, challenger)
    summary = build_dataset_summary(paired)
    decision = evaluate_promotion(paired, summary)
    assert decision["task_decisions"]["regression"]["pass"] is True
    assert decision["pass"] is True


def test_classification_promotion_fails_on_dataset_guardrail():
    champion = _frame(
        [
            {"dataset": "segment", "fold": 0, "task_type": "classification", "status": "ok", "f1_macro": 0.90, "log_loss": 0.30, "elapsed": 12.0},
            {"dataset": "sdss17", "fold": 0, "task_type": "classification", "status": "ok", "f1_macro": 0.88, "log_loss": 0.22, "elapsed": 12.0},
        ]
    )
    challenger = _frame(
        [
            {"dataset": "segment", "fold": 0, "task_type": "classification", "status": "ok", "f1_macro": 0.88, "log_loss": 0.29, "elapsed": 10.0},
            {"dataset": "sdss17", "fold": 0, "task_type": "classification", "status": "ok", "f1_macro": 0.89, "log_loss": 0.21, "elapsed": 10.0},
        ]
    )
    paired = build_paired_task_table(champion, challenger)
    summary = build_dataset_summary(paired)
    decision = evaluate_promotion(paired, summary)
    assert decision["task_decisions"]["classification"]["pass"] is False
    assert decision["pass"] is False


def test_efficiency_only_promotion_allows_small_metric_regressions():
    champion = _frame(
        [
            {"dataset": "california", "fold": 0, "task_type": "regression", "status": "ok", "rmse": 1.00, "mae": 0.50, "r2": 0.70, "elapsed": 10.0},
            {"dataset": "segment", "fold": 0, "task_type": "classification", "status": "ok", "f1_macro": 0.90, "log_loss": 0.30, "elapsed": 10.0},
        ]
    )
    challenger = _frame(
        [
            {"dataset": "california", "fold": 0, "task_type": "regression", "status": "ok", "rmse": 1.004, "mae": 0.501, "r2": 0.6995, "elapsed": 8.5},
            {"dataset": "segment", "fold": 0, "task_type": "classification", "status": "ok", "f1_macro": 0.899, "log_loss": 0.305, "elapsed": 8.7},
        ]
    )
    paired = build_paired_task_table(champion, challenger)
    summary = build_dataset_summary(paired)
    decision = evaluate_promotion(paired, summary, efficiency_only=True)
    assert decision["task_decisions"]["regression"]["pass"] is True
    assert decision["task_decisions"]["classification"]["pass"] is True
    assert decision["pass"] is True
