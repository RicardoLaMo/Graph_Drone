from __future__ import annotations

import pandas as pd

from graphdrone_fit.claim_checks import evaluate_claims


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_legitimacy_claim_reports_supported_when_exit_and_latency_improve():
    paired = _frame(
        [
            {
                "dataset": "california",
                "fold": 0,
                "task_type": "regression",
                "router_kind_challenger": "contextual_transformer_router",
                "exit_frac_challenger": 0.2,
                "latency_improvement": 0.15,
                "rmse_rel_improvement": -0.001,
            }
        ]
    )
    report = evaluate_claims(paired)
    claim = report["claims"]["legitimacy_gate"]
    assert claim["component_status"] == "supported"
    assert claim["integration_status"] == "not_translating"


def test_rotor_claim_reports_inactive_without_specialists():
    paired = _frame(
        [
            {
                "dataset": "california",
                "fold": 0,
                "task_type": "regression",
                "router_kind_challenger": "contextual_transformer_rotor",
                "n_specialists_challenger": 0.0,
                "alignment_cosine_gain_challenger": 0.0,
            }
        ]
    )
    report = evaluate_claims(paired)
    claim = report["claims"]["rotor_alignment"]
    assert claim["component_status"] == "inactive"


def test_rotor_claim_separates_component_success_from_bottom_line():
    paired = _frame(
        [
            {
                "dataset": "credit_g",
                "fold": 0,
                "task_type": "classification",
                "router_kind_challenger": "noise_gate_router_rotor",
                "n_specialists_challenger": 3.0,
                "alignment_cosine_gain_challenger": 0.08,
                "f1_delta": -0.01,
                "log_loss_rel_improvement": -0.02,
            }
        ]
    )
    report = evaluate_claims(paired)
    claim = report["claims"]["rotor_alignment"]
    assert claim["component_status"] == "supported"
    assert claim["integration_status"] == "not_translating"


def test_ot_claim_reports_supported_when_gate_closes():
    paired = _frame(
        [
            {
                "dataset": "diabetes",
                "fold": 0,
                "task_type": "classification",
                "router_kind_challenger": "ot_noise_gate_router",
                "mean_ot_cost_challenger": 0.4,
                "mean_specialist_validity_challenger": 0.6,
                "closed_specialist_frac_challenger": 0.3,
                "f1_delta": 0.01,
                "log_loss_rel_improvement": 0.03,
            }
        ]
    )
    report = evaluate_claims(paired)
    claim = report["claims"]["ot_noise_gate"]
    assert claim["component_status"] == "supported"
    assert claim["integration_status"] == "translating"
