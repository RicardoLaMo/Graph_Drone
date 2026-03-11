from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.openml_regression_benchmark.scripts.analyze_router_full_regret import (
    analyze_run as analyze_full_regret_run,
)
from experiments.openml_regression_benchmark.scripts.analyze_router_mechanism import analyze_run
from experiments.openml_regression_benchmark.scripts.analyze_view_home_quality import (
    analyze_run as analyze_view_home_run,
)
from experiments.openml_regression_benchmark.scripts.analyze_two_expert_competition import (
    analyze_run as analyze_two_expert_run,
)
from experiments.openml_regression_benchmark.scripts.analyze_signal_noise_tradeoff import (
    analyze_run as analyze_signal_noise_run,
)
from experiments.openml_regression_benchmark.scripts.summarize_full_regret_suite import (
    summarize as summarize_full_regret_suite,
)
from experiments.openml_regression_benchmark.scripts.summarize_houses_seed_sweep import summarize
from experiments.openml_regression_benchmark.scripts.summarize_two_expert_suite import (
    summarize as summarize_two_expert_suite,
)
from experiments.openml_regression_benchmark.scripts.summarize_signal_noise_suite import (
    summarize as summarize_signal_noise_suite,
)
from experiments.openml_regression_benchmark.scripts.summarize_view_home_suite import (
    summarize as summarize_view_home_suite,
)


def test_analyze_run_reports_adaptive_vs_fixed_delta(tmp_path) -> None:
    run_dir = tmp_path / "houses__r0f0"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    payload = {
        "rows": [
            {"model": "GraphDrone_router", "test_rmse": 0.19, "val_rmse": 0.18},
            {"model": "GraphDrone_router_fixed", "test_rmse": 0.20, "val_rmse": 0.19},
            {"model": "GraphDrone_crossfit", "test_rmse": 0.191, "val_rmse": 0.181},
            {"model": "GraphDrone_crossfit_fixed", "test_rmse": 0.201, "val_rmse": 0.191},
        ]
    }
    (run_dir / "graphdrone_results.json").write_text(json.dumps(payload) + "\n")

    np.savez_compressed(
        artifacts / "graphdrone_predictions.npz",
        view_names=np.array(["FULL", "GEO", "DOMAIN", "LOWRANK"]),
        y_test=np.array([1.0, 2.0], dtype=np.float32),
        pred_test=np.array([[1.0, 1.4, 0.9, 1.1], [2.0, 2.4, 1.8, 2.2]], dtype=np.float32),
        quality_test=np.array([[0.1] * 11, [0.2] * 11], dtype=np.float32),
        router_pred_test=np.array([1.05, 2.05], dtype=np.float32),
        router_fixed_pred_test=np.array([1.15, 2.15], dtype=np.float32),
        router_weights_test=np.array([[0.8, 0.1, 0.05, 0.05], [0.7, 0.2, 0.05, 0.05]], dtype=np.float32),
        router_fixed_weights_test=np.array([[0.75, 0.15, 0.05, 0.05], [0.75, 0.15, 0.05, 0.05]], dtype=np.float32),
        crossfit_pred_test=np.array([1.04, 2.04], dtype=np.float32),
        crossfit_fixed_pred_test=np.array([1.14, 2.14], dtype=np.float32),
        crossfit_weights_test=np.array([[0.79, 0.11, 0.05, 0.05], [0.71, 0.19, 0.05, 0.05]], dtype=np.float32),
        crossfit_fixed_weights_test=np.array([[0.75, 0.15, 0.05, 0.05], [0.75, 0.15, 0.05, 0.05]], dtype=np.float32),
    )

    summary = analyze_run(run_dir, adaptive_prefix="router", label="GraphDrone")
    assert summary["adaptive_minus_fixed_test_rmse"] > 0.0
    assert summary["positive_improvement_fraction"] == 1.0
    assert summary["mean_weight_l1_shift"] > 0.0


def test_summarize_houses_seed_sweep_aggregates_runs(tmp_path) -> None:
    root = tmp_path / "houses_sweep"
    for idx, delta in enumerate([0.001, 0.002]):
        run_dir = root / f"houses__seed{idx}"
        run_dir.mkdir(parents=True)
        payload = {
            "rows": [
                {"model": "GraphDrone_FULL", "test_rmse": 0.203},
                {"model": "GraphDrone_router", "test_rmse": 0.200 - delta},
                {"model": "GraphDrone_router_fixed", "test_rmse": 0.200},
                {"model": "GraphDrone_crossfit", "test_rmse": 0.199 - delta},
                {"model": "GraphDrone_crossfit_fixed", "test_rmse": 0.199},
            ]
        }
        (run_dir / "graphdrone_results.json").write_text(json.dumps(payload) + "\n")

    summary = summarize(root)
    assert summary["n_runs"] == 2
    assert summary["router_adaptive_minus_fixed"]["mean"] > 0.0
    assert summary["crossfit_adaptive_minus_fixed"]["positive_fraction"] == 1.0


def test_analyze_full_regret_separates_false_diversion_and_missed_opportunity(tmp_path) -> None:
    run_dir = tmp_path / "california_housing_openml__r0f0"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    payload = {
        "rows": [
            {"model": "GraphDrone_FULL", "test_rmse": 0.50},
            {"model": "GraphDrone_router", "test_rmse": 0.48},
            {"model": "GraphDrone_router_fixed", "test_rmse": 0.49},
        ]
    }
    (run_dir / "graphdrone_results.json").write_text(json.dumps(payload) + "\n")

    np.savez_compressed(
        artifacts / "graphdrone_predictions.npz",
        view_names=np.array(["FULL", "GEO"]),
        y_test=np.zeros(4, dtype=np.float32),
        pred_test=np.array(
            [
                [0.0, 1.0],
                [0.0, 0.5],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        quality_test=np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.1, 0.3, 0.4],
                [0.4, 0.1, 0.3, 0.5],
                [0.3, 0.1, 0.2, 0.6],
            ],
            dtype=np.float32,
        ),
        router_pred_test=np.array([0.4, 0.025, 0.3, 0.9], dtype=np.float32),
        router_fixed_pred_test=np.array([0.2, 0.1, 0.8, 0.8], dtype=np.float32),
        router_weights_test=np.array(
            [
                [0.6, 0.4],
                [0.95, 0.05],
                [0.3, 0.7],
                [0.9, 0.1],
            ],
            dtype=np.float32,
        ),
        router_fixed_weights_test=np.array(
            [
                [0.8, 0.2],
                [0.8, 0.2],
                [0.8, 0.2],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    summary = analyze_full_regret_run(run_dir, adaptive_prefix="router", label="GraphDrone")
    assert summary["global"]["full_oracle_fraction"] == 0.5
    assert summary["full_oracle_case"]["false_diversion_mean_cost"] > 0.0
    assert summary["non_full_oracle_case"]["mean_potential_gain"] == 1.0
    assert summary["non_full_oracle_case"]["adaptive_capture_ratio_total"] == pytest.approx(0.4)
    assert summary["non_full_oracle_case"]["high_fixed_full_weight_fraction"] == 1.0


def test_summarize_full_regret_suite_aggregates_run_summaries(tmp_path) -> None:
    root = tmp_path / "suite"
    for idx, capture in enumerate([0.10, 0.25]):
        artifacts = root / f"seed{idx}" / "artifacts"
        artifacts.mkdir(parents=True)
        payload = {
            "run_dir": f"run{idx}",
            "global": {
                "adaptive_minus_full_test_rmse": 0.001 + idx,
                "adaptive_minus_fixed_test_rmse": 0.0005 + idx,
                "full_oracle_fraction": 0.4 + 0.1 * idx,
            },
            "full_oracle_case": {
                "false_diversion_mean_cost": 0.01 + idx,
                "false_diversion_positive_fraction": 0.6 + 0.1 * idx,
            },
            "non_full_oracle_case": {
                "adaptive_capture_ratio_total": capture,
                "fixed_capture_ratio_total": capture - 0.05,
                "missed_opportunity_mean_cost": 0.09 + idx,
            },
        }
        (artifacts / "router_full_regret_summary.json").write_text(json.dumps(payload) + "\n")

    summary = summarize_full_regret_suite(root, adaptive_prefix="router")
    assert summary["n_runs"] == 2
    assert summary["adaptive_capture_minus_fixed"]["positive_fraction"] == 1.0
    assert summary["full_oracle_fraction"]["mean"] == pytest.approx(0.45)


def test_analyze_view_home_quality_reports_per_view_capture(tmp_path) -> None:
    run_dir = tmp_path / "houses__r0f0"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    payload = {
        "rows": [
            {"model": "GraphDrone_FULL", "test_rmse": 0.21},
            {"model": "GraphDrone_router", "test_rmse": 0.20},
            {"model": "GraphDrone_router_fixed", "test_rmse": 0.205},
        ]
    }
    (run_dir / "graphdrone_results.json").write_text(json.dumps(payload) + "\n")

    np.savez_compressed(
        artifacts / "graphdrone_predictions.npz",
        view_names=np.array(["FULL", "GEO"]),
        y_test=np.zeros(4, dtype=np.float32),
        pred_test=np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.2],
            ],
            dtype=np.float32,
        ),
        router_pred_test=np.array([0.2, 0.2, 0.8, 0.02], dtype=np.float32),
        router_fixed_pred_test=np.array([0.1, 0.7, 0.7, 0.04], dtype=np.float32),
        router_weights_test=np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.9, 0.1],
            ],
            dtype=np.float32,
        ),
        router_fixed_weights_test=np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.7, 0.3],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    summary = analyze_view_home_run(run_dir, adaptive_prefix="router", label="GraphDrone")
    assert summary["per_view_home_subset"]["FULL"]["n_rows"] == 2
    assert summary["per_view_home_subset"]["GEO"]["n_rows"] == 2
    assert summary["per_view_home_subset"]["GEO"]["capture_gap_vs_fixed"] > 0.0


def test_summarize_view_home_suite_aggregates_per_view_metrics(tmp_path) -> None:
    root = tmp_path / "suite"
    for idx, gap in enumerate([0.02, -0.01]):
        artifacts = root / f"seed{idx}" / "artifacts"
        artifacts.mkdir(parents=True)
        payload = {
            "per_view_home_subset": {
                "FULL": {
                    "capture_gap_vs_fixed": 0.0,
                    "mean_potential_gain_vs_full": 0.0,
                    "adaptive_capture_ratio_total": 0.0,
                    "mean_adaptive_full_weight": 0.8,
                },
                "GEO": {
                    "capture_gap_vs_fixed": gap,
                    "mean_potential_gain_vs_full": 0.1 + idx,
                    "adaptive_capture_ratio_total": 0.2 + idx,
                    "mean_adaptive_full_weight": 0.7 - 0.1 * idx,
                },
            }
        }
        (artifacts / "router_view_home_summary.json").write_text(json.dumps(payload) + "\n")

    summary = summarize_view_home_suite(root, adaptive_prefix="router")
    assert summary["n_runs"] == 2
    assert summary["per_view"]["GEO"]["capture_gap_vs_fixed"]["mean"] == pytest.approx(0.005)


def test_analyze_two_expert_competition_reports_best_candidate(tmp_path) -> None:
    run_dir = tmp_path / "miami_housing__r0f0"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    payload = {
        "rows": [
            {"model": "GraphDrone_FULL", "test_rmse": 0.40},
            {"model": "GraphDrone_router", "test_rmse": 0.39},
            {"model": "GraphDrone_router_fixed", "test_rmse": 0.395},
        ],
        "runtime": {"resolved_device": "cpu"},
    }
    (run_dir / "graphdrone_results.json").write_text(json.dumps(payload) + "\n")

    np.savez_compressed(
        artifacts / "graphdrone_predictions.npz",
        view_names=np.array(["FULL", "GEO", "DOMAIN"]),
        y_val=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        y_test=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        pred_val=np.array(
            [
                [0.0, 0.3, 0.2],
                [0.1, 0.2, 0.4],
                [0.8, 0.9, 0.1],
                [0.9, 0.8, 0.2],
            ],
            dtype=np.float32,
        ),
        pred_test=np.array(
            [
                [0.0, 0.25, 0.2],
                [0.1, 0.15, 0.3],
                [0.8, 0.9, 0.2],
                [0.9, 0.85, 0.25],
            ],
            dtype=np.float32,
        ),
        quality_val=np.array([[0.1] * 11, [0.2] * 11, [0.3] * 11, [0.4] * 11], dtype=np.float32),
        quality_test=np.array([[0.1] * 11, [0.2] * 11, [0.3] * 11, [0.4] * 11], dtype=np.float32),
    )

    summary = analyze_two_expert_run(run_dir, adaptive_prefix="router", label="GraphDrone", seed=7)
    assert summary["best_candidate_view"] in {"GEO", "DOMAIN"}
    assert set(summary["candidates"].keys()) == {"GEO", "DOMAIN"}


def test_summarize_two_expert_suite_aggregates_best_pair_gains(tmp_path) -> None:
    root = tmp_path / "suite"
    for idx, gain in enumerate([0.01, -0.02]):
        artifacts = root / f"seed{idx}" / "artifacts"
        artifacts.mkdir(parents=True)
        payload = {
            "best_candidate_view": "GEO" if idx == 0 else "DOMAIN",
            "best_two_expert": {
                "adaptive_minus_full_router": gain,
                "adaptive_minus_full_expert": gain - 0.01,
            },
        }
        (artifacts / "router_two_expert_summary.json").write_text(json.dumps(payload) + "\n")

    summary = summarize_two_expert_suite(root, adaptive_prefix="router")
    assert summary["n_runs"] == 2
    assert summary["best_two_expert_gain_vs_full_router"]["mean"] == pytest.approx(-0.005)
    assert summary["best_candidate_counts"]["GEO"] == 1


def test_analyze_signal_noise_tradeoff_classifies_competition_plus_weak_expert(tmp_path) -> None:
    run_dir = tmp_path / "diamonds__r0f0"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    full_regret = {
        "global": {"full_oracle_fraction": 0.5},
        "full_oracle_case": {
            "false_diversion_mean_cost": 10.0,
            "false_diversion_positive_fraction": 0.7,
        },
        "non_full_oracle_case": {
            "mean_potential_gain": 5.0,
            "mean_adaptive_realized_gain": 1.0,
            "adaptive_capture_ratio_total": 0.2,
            "fixed_capture_ratio_total": 0.15,
        },
    }
    view_home = {
        "per_view_home_subset": {
            "GEO": {
                "row_fraction": 0.3,
                "mean_potential_gain_vs_full": 4.0,
                "adaptive_capture_ratio_total": 0.25,
                "fixed_capture_ratio_total": 0.20,
                "capture_gap_vs_fixed": 0.05,
                "mean_adaptive_full_weight": 0.8,
                "mean_adaptive_view_weight": 0.2,
            }
        }
    }
    two_expert = {
        "best_candidate_view": "GEO",
        "best_two_expert": {
            "pair_name": "FULL+GEO",
            "adaptive_test_rmse": 1.0,
            "adaptive_minus_full_router": 0.4,
            "adaptive_minus_full_expert": -0.1,
            "adaptive_minus_full_fixed": 0.5,
        },
        "global_reference": {
            "full_router_test_rmse": 1.4,
            "full_expert_test_rmse": 0.9,
        },
    }
    (artifacts / "router_full_regret_summary.json").write_text(json.dumps(full_regret) + "\n")
    (artifacts / "router_view_home_summary.json").write_text(json.dumps(view_home) + "\n")
    (artifacts / "router_two_expert_summary.json").write_text(json.dumps(two_expert) + "\n")

    summary = analyze_signal_noise_run(run_dir, adaptive_prefix="router")
    assert summary["classification"] == "competition_noise_plus_weak_expert"
    assert summary["best_view"] == "GEO"


def test_summarize_signal_noise_suite_aggregates_classifications(tmp_path) -> None:
    root = tmp_path / "suite"
    payloads = [
        {
            "classification": "useful_signal_obscured_by_competition",
            "best_view": "GEO",
            "global": {
                "competition_noise_gain_vs_full_router": 0.1,
                "best_pair_gain_vs_full_expert": 0.02,
            },
            "best_view_tradeoff": {"capture_gap_vs_fixed": 0.03},
        },
        {
            "classification": "competition_noise_plus_weak_expert",
            "best_view": "LOWRANK",
            "global": {
                "competition_noise_gain_vs_full_router": 0.2,
                "best_pair_gain_vs_full_expert": -0.01,
            },
            "best_view_tradeoff": {"capture_gap_vs_fixed": -0.02},
        },
    ]
    for idx, payload in enumerate(payloads):
        artifacts = root / f"seed{idx}" / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "router_signal_noise_tradeoff.json").write_text(json.dumps(payload) + "\n")

    summary = summarize_signal_noise_suite(root, adaptive_prefix="router")
    assert summary["n_runs"] == 2
    assert summary["classification_counts"]["useful_signal_obscured_by_competition"] == 1
    assert summary["best_view_counts"]["LOWRANK"] == 1
