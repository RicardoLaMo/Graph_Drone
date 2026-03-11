from __future__ import annotations

import json

import numpy as np

from experiments.openml_regression_benchmark.scripts.analyze_router_mechanism import analyze_run
from experiments.openml_regression_benchmark.scripts.summarize_houses_seed_sweep import summarize


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
