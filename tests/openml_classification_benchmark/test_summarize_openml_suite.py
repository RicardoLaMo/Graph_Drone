from __future__ import annotations

import json

from experiments.openml_classification_benchmark.scripts.summarize_openml_suite import _load_rows


def test_load_rows_filters_out_disallowed_models(tmp_path) -> None:
    root = tmp_path / "reports"
    run_dir = root / "diabetes__r0f0"
    run_dir.mkdir(parents=True)
    payload = {
        "model": "TabR",
        "dataset": {
            "dataset_key": "diabetes",
            "dataset_name": "Diabetes",
            "repeat": 0,
            "fold": 0,
        },
        "rows": [
            {
                "model": "TabR",
                "test_accuracy": 0.7,
                "test_macro_f1": 0.6,
                "test_roc_auc": 0.8,
                "test_pr_auc": 0.75,
                "test_log_loss": 0.5,
            }
        ],
    }
    (run_dir / "tabr_results.json").write_text(json.dumps(payload) + "\n")

    rows = _load_rows(root, allowed_models={"GraphDrone", "TabPFN", "TabM"})
    assert rows == []


def test_load_rows_keeps_allowed_models(tmp_path) -> None:
    root = tmp_path / "reports"
    run_dir = root / "diabetes__r0f0"
    run_dir.mkdir(parents=True)
    payload = {
        "model": "GraphDrone",
        "dataset": {
            "dataset_key": "diabetes",
            "dataset_name": "Diabetes",
            "repeat": 0,
            "fold": 0,
        },
        "rows": [
            {
                "model": "GraphDrone",
                "test_accuracy": 0.8,
                "test_macro_f1": 0.7,
                "test_roc_auc": 0.85,
                "test_pr_auc": 0.8,
                "test_log_loss": 0.4,
            }
        ],
    }
    (run_dir / "graphdrone_results.json").write_text(json.dumps(payload) + "\n")

    rows = _load_rows(root, allowed_models={"GraphDrone", "TabPFN", "TabM"})
    assert len(rows) == 1
    assert rows[0]["model"] == "GraphDrone"
