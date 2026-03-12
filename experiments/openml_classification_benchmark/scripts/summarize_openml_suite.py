from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OpenML classification benchmark runs")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["GraphDrone", "TabPFN", "TabM"])
    return parser.parse_args()


def _load_rows(input_root: Path, *, allowed_models: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result_path in sorted(input_root.rglob("*_results.json")):
        if result_path.name == "graphdrone_fit_results.json":
            continue
        payload = json.loads(result_path.read_text())
        payload_model = payload.get("model")
        if payload_model is not None and str(payload_model) not in allowed_models:
            continue
        dataset = payload["dataset"]
        payload_rows = payload.get("rows", [])
        if not payload_rows and "metrics" in payload and "model" in payload:
            metrics = payload["metrics"]["test"]
            payload_rows = [
                {
                    "model": payload["model"],
                    "test_accuracy": metrics.get("accuracy"),
                    "test_macro_f1": metrics.get("f1_macro", metrics.get("macro_f1")),
                    "test_roc_auc": metrics.get("roc_auc", metrics.get("roc_auc_ovr_macro")),
                    "test_pr_auc": metrics.get("pr_auc", metrics.get("pr_auc_ovr_macro")),
                    "test_log_loss": metrics.get("log_loss"),
                }
            ]
        for row in payload_rows:
            row_model = str(row.get("model", payload_model))
            if row_model not in allowed_models:
                continue
            rows.append(
                {
                    "dataset_key": dataset["dataset_key"],
                    "dataset_name": dataset["dataset_name"],
                    "repeat": dataset["repeat"],
                    "fold": dataset["fold"],
                    **{**row, "model": row_model},
                }
            )
    return rows


def _mean_or_nan(series: pd.Series) -> float | None:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.input_root, allowed_models=set(args.models))
    if not rows:
        raise SystemExit(f"No benchmark result json files found under {args.input_root}")

    df = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (dataset_key, dataset_name, model), group in df.groupby(["dataset_key", "dataset_name", "model"], sort=True):
        summary_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "model": model,
                "mean_test_accuracy": _mean_or_nan(group["test_accuracy"]),
                "mean_test_macro_f1": _mean_or_nan(group["test_macro_f1"]),
                "mean_test_roc_auc": _mean_or_nan(group["test_roc_auc"]),
                "mean_test_pr_auc": _mean_or_nan(group["test_pr_auc"]),
                "mean_test_log_loss": _mean_or_nan(group["test_log_loss"]),
                "runs": int(len(group)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset_key", "model"]).reset_index(drop=True)
    summary_df.to_csv(args.input_root / "openml_benchmark_summary.csv", index=False)
    try:
        table_text = summary_df.to_markdown(index=False)
    except ImportError:
        table_text = summary_df.to_string(index=False)
    md = "# OpenML Classification Benchmark Summary\n\n" + table_text + "\n"
    (args.input_root / "openml_benchmark_summary.md").write_text(md)


if __name__ == "__main__":
    main()
