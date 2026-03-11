from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OpenML regression benchmark runs")
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports",
    )
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def load_rows(reports_root: Path, *, include_smoke: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    required = {"graphdrone_results.json", "tabpfn_results.json", "tabr_results.json", "tabm_results.json"}
    for run_dir in sorted(path for path in reports_root.iterdir() if path.is_dir()):
        if not include_smoke and run_dir.name.endswith("__smoke"):
            continue
        present = {path.name for path in run_dir.glob("*.json")}
        if not required.issubset(present):
            continue

        payload = json.loads((run_dir / "graphdrone_results.json").read_text())
        dataset = payload["dataset"]
        for row in payload["rows"]:
            dataset = payload["dataset"]
            rows.append(
                {
                    "dataset": dataset["dataset_key"],
                    "repeat": dataset["repeat"],
                    "fold": dataset["fold"],
                    "model": row["model"],
                    "test_rmse": row["test_rmse"],
                    "val_rmse": row["val_rmse"],
                    "test_mae": row["test_mae"],
                    "test_r2": row["test_r2"],
                }
            )

        for name in ("tabpfn_results.json", "tabr_results.json", "tabm_results.json"):
            payload = json.loads((run_dir / name).read_text())
            dataset = payload["dataset"]
            metrics = payload["metrics"]
            rows.append(
                {
                    "dataset": dataset["dataset_key"],
                    "repeat": dataset["repeat"],
                    "fold": dataset["fold"],
                    "model": payload["model"],
                    "test_rmse": metrics["test"]["rmse"],
                    "val_rmse": metrics["val"]["rmse"],
                    "test_mae": metrics["test"]["mae"],
                    "test_r2": metrics["test"]["r2"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.reports_root, include_smoke=args.include_smoke)
    if not rows:
        raise SystemExit(f"No run JSON files found under {args.reports_root}")

    df = pd.DataFrame(rows).sort_values(["dataset", "model", "repeat", "fold"])
    raw_path = args.reports_root / "openml_benchmark_raw.csv"
    df.to_csv(raw_path, index=False)

    summary = (
        df.groupby(["dataset", "model"], as_index=False)
        .agg(
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_test_mae=("test_mae", "mean"),
            mean_test_r2=("test_r2", "mean"),
            runs=("fold", "count"),
        )
        .sort_values(["dataset", "mean_test_rmse", "model"])
    )
    summary["std_test_rmse"] = summary["std_test_rmse"].fillna(0.0)
    summary_path = args.reports_root / "openml_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    lines = ["# OpenML Regression Benchmark Summary", ""]
    for dataset, dataset_df in summary.groupby("dataset"):
        lines.extend([f"## {dataset}", "", "| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in dataset_df.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.mean_test_rmse:.4f} | {row.std_test_rmse:.4f} | "
                f"{row.mean_val_rmse:.4f} | {row.mean_test_mae:.4f} | {row.mean_test_r2:.4f} | {int(row.runs)} |"
            )
        lines.append("")

        dataset_rows = df[df["dataset"] == dataset].pivot_table(
            index=["repeat", "fold"],
            columns="model",
            values="test_rmse",
        )
        dataset_rows = dataset_rows.sort_index()
        lines.extend(["### Per Fold Test RMSE", "", dataframe_to_markdown(dataset_rows.reset_index()), ""])

        if {"GraphDrone", "TabR", "TabM", "TabPFN"} <= set(dataset_rows.columns):
            router = dataset_rows["GraphDrone"]
            deltas = {
                "vs TabR": float((router - dataset_rows["TabR"]).mean()),
                "vs TabM": float((router - dataset_rows["TabM"]).mean()),
                "vs TabPFN": float((router - dataset_rows["TabPFN"]).mean()),
            }
            lines.extend(["### GraphDrone Deltas", ""])
            for label, value in deltas.items():
                lines.append(f"- {label}: `{value:+.4f}` mean RMSE delta")
            lines.append("")

    (args.reports_root / "openml_benchmark_summary.md").write_text("\n".join(lines) + "\n")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float) and np.isfinite(value):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append(cells)
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


if __name__ == "__main__":
    main()
