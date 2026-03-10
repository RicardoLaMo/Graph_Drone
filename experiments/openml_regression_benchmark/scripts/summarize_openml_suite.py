from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.src.openml_tasks import dataset_run_tag

RESULT_FILES = (
    "graphdrone_results.json",
    "tabpfn_results.json",
    "autogluon_results.json",
    "tabr_results.json",
    "tabm_results.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OpenML regression benchmark runs")
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports",
    )
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def discover_run_dirs(reports_root: Path) -> list[Path]:
    manifest_path = reports_root / "suite_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        run_dirs: list[Path] = []
        seen: set[Path] = set()
        for row in payload:
            run_name = row.get(
                "run_name",
                dataset_run_tag(
                    str(row["dataset"]),
                    repeat=int(row.get("repeat", 0)),
                    fold=int(row["fold"]),
                    smoke=bool(row.get("smoke", False)),
                ),
            )
            run_dir = reports_root / str(run_name)
            if run_dir in seen:
                continue
            seen.add(run_dir)
            run_dirs.append(run_dir)
        return sorted(run_dirs)
    return sorted(
        {
            path.parent
            for result_name in RESULT_FILES
            for path in reports_root.rglob(result_name)
        }
    )


def load_rows(reports_root: Path, *, include_smoke: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_dirs = discover_run_dirs(reports_root)
    for run_dir in run_dirs:
        if not include_smoke and any(part.endswith("__smoke") for part in run_dir.parts):
            continue
        if not run_dir.exists():
            continue
        present = {path.name for path in run_dir.glob("*.json")}
        if "graphdrone_results.json" in present:
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

        for name in ("tabpfn_results.json", "autogluon_results.json", "tabr_results.json", "tabm_results.json"):
            if name not in present:
                continue
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

        if {"GraphDrone_router", "TabR", "TabM", "TabPFN"} <= set(dataset_rows.columns):
            router = dataset_rows["GraphDrone_router"]
            deltas = {
                "vs TabR": float((router - dataset_rows["TabR"]).mean()),
                "vs TabM": float((router - dataset_rows["TabM"]).mean()),
                "vs TabPFN": float((router - dataset_rows["TabPFN"]).mean()),
            }
            if "AutoGluon" in dataset_rows.columns:
                deltas["vs AutoGluon"] = float((router - dataset_rows["AutoGluon"]).mean())
            lines.extend(["### GraphDrone Router Deltas", ""])
            for label, value in deltas.items():
                lines.append(f"- {label}: `{value:+.4f}` mean RMSE delta")
            lines.append("")
        else:
            lines.extend(["### GraphDrone Router Deltas", "", "- skipped: missing one or more baseline columns", ""])

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
