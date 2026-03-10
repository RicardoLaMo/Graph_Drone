from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_ROOT = REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports_foundation_view_00e5903_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze foundation-router weight usage from saved GraphDrone artifacts")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <reports-root>/foundation_router_usage_summary.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output markdown path. Defaults to <reports-root>/foundation_router_usage_summary.md",
    )
    return parser.parse_args()


def decode_names(values: np.ndarray) -> list[str]:
    names: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def mean_by_name(weights: np.ndarray, names: list[str]) -> dict[str, float]:
    return {name: float(weights[:, idx].mean()) for idx, name in enumerate(names)}


def top1_share_by_name(weights: np.ndarray, names: list[str]) -> dict[str, float]:
    winners = np.argmax(weights, axis=1)
    return {name: float((winners == idx).mean()) for idx, name in enumerate(names)}


def best_internal_name(metric_by_name: dict[str, float]) -> str:
    internal = {name: value for name, value in metric_by_name.items() if name != "FOUNDATION"}
    return max(internal, key=internal.get) if internal else "-"


def load_model_rmse(run_dir: Path) -> dict[str, float]:
    payload = json.loads((run_dir / "graphdrone_results.json").read_text())
    return {str(row["model"]): float(row["test_rmse"]) for row in payload["rows"]}


def load_rows(reports_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(path.parent for path in reports_root.rglob("graphdrone_results.json")):
        artifact_path = run_dir / "artifacts" / "graphdrone_predictions.npz"
        if not artifact_path.exists():
            continue
        metrics = load_model_rmse(run_dir)
        data = np.load(artifact_path)
        names = decode_names(data["foundation_view_names"])

        foundation_router_weights = np.asarray(data["foundation_router_weights_test"], dtype=float)
        foundation_crossfit_weights = np.asarray(data["foundation_crossfit_weights_test"], dtype=float)
        router_mean = mean_by_name(foundation_router_weights, names)
        crossfit_mean = mean_by_name(foundation_crossfit_weights, names)
        router_top1 = top1_share_by_name(foundation_router_weights, names)
        crossfit_top1 = top1_share_by_name(foundation_crossfit_weights, names)

        dataset = run_dir.parent.name
        fold = int(run_dir.name.rsplit("f", 1)[1])
        rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "foundation_rmse": metrics["GraphDrone_FOUNDATION"],
                "foundation_router_rmse": metrics["GraphDrone_foundation_router"],
                "foundation_crossfit_rmse": metrics["GraphDrone_foundation_crossfit"],
                "router_delta_vs_foundation": metrics["GraphDrone_foundation_router"] - metrics["GraphDrone_FOUNDATION"],
                "crossfit_delta_vs_foundation": metrics["GraphDrone_foundation_crossfit"] - metrics["GraphDrone_FOUNDATION"],
                "router_foundation_weight": router_mean.get("FOUNDATION", float("nan")),
                "router_foundation_top1_share": router_top1.get("FOUNDATION", float("nan")),
                "router_best_internal_weight_name": best_internal_name(router_mean),
                "router_best_internal_weight": max(
                    (value for name, value in router_mean.items() if name != "FOUNDATION"),
                    default=float("nan"),
                ),
                "crossfit_foundation_weight": crossfit_mean.get("FOUNDATION", float("nan")),
                "crossfit_foundation_top1_share": crossfit_top1.get("FOUNDATION", float("nan")),
                "crossfit_best_internal_weight_name": best_internal_name(crossfit_mean),
                "crossfit_best_internal_weight": max(
                    (value for name, value in crossfit_mean.items() if name != "FOUNDATION"),
                    default=float("nan"),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "fold"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("dataset", as_index=False)
        .agg(
            foundation_rmse=("foundation_rmse", "mean"),
            foundation_router_rmse=("foundation_router_rmse", "mean"),
            foundation_crossfit_rmse=("foundation_crossfit_rmse", "mean"),
            router_delta_vs_foundation=("router_delta_vs_foundation", "mean"),
            crossfit_delta_vs_foundation=("crossfit_delta_vs_foundation", "mean"),
            router_foundation_weight=("router_foundation_weight", "mean"),
            router_foundation_top1_share=("router_foundation_top1_share", "mean"),
            crossfit_foundation_weight=("crossfit_foundation_weight", "mean"),
            crossfit_foundation_top1_share=("crossfit_foundation_top1_share", "mean"),
            router_best_internal_weight_name=("router_best_internal_weight_name", lambda s: s.value_counts().index[0]),
            crossfit_best_internal_weight_name=("crossfit_best_internal_weight_name", lambda s: s.value_counts().index[0]),
        )
        .sort_values("dataset")
    )
    return grouped


def format_float(value: float) -> str:
    return "-" if pd.isna(value) else f"{float(value):.4f}"


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Foundation Router Usage Summary",
        "",
        "Mean test-time routing behavior aggregated across folds.",
        "",
        "| Dataset | Foundation | Foundation Router | Router Delta | Router Foundation Weight | Router Foundation Top1 | Router Best Internal | Foundation Crossfit | Crossfit Delta | Crossfit Foundation Weight | Crossfit Foundation Top1 | Crossfit Best Internal |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.dataset),
                    format_float(row.foundation_rmse),
                    format_float(row.foundation_router_rmse),
                    format_float(row.router_delta_vs_foundation),
                    format_float(row.router_foundation_weight),
                    format_float(row.router_foundation_top1_share),
                    str(row.router_best_internal_weight_name),
                    format_float(row.foundation_crossfit_rmse),
                    format_float(row.crossfit_delta_vs_foundation),
                    format_float(row.crossfit_foundation_weight),
                    format_float(row.crossfit_foundation_top1_share),
                    str(row.crossfit_best_internal_weight_name),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or (args.reports_root / "foundation_router_usage_summary.csv")
    output_md = args.output_md or (args.reports_root / "foundation_router_usage_summary.md")
    rows = load_rows(args.reports_root)
    if rows.empty:
        raise SystemExit(f"No graphdrone_predictions.npz artifacts found under {args.reports_root}")
    summary = summarize(rows)
    summary.to_csv(output_csv, index=False)
    output_md.write_text(to_markdown(summary))


if __name__ == "__main__":
    main()
