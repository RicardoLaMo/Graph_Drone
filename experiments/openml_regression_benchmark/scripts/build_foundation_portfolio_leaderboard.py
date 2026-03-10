from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_ROOT = REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports_foundation_view_00e5903_full"
MODEL_COLUMNS = [
    "GraphDrone_FULL",
    "GraphDrone_FOUNDATION",
    "GraphDrone_router",
    "GraphDrone_crossfit",
    "GraphDrone_foundation_router",
    "GraphDrone_foundation_crossfit",
    "GraphDrone_trust_gate",
    "AutoGluon",
    "TabPFN",
    "TabR",
    "TabM",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a wide leaderboard for the GraphDrone foundation-view benchmark")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to openml_benchmark_summary.csv. Defaults to <reports-root>/openml_benchmark_summary.csv",
    )
    parser.add_argument(
        "--extra-summary-csv",
        type=Path,
        nargs="*",
        default=(),
        help="Optional additional summary CSV files to merge in before building the leaderboard",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output path for the markdown leaderboard. Defaults to <reports-root>/graphdrone_foundation_portfolio_leaderboard.md",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output path for the wide CSV leaderboard. Defaults to <reports-root>/graphdrone_foundation_portfolio_leaderboard.csv",
    )
    return parser.parse_args()


def build_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(index="dataset", columns="model", values="mean_test_rmse")
    for model in MODEL_COLUMNS:
        if model not in pivot.columns:
            pivot[model] = pd.NA
    pivot = pivot[MODEL_COLUMNS].sort_index()
    pivot["Best"] = pivot[MODEL_COLUMNS].astype(float).idxmin(axis=1)
    return pivot.reset_index().rename(columns={"dataset": "Dataset"})


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# GraphDrone Foundation-View Portfolio Leaderboard",
        "",
        "Mean test RMSE by dataset across the foundation-view OpenML benchmark.",
        "",
        "| Dataset | GraphDrone_FULL | GraphDrone_FOUNDATION | GraphDrone_router | GraphDrone_crossfit | GraphDrone_foundation_router | GraphDrone_foundation_crossfit | GraphDrone_trust_gate | AutoGluon | TabPFN | TabR | TabM | Best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in df.itertuples(index=False):
        values = [row.Dataset]
        for column in MODEL_COLUMNS:
            value = getattr(row, column)
            if pd.isna(value):
                values.append("-")
            else:
                values.append(f"{float(value):.4f}")
        values.append(row.Best)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv or (args.reports_root / "openml_benchmark_summary.csv")
    output_md = args.output_md or (args.reports_root / "graphdrone_foundation_portfolio_leaderboard.md")
    output_csv = args.output_csv or (args.reports_root / "graphdrone_foundation_portfolio_leaderboard.csv")

    summary = pd.read_csv(summary_csv)
    for extra_path in args.extra_summary_csv:
        extra_df = pd.read_csv(extra_path)
        summary = pd.concat([summary, extra_df], ignore_index=True)
    summary = summary.sort_values(["dataset", "model"]).drop_duplicates(["dataset", "model"], keep="last")
    leaderboard = build_leaderboard(summary)
    leaderboard.to_csv(output_csv, index=False)
    output_md.write_text(to_markdown(leaderboard))


if __name__ == "__main__":
    main()
