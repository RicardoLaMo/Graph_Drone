#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.src.openml_regression import BENCHMARK_SPECS, run_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OpenML regression benchmark results.")
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARK_SPECS))
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports" / "openml_regression_summary.md",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _graphdrone_path(root: Path, dataset_key: str, repeat: int, fold: int, smoke: bool) -> Path:
    slug = run_slug(dataset_key, repeat=repeat, fold=fold, smoke=smoke)
    return root / dataset_key / f"graphdrone__{slug}" / "graphdrone_results.json"


def _model_path(root: Path, dataset_key: str, prefix: str, repeat: int, fold: int, smoke: bool) -> Path:
    slug = run_slug(dataset_key, repeat=repeat, fold=fold, smoke=smoke)
    return root / dataset_key / f"{prefix}__{slug}.json"


def _mean_std(values: list[float]) -> tuple[float, float]:
    return st.mean(values), st.pstdev(values) if len(values) > 1 else 0.0


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    reports_root = REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports"
    lines = ["# OpenML Regression Benchmark Summary", ""]

    for dataset_key in args.datasets:
        spec = BENCHMARK_SPECS[dataset_key]
        graphdrone_rows: dict[str, list[tuple[int, float]]] = {}
        tabr_rows: list[tuple[int, float]] = []
        tabm_rows: list[tuple[int, float]] = []
        tabpfn_rows: list[tuple[int, float]] = []

        for fold in args.folds:
            graphdrone_payload = _load_json(_graphdrone_path(reports_root, dataset_key, args.repeat, fold, args.smoke))
            for row in graphdrone_payload["results"]:
                graphdrone_rows.setdefault(row["model"], []).append((fold, float(row["test_rmse"])))
            tabr_rows.append((fold, float(_load_json(_model_path(reports_root, dataset_key, "tabr", args.repeat, fold, args.smoke))["metrics"]["test"]["rmse"])))
            tabm_rows.append((fold, float(_load_json(_model_path(reports_root, dataset_key, "tabm", args.repeat, fold, args.smoke))["metrics"]["test"]["rmse"])))
            tabpfn_rows.append((fold, float(_load_json(_model_path(reports_root, dataset_key, "tabpfn", args.repeat, fold, args.smoke))["metrics"]["test"]["rmse"])))

        lines.extend(
            [
                f"## {spec.display_name}",
                "",
                f"- OpenML dataset id: `{spec.dataset_id}`",
                f"- OpenML task id: `{spec.task_id}`",
                "",
                "| Fold | GraphDrone_router | GraphDrone_crossfit | GraphDrone_FULL | TabPFN | TabR | TabM |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        router_lookup = dict(graphdrone_rows.get("GraphDrone_router", []))
        crossfit_lookup = dict(graphdrone_rows.get("GraphDrone_crossfit", []))
        full_lookup = dict(graphdrone_rows.get("GraphDrone_FULL", []))
        tabr_lookup = dict(tabr_rows)
        tabm_lookup = dict(tabm_rows)
        tabpfn_lookup = dict(tabpfn_rows)
        for fold in args.folds:
            lines.append(
                f"| {fold} | {_fmt(router_lookup.get(fold, float('nan')))} | {_fmt(crossfit_lookup.get(fold, float('nan')))} | {_fmt(full_lookup.get(fold, float('nan')))} | {_fmt(tabpfn_lookup.get(fold, float('nan')))} | {_fmt(tabr_lookup.get(fold, float('nan')))} | {_fmt(tabm_lookup.get(fold, float('nan')))} |"
            )

        lines.extend(["", "| Model | Mean RMSE | Std |", "|---|---:|---:|"])
        for label, rows in [
            ("GraphDrone_router", graphdrone_rows.get("GraphDrone_router", [])),
            ("GraphDrone_crossfit", graphdrone_rows.get("GraphDrone_crossfit", [])),
            ("GraphDrone_FULL", graphdrone_rows.get("GraphDrone_FULL", [])),
            ("TabPFN", tabpfn_rows),
            ("TabR", tabr_rows),
            ("TabM", tabm_rows),
        ]:
            values = [value for _, value in rows]
            mean, std = _mean_std(values)
            lines.append(f"| {label} | {mean:.4f} | {std:.4f} |")

        router_values = graphdrone_rows.get("GraphDrone_router", [])
        wins_vs_tabr = sum(1 for fold, value in router_values if value < tabr_lookup[fold])
        wins_vs_tabpfn = sum(1 for fold, value in router_values if value < tabpfn_lookup[fold])
        wins_vs_tabm = sum(1 for fold, value in router_values if value < tabm_lookup[fold])
        lines.extend(
            [
                "",
                "Notes:",
                f"- `GraphDrone_router` beats `TabR` on `{wins_vs_tabr}/{len(router_values)}` folds.",
                f"- `GraphDrone_router` beats `TabPFN` on `{wins_vs_tabpfn}/{len(router_values)}` folds.",
                f"- `GraphDrone_router` beats `TabM` on `{wins_vs_tabm}/{len(router_values)}` folds.",
                "",
            ]
        )

    markdown = "\n".join(lines) + "\n"
    args.output.write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
