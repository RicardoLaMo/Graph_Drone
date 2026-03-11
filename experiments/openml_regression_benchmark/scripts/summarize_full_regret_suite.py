from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def summarize(root: Path, *, adaptive_prefix: str = "router") -> dict[str, object]:
    paths = sorted(root.glob(f"**/artifacts/{adaptive_prefix}_full_regret_summary.json"))
    runs: list[dict[str, object]] = []
    for path in paths:
        runs.append(json.loads(path.read_text()))
    if not runs:
        raise FileNotFoundError(f"No {adaptive_prefix}_full_regret_summary.json files found under {root}")

    adaptive_minus_full = [run["global"]["adaptive_minus_full_test_rmse"] for run in runs]
    adaptive_minus_fixed = [run["global"]["adaptive_minus_fixed_test_rmse"] for run in runs]
    full_oracle_fraction = [run["global"]["full_oracle_fraction"] for run in runs]
    false_diversion_cost = [run["full_oracle_case"]["false_diversion_mean_cost"] for run in runs]
    false_diversion_fraction = [run["full_oracle_case"]["false_diversion_positive_fraction"] for run in runs]
    capture_ratio = [run["non_full_oracle_case"]["adaptive_capture_ratio_total"] for run in runs]
    capture_vs_fixed = [
        run["non_full_oracle_case"]["adaptive_capture_ratio_total"]
        - run["non_full_oracle_case"]["fixed_capture_ratio_total"]
        for run in runs
    ]
    missed_opportunity = [run["non_full_oracle_case"]["missed_opportunity_mean_cost"] for run in runs]

    return {
        "root": str(root),
        "adaptive_prefix": adaptive_prefix,
        "n_runs": len(runs),
        "run_dirs": [run["run_dir"] for run in runs],
        "adaptive_minus_full_test_rmse": {
            **_mean_std(adaptive_minus_full),
            "positive_fraction": float((np.asarray(adaptive_minus_full) > 0.0).mean()),
        },
        "adaptive_minus_fixed_test_rmse": {
            **_mean_std(adaptive_minus_fixed),
            "positive_fraction": float((np.asarray(adaptive_minus_fixed) > 0.0).mean()),
        },
        "full_oracle_fraction": _mean_std(full_oracle_fraction),
        "false_diversion_mean_cost": _mean_std(false_diversion_cost),
        "false_diversion_positive_fraction": _mean_std(false_diversion_fraction),
        "adaptive_capture_ratio_total": _mean_std(capture_ratio),
        "adaptive_capture_minus_fixed": {
            **_mean_std(capture_vs_fixed),
            "positive_fraction": float((np.asarray(capture_vs_fixed) > 0.0).mean()),
        },
        "missed_opportunity_mean_cost": _mean_std(missed_opportunity),
    }


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# FULL Regret Suite Summary",
        "",
        f"- root: `{summary['root']}`",
        f"- adaptive_prefix: `{summary['adaptive_prefix']}`",
        f"- n_runs: `{summary['n_runs']}`",
        f"- adaptive minus FULL RMSE: `{summary['adaptive_minus_full_test_rmse']['mean']:.6f} ± {summary['adaptive_minus_full_test_rmse']['std']:.6f}`",
        f"- adaptive minus fixed RMSE: `{summary['adaptive_minus_fixed_test_rmse']['mean']:.6f} ± {summary['adaptive_minus_fixed_test_rmse']['std']:.6f}`",
        f"- full oracle fraction: `{summary['full_oracle_fraction']['mean']:.6f} ± {summary['full_oracle_fraction']['std']:.6f}`",
        f"- false diversion mean cost: `{summary['false_diversion_mean_cost']['mean']:.6f} ± {summary['false_diversion_mean_cost']['std']:.6f}`",
        f"- adaptive capture ratio total: `{summary['adaptive_capture_ratio_total']['mean']:.6f} ± {summary['adaptive_capture_ratio_total']['std']:.6f}`",
        f"- adaptive capture minus fixed: `{summary['adaptive_capture_minus_fixed']['mean']:.6f} ± {summary['adaptive_capture_minus_fixed']['std']:.6f}`",
        f"- missed opportunity mean cost: `{summary['missed_opportunity_mean_cost']['mean']:.6f} ± {summary['missed_opportunity_mean_cost']['std']:.6f}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate FULL-vs-router diagnosis summaries over many runs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    args = parser.parse_args()

    summary = summarize(args.root, adaptive_prefix=args.adaptive_prefix)
    out_json = args.root / f"{args.adaptive_prefix}_full_regret_suite_summary.json"
    out_md = args.root / f"{args.adaptive_prefix}_full_regret_suite_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
