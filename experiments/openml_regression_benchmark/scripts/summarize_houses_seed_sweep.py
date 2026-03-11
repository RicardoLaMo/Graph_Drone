from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGET_MODELS = [
    "GraphDrone_FULL",
    "GraphDrone_router",
    "GraphDrone_router_fixed",
    "GraphDrone_crossfit",
    "GraphDrone_crossfit_fixed",
]


def load_rows(run_dir: Path) -> dict[str, dict[str, object]]:
    payload = json.loads((run_dir / "graphdrone_results.json").read_text())
    return {row["model"]: row for row in payload["rows"]}


def summarize(root: Path) -> dict[str, object]:
    run_dirs = sorted(path.parent for path in root.glob("**/graphdrone_results.json"))
    if not run_dirs:
        raise FileNotFoundError(f"No houses seed runs found under {root}")

    table: dict[str, list[float]] = {name: [] for name in TARGET_MODELS}
    deltas_router: list[float] = []
    deltas_crossfit: list[float] = []
    by_run: list[dict[str, object]] = []
    for run_dir in run_dirs:
        rows = load_rows(run_dir)
        for name in TARGET_MODELS:
            table[name].append(float(rows[name]["test_rmse"]))
        deltas_router.append(float(rows["GraphDrone_router_fixed"]["test_rmse"] - rows["GraphDrone_router"]["test_rmse"]))
        deltas_crossfit.append(
            float(rows["GraphDrone_crossfit_fixed"]["test_rmse"] - rows["GraphDrone_crossfit"]["test_rmse"])
        )
        by_run.append(
            {
                "run_dir": str(run_dir),
                "GraphDrone_router": float(rows["GraphDrone_router"]["test_rmse"]),
                "GraphDrone_router_fixed": float(rows["GraphDrone_router_fixed"]["test_rmse"]),
                "GraphDrone_crossfit": float(rows["GraphDrone_crossfit"]["test_rmse"]),
                "GraphDrone_crossfit_fixed": float(rows["GraphDrone_crossfit_fixed"]["test_rmse"]),
            }
        )

    return {
        "root": str(root),
        "n_runs": len(run_dirs),
        "per_model": {
            name: {
                "mean_test_rmse": float(np.mean(values)),
                "std_test_rmse": float(np.std(values, ddof=0)),
            }
            for name, values in table.items()
        },
        "router_adaptive_minus_fixed": {
            "mean": float(np.mean(deltas_router)),
            "std": float(np.std(deltas_router, ddof=0)),
            "positive_fraction": float((np.asarray(deltas_router) > 0.0).mean()),
        },
        "crossfit_adaptive_minus_fixed": {
            "mean": float(np.mean(deltas_crossfit)),
            "std": float(np.std(deltas_crossfit, ddof=0)),
            "positive_fraction": float((np.asarray(deltas_crossfit) > 0.0).mean()),
        },
        "runs": by_run,
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Houses Seed Sweep",
        "",
        f"- root: `{summary['root']}`",
        f"- n_runs: `{summary['n_runs']}`",
        "",
        "## Per-Model Test RMSE",
        "",
    ]
    for name, stats in summary["per_model"].items():
        lines.append(f"- {name}: mean `{stats['mean_test_rmse']:.4f}` / std `{stats['std_test_rmse']:.4f}`")
    lines.extend(
        [
            "",
            "## Adaptive Minus Fixed",
            "",
            f"- router: mean `{summary['router_adaptive_minus_fixed']['mean']:.6f}` / std `{summary['router_adaptive_minus_fixed']['std']:.6f}` / positive fraction `{summary['router_adaptive_minus_fixed']['positive_fraction']:.3f}`",
            f"- crossfit: mean `{summary['crossfit_adaptive_minus_fixed']['mean']:.6f}` / std `{summary['crossfit_adaptive_minus_fixed']['std']:.6f}` / positive fraction `{summary['crossfit_adaptive_minus_fixed']['positive_fraction']:.3f}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize houses seed sweep stability for adaptive vs fixed routing")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(args.root)
    out_json = args.root / "houses_seed_sweep_summary.json"
    out_md = args.root / "houses_seed_sweep_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_summary(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
