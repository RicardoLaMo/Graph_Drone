from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.src.graphdrone_fit_adapter import build_benchmark_expert_plan
from experiments.openml_regression_benchmark.src.openml_tasks import (
    available_dataset_keys,
    build_graphdrone_view_data,
    build_openml_regression_split,
    dataset_run_tag,
    limit_train_rows,
    split_summary,
)
from experiments.tabpfn_view_router.src.runtime import build_device_plan
from src.graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig
from src.graphdrone_fit.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase I-B GraphDrone.fit() benchmark runner")
    parser.add_argument("--dataset", choices=available_dataset_keys(), required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--device-mode", choices=["per_view", "per_model"], default="per_view")
    parser.add_argument("--all-gpus", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports_fit",
    )
    return parser.parse_args()


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    split = build_openml_regression_split(
        args.dataset,
        repeat=args.repeat,
        fold=args.fold,
        split_seed=args.split_seed,
        smoke=args.smoke,
    )
    split = limit_train_rows(
        split,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )
    views = build_graphdrone_view_data(split)
    device_plan = build_device_plan(
        views.view_names,
        requested_device=args.device,
        device_mode=args.device_mode,
        all_gpus=args.all_gpus,
        parallel_workers=args.parallel_workers,
    )
    expert_plan = build_benchmark_expert_plan(
        split,
        views,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_preprocessing_jobs=args.n_preprocessing_jobs,
        view_devices=device_plan.view_devices,
    )
    model = GraphDrone(
        GraphDroneConfig(
            portfolio=None,
            full_expert_id="FULL",
            router=SetRouterConfig(kind="bootstrap_full_only"),
        )
    )
    model.fit(split.X_train, split.y_train, expert_specs=expert_plan.specs)

    batch_val = model.predict_experts(split.X_val)
    batch_test = model.predict_experts(split.X_test)
    bootstrap_val = model.predict(split.X_val, return_diagnostics=True)
    bootstrap_test = model.predict(split.X_test, return_diagnostics=True)

    rows: list[dict[str, object]] = []
    for idx, expert_id in enumerate(batch_val.expert_ids):
        val_metrics = regression_metrics(split.y_val, batch_val.predictions[:, idx])
        test_metrics = regression_metrics(split.y_test, batch_test.predictions[:, idx])
        rows.append(
            {
                "model": f"GraphDroneFit_{expert_id}",
                "val_rmse": val_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "notes": "expert built through GraphDrone.fit() portfolio path",
            }
        )

    val_metrics = regression_metrics(split.y_val, bootstrap_val.predictions)
    test_metrics = regression_metrics(split.y_test, bootstrap_test.predictions)
    rows.append(
        {
            "model": "GraphDroneFit_bootstrap",
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "notes": "explicit bootstrap_full_only router",
        }
    )

    run_name = dataset_run_tag(args.dataset, repeat=args.repeat, fold=args.fold, smoke=args.smoke)
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(output_dir / "metrics.csv", rows)
    report = {
        "run_name": run_name,
        "dataset": split_summary(split),
        "runtime": {
            "requested_device": args.device,
            "resolved_device": str(device_plan.resolved_device),
            "device_mode": args.device_mode,
            "parallel_workers": device_plan.parallel_workers,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "expert_ids": list(batch_test.expert_ids),
        "rows": rows,
        "bootstrap_test_diagnostics": bootstrap_test.diagnostics,
    }
    (output_dir / "graphdrone_fit_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
