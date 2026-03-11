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
from experiments.tabpfn_view_router.src.data import build_quality_features
from experiments.tabpfn_view_router.src.runtime import build_device_plan
from src.graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig
from src.graphdrone_fit.metrics import regression_metrics
from experiments.openml_regression_benchmark.src.graphdrone_fit_adapter import build_benchmark_quality_encodings


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
    quality = build_quality_features(split, views, k=24)
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
    quality_encodings = build_benchmark_quality_encodings(views, quality)
    model = GraphDrone(
        GraphDroneConfig(
            portfolio=None,
            full_expert_id=expert_plan.full_expert_id,
            router=SetRouterConfig(kind="contextual_sparse_mlp", sparse_top_k=2),
        )
    )
    model.fit(split.X_train, split.y_train, expert_specs=expert_plan.specs)
    router_fit_summary = model.fit_router(
        split.X_val,
        split.y_val,
        quality_features=quality_encodings["val"],
    )

    batch_val = model.predict_experts(split.X_val)
    batch_test = model.predict_experts(split.X_test)
    graphdrone_val = model.predict(
        split.X_val,
        quality_features=quality_encodings["val"],
        return_diagnostics=True,
    )
    graphdrone_test = model.predict(
        split.X_test,
        quality_features=quality_encodings["test"],
        return_diagnostics=True,
    )

    internal_rows: list[dict[str, object]] = []
    for idx, expert_id in enumerate(batch_val.expert_ids):
        val_metrics = regression_metrics(split.y_val, batch_val.predictions[:, idx])
        test_metrics = regression_metrics(split.y_test, batch_test.predictions[:, idx])
        internal_rows.append(
            {
                "expert_id": expert_id,
                "view_name": expert_plan.expert_view_map.get(expert_id, expert_id),
                "val_rmse": val_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "notes": "internal expert diagnostics only; not a public benchmark row",
            }
        )

    val_metrics = regression_metrics(split.y_val, graphdrone_val.predictions)
    test_metrics = regression_metrics(split.y_test, graphdrone_test.predictions)
    rows = [
        {
            "model": "GraphDrone",
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "notes": "contextual_sparse_mlp over generic internal experts",
        }
    ]

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
        "expert_view_map": expert_plan.expert_view_map,
        "rows": rows,
        "graphdrone_test_diagnostics": graphdrone_test.diagnostics,
        "graphdrone_val_diagnostics": graphdrone_val.diagnostics,
        "router_fit_summary": router_fit_summary,
    }
    internal_payload = {
        "run_name": run_name,
        "dataset": split_summary(split),
        "rows": internal_rows,
        "expert_ids": list(batch_test.expert_ids),
        "expert_view_map": expert_plan.expert_view_map,
        "descriptors": [descriptor.to_dict() for descriptor in expert_plan.descriptors],
    }
    (output_dir / "graphdrone_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "graphdrone_internal_experts.json").write_text(json.dumps(internal_payload, indent=2) + "\n")
    (output_dir / "graphdrone_fit_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
