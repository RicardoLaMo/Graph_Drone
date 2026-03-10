from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.src.openml_tasks import (
    available_dataset_keys,
    build_graphdrone_view_data,
    build_openml_regression_split,
    dataset_run_tag,
    limit_train_rows,
    split_summary,
)
from experiments.tabpfn_view_router.scripts.run_experiment import fit_view_experts
from experiments.tabpfn_view_router.src.data import build_quality_features
from experiments.tabpfn_view_router.src.router import (
    fit_crossfit_router,
    fit_soft_router,
    gora_mix,
    score_regression,
    sigma2_mix,
    uniform_mix,
)
from experiments.tabpfn_view_router.src.runtime import build_device_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphDrone series on OpenML regression task splits")
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
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports",
    )
    return parser.parse_args()


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    output_path: Path,
    *,
    run_name: str,
    rows: list[dict[str, object]],
    weights_summary: dict[str, list[float]],
    dataset_summary: dict[str, object],
    runtime_summary: dict[str, object],
) -> None:
    lines = [
        "# GraphDrone OpenML Report",
        "",
        f"- run: `{run_name}`",
        f"- dataset: `{dataset_summary['dataset_name']}` ({dataset_summary['dataset_key']})",
        f"- OpenML dataset id: `{dataset_summary['dataset_id']}`",
        f"- OpenML task id: `{dataset_summary['task_id']}`",
        f"- repeat / fold: `{dataset_summary['repeat']}` / `{dataset_summary['fold']}`",
        f"- rows train / val / test: `{dataset_summary['train_rows']}` / `{dataset_summary['val_rows']}` / `{dataset_summary['test_rows']}`",
        f"- feature counts num / cat / total: `{dataset_summary['num_features']}` / `{dataset_summary['cat_features']}` / `{len(dataset_summary['feature_names'])}`",
        f"- requested device: `{runtime_summary['requested_device']}`",
        f"- resolved device plan: `{runtime_summary['resolved_device']}`",
        f"- device mode: `{runtime_summary['device_mode']}`",
        f"- parallel workers: `{runtime_summary['parallel_workers']}`",
        f"- CUDA_VISIBLE_DEVICES: `{runtime_summary['cuda_visible_devices']}`",
        "",
        "## Results",
        "",
        "| Model | Test RMSE | Val RMSE | Test MAE | Test R2 | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['test_rmse']:.4f} | {row['val_rmse']:.4f} | "
            f"{row['test_mae']:.4f} | {row['test_r2']:.4f} | {row['notes']} |"
        )
    lines.extend(["", "## View Weights", ""])
    for name, values in weights_summary.items():
        lines.append(f"- {name}: val `{values[0]:.3f}` / test `{values[1]:.3f}`")
    output_path.write_text("\n".join(lines) + "\n")


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

    pred_val, pred_test, view_devices_used = fit_view_experts(
        views=views,
        split=split,
        seed=args.seed,
        n_estimators=args.n_estimators,
        view_devices=device_plan.view_devices,
        n_preprocessing_jobs=args.n_preprocessing_jobs,
        parallel_workers=device_plan.parallel_workers,
    )

    model_rows: list[dict[str, object]] = []
    weights_summary: dict[str, list[float]] = {}
    for idx, name in enumerate(views.view_names):
        val_metrics = score_regression(split.y_val, pred_val[:, idx])
        test_metrics = score_regression(split.y_test, pred_test[:, idx])
        model_rows.append(
            {
                "model": f"GraphDrone_{name}",
                "val_rmse": val_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "notes": "single-view TabPFN expert",
            }
        )

    pred_uniform_val, w_uniform_val = uniform_mix(pred_val)
    pred_uniform_test, w_uniform_test = uniform_mix(pred_test)
    val_metrics = score_regression(split.y_val, pred_uniform_val)
    test_metrics = score_regression(split.y_test, pred_uniform_test)
    model_rows.append(
        {
            "model": "GraphDrone_uniform",
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "notes": "uniform mean over view experts",
        }
    )
    weights_summary["uniform"] = [float(w_uniform_val.mean()), float(w_uniform_test.mean())]
    for idx, name in enumerate(views.view_names):
        weights_summary[f"uniform_{name}"] = [
            float(w_uniform_val[:, idx].mean()),
            float(w_uniform_test[:, idx].mean()),
        ]

    pred_sigma_val, w_sigma_val = sigma2_mix(pred_val, quality.sigma2_val)
    pred_sigma_test, w_sigma_test = sigma2_mix(pred_test, quality.sigma2_test)
    val_metrics = score_regression(split.y_val, pred_sigma_val)
    test_metrics = score_regression(split.y_test, pred_sigma_test)
    model_rows.append(
        {
            "model": "GraphDrone_sigma2",
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "notes": "inverse-sigma2 routing without val labels",
        }
    )
    weights_summary["sigma2"] = [float(w_sigma_val.mean()), float(w_sigma_test.mean())]
    for idx, name in enumerate(views.view_names):
        weights_summary[f"sigma2_{name}"] = [
            float(w_sigma_val[:, idx].mean()),
            float(w_sigma_test[:, idx].mean()),
        ]

    pred_gora_val, w_gora_val = gora_mix(pred_val, quality.sigma2_val, quality.mean_j_val)
    pred_gora_test, w_gora_test = gora_mix(pred_test, quality.sigma2_test, quality.mean_j_test)
    val_metrics = score_regression(split.y_val, pred_gora_val)
    test_metrics = score_regression(split.y_test, pred_gora_test)
    model_rows.append(
        {
            "model": "GraphDrone_gora",
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "notes": "analytical GoRA-style routing without learned parameters",
        }
    )
    weights_summary["gora"] = [float(w_gora_val.mean()), float(w_gora_test.mean())]
    for idx, name in enumerate(views.view_names):
        weights_summary[f"gora_{name}"] = [
            float(w_gora_val[:, idx].mean()),
            float(w_gora_test[:, idx].mean()),
        ]

    router = fit_soft_router(
        x_val=quality.val,
        pred_val=pred_val,
        y_val=split.y_val,
        x_test=quality.test,
        pred_test=pred_test,
        seed=args.seed,
    )
    val_metrics = score_regression(split.y_val, router.pred_val)
    test_metrics = score_regression(split.y_test, router.pred_test)
    model_rows.append(
        {
            "model": "GraphDrone_router",
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "notes": f"softmax router on sigma2 + J features (best_epoch={router.best_epoch})",
        }
    )
    weights_summary["router"] = [float(router.weights_val.mean()), float(router.weights_test.mean())]
    for idx, name in enumerate(views.view_names):
        weights_summary[f"router_{name}"] = [
            float(router.weights_val[:, idx].mean()),
            float(router.weights_test[:, idx].mean()),
        ]

    crossfit = fit_crossfit_router(
        x_val=quality.val,
        pred_val=pred_val,
        y_val=split.y_val,
        x_test=quality.test,
        pred_test=pred_test,
        seed=args.seed,
    )
    val_metrics = score_regression(split.y_val, crossfit.pred_val_oof)
    test_metrics = score_regression(split.y_test, crossfit.pred_test)
    model_rows.append(
        {
            "model": "GraphDrone_crossfit",
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "notes": f"{crossfit.n_splits}-fold OOF router on val with final test refit",
        }
    )
    weights_summary["crossfit"] = [
        float(crossfit.weights_val_oof.mean()),
        float(crossfit.weights_test.mean()),
    ]
    for idx, name in enumerate(views.view_names):
        weights_summary[f"crossfit_{name}"] = [
            float(crossfit.weights_val_oof[:, idx].mean()),
            float(crossfit.weights_test[:, idx].mean()),
        ]

    run_name = dataset_run_tag(args.dataset, repeat=args.repeat, fold=args.fold, smoke=args.smoke)
    output_dir = (args.output_root / run_name).resolve()
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        artifacts_dir / "graphdrone_predictions.npz",
        view_names=np.array(views.view_names),
        y_val=split.y_val,
        y_test=split.y_test,
        pred_val=pred_val,
        pred_test=pred_test,
        sigma2_val=quality.sigma2_val,
        sigma2_test=quality.sigma2_test,
        mean_j_val=quality.mean_j_val,
        mean_j_test=quality.mean_j_test,
        router_pred_val=router.pred_val,
        router_pred_test=router.pred_test,
        router_weights_val=router.weights_val,
        router_weights_test=router.weights_test,
        crossfit_pred_val=crossfit.pred_val_oof,
        crossfit_pred_test=crossfit.pred_test,
        crossfit_weights_val=crossfit.weights_val_oof,
        crossfit_weights_test=crossfit.weights_test,
        uniform_weights_val=w_uniform_val,
        uniform_weights_test=w_uniform_test,
        sigma2_weights_val=w_sigma_val,
        sigma2_weights_test=w_sigma_test,
        gora_weights_val=w_gora_val,
        gora_weights_test=w_gora_test,
    )
    write_metrics_csv(artifacts_dir / "metrics.csv", model_rows)
    payload = {
        "run_name": run_name,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "dataset": split_summary(split),
        "runtime": {
            "requested_device": args.device,
            "resolved_device": device_plan.resolved_device,
            "device_mode": args.device_mode,
            "parallel_workers": device_plan.parallel_workers,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "view_devices": view_devices_used,
            "max_train_samples": args.max_train_samples,
        },
        "rows": model_rows,
        "weights_summary": weights_summary,
    }
    (output_dir / "graphdrone_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(
        output_dir / "report.md",
        run_name=run_name,
        rows=model_rows,
        weights_summary=weights_summary,
        dataset_summary=payload["dataset"],
        runtime_summary=payload["runtime"],
    )


if __name__ == "__main__":
    main()
