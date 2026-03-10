from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tabpfn_view_router.src.data import (
    OPENML_CALIFORNIA_DID,
    SEED,
    build_aligned_california_split,
    build_quality_features,
    build_view_data,
)
from experiments.tabpfn_view_router.src.runtime import build_device_plan, serialize_device_spec
from experiments.tabpfn_view_router.src.router import (
    CrossfitRouterResult,
    fit_crossfit_router,
    fit_soft_router,
    gora_mix,
    score_regression,
    sigma2_mix,
    uniform_mix,
)


TABR_RMSE = 0.3829
TABM_RMSE = 0.4290
TABPFN_FULL_RMSE = 0.3891
TABPFN_FULL_MEAN = 0.3932
A6F_ARTIFACT = 0.4063


def _fit_single_view(task: dict[str, object]) -> tuple[str, str | list[str], np.ndarray, np.ndarray]:
    from tabpfn import TabPFNRegressor

    name = str(task["name"])
    device = task["device"]
    model = TabPFNRegressor(
        n_estimators=int(task["n_estimators"]),
        random_state=int(task["seed"]),
        device=device,
        ignore_pretraining_limits=bool(task["ignore_pretraining_limits"]),
        n_preprocessing_jobs=int(task["n_preprocessing_jobs"]),
    )
    model.fit(task["x_train"], task["y_train"])
    pred_val = model.predict(task["x_val"]).astype(np.float32)
    pred_test = model.predict(task["x_test"]).astype(np.float32)
    return name, serialize_device_spec(device), pred_val, pred_test


def fit_view_experts(
    views,
    split,
    *,
    seed: int,
    n_estimators: int,
    view_devices: dict[str, str | list[str]],
    n_preprocessing_jobs: int,
    parallel_workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str | list[str]]]:
    tasks = [
        {
            "name": name,
            "x_train": views.train[name],
            "x_val": views.val[name],
            "x_test": views.test[name],
            "y_train": split.y_train,
            "seed": seed,
            "n_estimators": n_estimators,
            "device": view_devices[name],
            "ignore_pretraining_limits": len(views.train[name]) > 1000,
            "n_preprocessing_jobs": n_preprocessing_jobs,
        }
        for name in views.view_names
    ]

    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    used_devices: dict[str, str | list[str]] = {}

    if parallel_workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=ctx) as executor:
            future_map = {executor.submit(_fit_single_view, task): str(task["name"]) for task in tasks}
            for future in as_completed(future_map):
                name, device_used, pred_val, pred_test = future.result()
                outputs[name] = (pred_val, pred_test)
                used_devices[name] = device_used
    else:
        for task in tasks:
            name, device_used, pred_val, pred_test = _fit_single_view(task)
            outputs[name] = (pred_val, pred_test)
            used_devices[name] = device_used

    pred_val = np.stack([outputs[name][0] for name in views.view_names], axis=1).astype(np.float32)
    pred_test = np.stack([outputs[name][1] for name in views.view_names], axis=1).astype(np.float32)
    return pred_val, pred_test, used_devices


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: list[dict],
    weights_summary: dict[str, list[float]],
    seed: int,
    split_seed: int,
    n_estimators: int,
    dataset_summary: dict[str, object],
    runtime_summary: dict[str, object],
) -> None:
    lines = [
        "# P0 TabPFN View Router",
        "",
        f"- seed: `{seed}`",
        f"- split_seed: `{split_seed}`",
        f"- n_estimators per expert: `{n_estimators}`",
        f"- dataset source: `{dataset_summary['source']}`",
        f"- OpenML dataset id: `{dataset_summary['openml_dataset_id']}`",
        f"- requested device: `{runtime_summary['requested_device']}`",
        f"- resolved device plan: `{runtime_summary['resolved_device']}`",
        f"- device mode: `{runtime_summary['device_mode']}`",
        f"- parallel workers: `{runtime_summary['parallel_workers']}`",
        f"- CUDA_VISIBLE_DEVICES: `{runtime_summary['cuda_visible_devices']}`",
        "",
        "## Results",
        "",
        "| Model | Test RMSE | Val RMSE | Notes |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['test_rmse']:.4f} | {row['val_rmse']:.4f} | {row['notes']} |")
    lines.extend(
        [
            "",
            "## Reference Anchors",
            "",
            f"- TabR_on_our_split: `{TABR_RMSE:.4f}`",
            f"- TabM_on_our_split: `{TABM_RMSE:.4f}`",
            f"- TabPFN_full_best: `{TABPFN_FULL_RMSE:.4f}`",
            f"- TabPFN_full_multiseed_mean: `{TABPFN_FULL_MEAN:.4f}`",
            f"- MV-TabR-GoRA A6f artifact: `{A6F_ARTIFACT:.4f}`",
            "",
            "## View Device Assignment",
            "",
        ]
    )
    for name, device_spec in runtime_summary["view_devices"].items():
        lines.append(f"- {name}: `{device_spec}`")
    lines.extend(
        [
            "",
            "## Learned Router Mean Weights (val / test)",
            "",
        ]
    )
    for name, vals in weights_summary.items():
        lines.append(f"- {name}: val `{vals[0]:.3f}` / test `{vals[1]:.3f}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="P0: per-view TabPFN experts with GoRA-style view routing")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--dataset-source", choices=["sklearn", "openml"], default="sklearn")
    parser.add_argument("--openml-dataset-id", type=int, default=OPENML_CALIFORNIA_DID)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--device-mode", choices=["per_view", "per_model"], default="per_view")
    parser.add_argument("--all-gpus", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments/tabpfn_view_router/reports/p0_full",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split = build_aligned_california_split(
        seed=args.split_seed,
        smoke=args.smoke,
        dataset_source=args.dataset_source,
        openml_dataset_id=args.openml_dataset_id,
    )
    views = build_view_data(split)
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

    model_rows = []
    weights_summary: dict[str, list[float]] = {}

    view_order = views.view_names
    for i, name in enumerate(view_order):
        score_val = score_regression(split.y_val, pred_val[:, i])
        score_test = score_regression(split.y_test, pred_test[:, i])
        model_rows.append(
            {
                "model": f"P0_{name}",
                "val_rmse": score_val["rmse"],
                "test_rmse": score_test["rmse"],
                "val_mae": score_val["mae"],
                "test_mae": score_test["mae"],
                "notes": "single-view TabPFN expert",
            }
        )

    pred_uniform_val, w_uniform_val = uniform_mix(pred_val)
    pred_uniform_test, w_uniform_test = uniform_mix(pred_test)
    model_rows.append(
        {
            "model": "P0_uniform",
            "val_rmse": score_regression(split.y_val, pred_uniform_val)["rmse"],
            "test_rmse": score_regression(split.y_test, pred_uniform_test)["rmse"],
            "val_mae": score_regression(split.y_val, pred_uniform_val)["mae"],
            "test_mae": score_regression(split.y_test, pred_uniform_test)["mae"],
            "notes": "uniform mean over four TabPFN view experts",
        }
    )
    weights_summary["uniform"] = [float(w_uniform_val.mean()), float(w_uniform_test.mean())]

    pred_sigma_val, w_sigma_val = sigma2_mix(pred_val, quality.sigma2_val)
    pred_sigma_test, w_sigma_test = sigma2_mix(pred_test, quality.sigma2_test)
    model_rows.append(
        {
            "model": "P0_sigma2",
            "val_rmse": score_regression(split.y_val, pred_sigma_val)["rmse"],
            "test_rmse": score_regression(split.y_test, pred_sigma_test)["rmse"],
            "val_mae": score_regression(split.y_val, pred_sigma_val)["mae"],
            "test_mae": score_regression(split.y_test, pred_sigma_test)["mae"],
            "notes": "inverse-sigma2 routing — no val labels used",
        }
    )
    for i, name in enumerate(view_order):
        weights_summary[f"sigma2_{name}"] = [float(w_sigma_val[:, i].mean()), float(w_sigma_test[:, i].mean())]

    # Option A: GoRA analytical routing — zero free parameters, no val labels
    pred_gora_val, w_gora_val = gora_mix(pred_val, quality.sigma2_val, quality.mean_j_val)
    pred_gora_test, w_gora_test = gora_mix(pred_test, quality.sigma2_test, quality.mean_j_test)
    model_rows.append(
        {
            "model": "P0_gora",
            "val_rmse": score_regression(split.y_val, pred_gora_val)["rmse"],
            "test_rmse": score_regression(split.y_test, pred_gora_test)["rmse"],
            "val_mae": score_regression(split.y_val, pred_gora_val)["mae"],
            "test_mae": score_regression(split.y_test, pred_gora_test)["mae"],
            "notes": "GoRA analytical routing: softmax(-sigma2*tau), tau=1/(mean_J+eps) — zero params, no val labels",
        }
    )
    for i, name in enumerate(view_order):
        weights_summary[f"gora_{name}"] = [float(w_gora_val[:, i].mean()), float(w_gora_test[:, i].mean())]

    router = fit_soft_router(
        x_val=quality.val,
        pred_val=pred_val,
        y_val=split.y_val,
        x_test=quality.test,
        pred_test=pred_test,
        seed=args.seed,
    )
    model_rows.append(
        {
            "model": "P0_router",
            "val_rmse": score_regression(split.y_val, router.pred_val)["rmse"],
            "test_rmse": score_regression(split.y_test, router.pred_test)["rmse"],
            "val_mae": score_regression(split.y_val, router.pred_val)["mae"],
            "test_mae": score_regression(split.y_test, router.pred_test)["mae"],
            "notes": f"softmax router on sigma2_v + J_flat + mean_J (best_epoch={router.best_epoch})",
        }
    )
    for i, name in enumerate(view_order):
        weights_summary[f"router_{name}"] = [
            float(router.weights_val[:, i].mean()),
            float(router.weights_test[:, i].mean()),
        ]

    # Option B: 5-fold cross-fit router — clean OOF val RMSE, no val-label leakage
    crossfit = fit_crossfit_router(
        x_val=quality.val,
        pred_val=pred_val,
        y_val=split.y_val,
        x_test=quality.test,
        pred_test=pred_test,
        seed=args.seed,
    )
    model_rows.append(
        {
            "model": "P0_crossfit",
            "val_rmse": score_regression(split.y_val, crossfit.pred_val_oof)["rmse"],
            "test_rmse": score_regression(split.y_test, crossfit.pred_test)["rmse"],
            "val_mae": score_regression(split.y_val, crossfit.pred_val_oof)["mae"],
            "test_mae": score_regression(split.y_test, crossfit.pred_test)["mae"],
            "notes": f"5-fold OOF router — val RMSE is clean (unbiased); test uses router trained on all val (n_splits={crossfit.n_splits})",
        }
    )
    for i, name in enumerate(view_order):
        weights_summary[f"crossfit_{name}"] = [
            float(crossfit.weights_val_oof[:, i].mean()),
            float(crossfit.weights_test[:, i].mean()),
        ]

    payload = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_estimators": args.n_estimators,
        "view_names": view_order,
        "runtime": {
            "requested_device": args.device,
            "resolved_device": serialize_device_spec(device_plan.resolved_device),
            "device_mode": args.device_mode,
            "parallel_workers": device_plan.parallel_workers,
            "n_preprocessing_jobs": args.n_preprocessing_jobs,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "view_devices": view_devices_used,
        },
        "dataset": {
            "source": args.dataset_source,
            "openml_dataset_id": args.openml_dataset_id,
        },
        "results": model_rows,
        "weights_summary": weights_summary,
        "references": {
            "tabr_rmse": TABR_RMSE,
            "tabm_rmse": TABM_RMSE,
            "tabpfn_full_best": TABPFN_FULL_RMSE,
            "tabpfn_full_mean": TABPFN_FULL_MEAN,
            "a6f_artifact": A6F_ARTIFACT,
        },
    }

    (output_dir / "p0_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_metrics_csv(artifacts_dir / "metrics.csv", model_rows)
    write_report(
        output_dir / "report.md",
        model_rows,
        weights_summary,
        args.seed,
        args.split_seed,
        args.n_estimators,
        payload["dataset"],
        payload["runtime"],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
