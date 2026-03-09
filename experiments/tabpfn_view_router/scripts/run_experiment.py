from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tabpfn_view_router.src.data import (
    SEED,
    build_aligned_california_split,
    build_quality_features,
    build_view_data,
)
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
TABPFN_FULL_RMSE = 0.3891
TABPFN_FULL_MEAN = 0.3932
A6F_ARTIFACT = 0.4063


def fit_view_experts(views, split, seed: int, n_estimators: int, device: str) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    from tabpfn import TabPFNRegressor

    pred_val_cols = []
    pred_test_cols = []
    models = {}
    for name in views.view_names:
        model = TabPFNRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            device=device,
            ignore_pretraining_limits=len(views.train[name]) > 1000,
            n_preprocessing_jobs=1,
        )
        model.fit(views.train[name], split.y_train)
        models[name] = model
        pred_val_cols.append(model.predict(views.val[name]).astype(np.float32))
        pred_test_cols.append(model.predict(views.test[name]).astype(np.float32))
    pred_val = np.stack(pred_val_cols, axis=1).astype(np.float32)
    pred_test = np.stack(pred_test_cols, axis=1).astype(np.float32)
    return models, pred_val, pred_test


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], weights_summary: dict[str, list[float]], seed: int, n_estimators: int) -> None:
    lines = [
        "# P0 TabPFN View Router",
        "",
        f"- seed: `{seed}`",
        f"- n_estimators per expert: `{n_estimators}`",
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
            f"- TabPFN_full_best: `{TABPFN_FULL_RMSE:.4f}`",
            f"- TabPFN_full_multiseed_mean: `{TABPFN_FULL_MEAN:.4f}`",
            f"- MV-TabR-GoRA A6f artifact: `{A6F_ARTIFACT:.4f}`",
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments/tabpfn_view_router/reports/p0_full",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir.parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split = build_aligned_california_split(seed=args.split_seed, smoke=args.smoke)
    views = build_view_data(split)
    quality = build_quality_features(split, views, k=24)

    _, pred_val, pred_test = fit_view_experts(
        views=views,
        split=split,
        seed=args.seed,
        n_estimators=args.n_estimators,
        device=args.device,
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
            float(crossfit.weights_test[:, i].mean()),
            float(crossfit.weights_test[:, i].mean()),
        ]

    payload = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_estimators": args.n_estimators,
        "view_names": view_order,
        "results": model_rows,
        "weights_summary": weights_summary,
        "references": {
            "tabr_rmse": TABR_RMSE,
            "tabpfn_full_best": TABPFN_FULL_RMSE,
            "tabpfn_full_mean": TABPFN_FULL_MEAN,
            "a6f_artifact": A6F_ARTIFACT,
        },
    }

    (output_dir / "p0_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_metrics_csv(artifacts_dir / "metrics.csv", model_rows)
    write_report(output_dir / "report.md", model_rows, weights_summary, args.seed, args.n_estimators)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
