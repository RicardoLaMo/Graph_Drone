from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _bucket_summary(values: np.ndarray, improvements: np.ndarray, *, quantiles: int = 4) -> list[dict[str, float | int]]:
    if len(values) != len(improvements):
        raise ValueError("Expected values and improvements to have matching lengths")
    edges = np.quantile(values, np.linspace(0.0, 1.0, quantiles + 1))
    summary: list[dict[str, float | int]] = []
    for idx in range(quantiles):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == quantiles - 1:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        if not np.any(mask):
            continue
        bucket = improvements[mask]
        summary.append(
            {
                "bucket_index": idx,
                "lower": lo,
                "upper": hi,
                "n_rows": int(mask.sum()),
                "mean_improvement": float(bucket.mean()),
                "median_improvement": float(np.median(bucket)),
                "positive_fraction": float((bucket > 0.0).mean()),
            }
        )
    return summary


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _extract_rows(payload: dict[str, object]) -> tuple[dict[str, dict[str, object]], str]:
    if "rows" in payload:
        rows_key = "rows"
    elif "results" in payload:
        rows_key = "results"
    else:
        raise ValueError("Expected payload to contain 'rows' or 'results'")
    return {row["model"]: row for row in payload[rows_key]}, rows_key


def analyze_run(run_dir: Path, *, adaptive_prefix: str, label: str | None = None) -> dict[str, object]:
    result_candidates = [run_dir / "graphdrone_results.json", run_dir / "p0_results.json"]
    result_path = next((path for path in result_candidates if path.exists()), None)
    if result_path is None:
        raise FileNotFoundError(f"Could not find result payload in {run_dir}")

    artifact_candidates = [
        run_dir / "artifacts" / "graphdrone_predictions.npz",
        run_dir / "artifacts" / "p0_router_diagnostics.npz",
    ]
    artifact_path = next((path for path in artifact_candidates if path.exists()), None)
    if artifact_path is None:
        raise FileNotFoundError(f"Could not find prediction artifact bundle in {run_dir}")

    payload = json.loads(result_path.read_text())
    rows, _ = _extract_rows(payload)
    arrays = np.load(artifact_path)

    adaptive_pred_test = arrays[f"{adaptive_prefix}_pred_test"].astype(np.float32)
    fixed_pred_test = arrays[f"{adaptive_prefix}_fixed_pred_test"].astype(np.float32)
    adaptive_weights_test = arrays[f"{adaptive_prefix}_weights_test"].astype(np.float32)
    fixed_weights_test = arrays[f"{adaptive_prefix}_fixed_weights_test"].astype(np.float32)
    pred_views_test = arrays["pred_test"].astype(np.float32)
    y_test = arrays["y_test"].astype(np.float32)
    quality_test = arrays["quality_test"].astype(np.float32)
    view_names = [str(v) for v in arrays["view_names"].tolist()]

    adaptive_improvement = np.abs(fixed_pred_test - y_test) - np.abs(adaptive_pred_test - y_test)
    disagreement = pred_views_test.std(axis=1)
    weight_l1_shift = np.abs(adaptive_weights_test - fixed_weights_test).sum(axis=1)
    mean_j = quality_test[:, -1]
    sigma2_full = quality_test[:, 0]
    router_entropy = -(adaptive_weights_test * np.log(adaptive_weights_test + 1e-8)).sum(axis=1)

    adaptive_model = f"{label}_{adaptive_prefix}" if label else adaptive_prefix
    fixed_model = f"{adaptive_model}_fixed"
    adaptive_metrics = rows[adaptive_model]
    fixed_metrics = rows[fixed_model]

    return {
        "run_dir": str(run_dir),
        "adaptive_model": adaptive_model,
        "fixed_model": fixed_model,
        "test_rmse_adaptive": float(adaptive_metrics["test_rmse"]),
        "test_rmse_fixed": float(fixed_metrics["test_rmse"]),
        "adaptive_minus_fixed_test_rmse": float(fixed_metrics["test_rmse"] - adaptive_metrics["test_rmse"]),
        "mean_abs_error_improvement": float(adaptive_improvement.mean()),
        "median_abs_error_improvement": float(np.median(adaptive_improvement)),
        "positive_improvement_fraction": float((adaptive_improvement > 0.0).mean()),
        "mean_weight_l1_shift": float(weight_l1_shift.mean()),
        "median_weight_l1_shift": float(np.median(weight_l1_shift)),
        "weight_shift_gt_0_05_fraction": float((weight_l1_shift > 0.05).mean()),
        "weight_shift_gt_0_10_fraction": float((weight_l1_shift > 0.10).mean()),
        "corr_improvement_vs_disagreement": _corr(adaptive_improvement, disagreement),
        "corr_improvement_vs_weight_shift": _corr(adaptive_improvement, weight_l1_shift),
        "corr_improvement_vs_mean_j": _corr(adaptive_improvement, mean_j),
        "corr_improvement_vs_sigma2_full": _corr(adaptive_improvement, sigma2_full),
        "corr_improvement_vs_entropy": _corr(adaptive_improvement, router_entropy),
        "top_disagreement_buckets": _bucket_summary(disagreement, adaptive_improvement),
        "top_weight_means": {
            name: float(adaptive_weights_test[:, idx].mean())
            for idx, name in enumerate(view_names)
        },
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Router Mechanism Summary",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- adaptive_model: `{summary['adaptive_model']}`",
        f"- fixed_model: `{summary['fixed_model']}`",
        f"- adaptive test RMSE: `{summary['test_rmse_adaptive']:.4f}`",
        f"- fixed test RMSE: `{summary['test_rmse_fixed']:.4f}`",
        f"- adaptive minus fixed test RMSE: `{summary['adaptive_minus_fixed_test_rmse']:.4f}`",
        f"- positive improvement fraction: `{summary['positive_improvement_fraction']:.3f}`",
        f"- mean row weight L1 shift: `{summary['mean_weight_l1_shift']:.4f}`",
        f"- corr(improvement, disagreement): `{summary['corr_improvement_vs_disagreement']:.4f}`",
        f"- corr(improvement, weight shift): `{summary['corr_improvement_vs_weight_shift']:.4f}`",
        f"- corr(improvement, mean_J): `{summary['corr_improvement_vs_mean_j']:.4f}`",
        f"- corr(improvement, sigma2_FULL): `{summary['corr_improvement_vs_sigma2_full']:.4f}`",
        f"- corr(improvement, entropy): `{summary['corr_improvement_vs_entropy']:.4f}`",
        "",
        "## Mean Adaptive Test Weights",
        "",
    ]
    for name, value in summary["top_weight_means"].items():
        lines.append(f"- {name}: `{value:.3f}`")
    lines.extend(["", "## Disagreement Buckets", ""])
    for bucket in summary["top_disagreement_buckets"]:
        lines.append(
            f"- q{bucket['bucket_index']}: rows `{bucket['n_rows']}`, "
            f"mean improvement `{bucket['mean_improvement']:.4f}`, "
            f"positive fraction `{bucket['positive_fraction']:.3f}`"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize adaptive vs fixed router behavior for one run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    parser.add_argument("--label", type=str, default="GraphDrone")
    args = parser.parse_args()

    if (args.run_dir / "p0_results.json").exists():
        label = "P0"
    else:
        label = args.label

    summary = analyze_run(args.run_dir, adaptive_prefix=args.adaptive_prefix, label=label)
    out_json = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_mechanism_summary.json"
    out_md = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_mechanism_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_summary(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
