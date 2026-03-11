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


def _fixed_weight_mix(preds: np.ndarray, mean_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(mean_weights, dtype=np.float32).reshape(-1)
    clipped = np.clip(weights, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError("Expected positive fixed-weight mass")
    normalized = (clipped / total).astype(np.float32)
    tiled = np.broadcast_to(normalized[None, :], preds.shape).copy().astype(np.float32)
    pred = (tiled * preds).sum(axis=1)
    return pred.astype(np.float32), tiled


def _score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    diff = y_true - y_pred
    mse = float(np.mean(diff**2)) if diff.size else 0.0
    mae = float(np.mean(np.abs(diff))) if diff.size else 0.0
    denom = float(np.sum((y_true - y_true.mean()) ** 2)) if y_true.size else 0.0
    r2 = 0.0 if denom <= 0.0 else float(1.0 - np.sum(diff**2) / denom)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": r2,
    }


def _maybe_derive_quality(arrays: dict[str, np.ndarray]) -> str:
    if "quality_val" in arrays and "quality_test" in arrays:
        return "explicit"
    if not {"sigma2_val", "sigma2_test", "mean_j_val", "mean_j_test"}.issubset(arrays):
        raise KeyError("Missing both explicit quality tensors and sigma2/mean_J fallback arrays")
    arrays["quality_val"] = np.concatenate(
        [
            arrays["sigma2_val"].astype(np.float32),
            arrays["mean_j_val"].astype(np.float32).reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    arrays["quality_test"] = np.concatenate(
        [
            arrays["sigma2_test"].astype(np.float32),
            arrays["mean_j_test"].astype(np.float32).reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    return "derived_sigma2_plus_mean_j"


def _maybe_derive_fixed(
    *,
    rows: dict[str, dict[str, object]],
    arrays: dict[str, np.ndarray],
    adaptive_prefix: str,
    run_label: str,
) -> str:
    fixed_pred_key = f"{adaptive_prefix}_fixed_pred_test"
    fixed_weight_key = f"{adaptive_prefix}_fixed_weights_test"
    if fixed_pred_key in arrays and fixed_weight_key in arrays:
        return "explicit"

    pred_val = arrays["pred_val"].astype(np.float32)
    pred_test = arrays["pred_test"].astype(np.float32)
    y_val = arrays["y_val"].astype(np.float32)
    y_test = arrays["y_test"].astype(np.float32)
    val_weights = arrays[f"{adaptive_prefix}_weights_val"].astype(np.float32)
    mean_weights = val_weights.mean(axis=0)

    fixed_pred_val, fixed_weights_val = _fixed_weight_mix(pred_val, mean_weights)
    fixed_pred_test, fixed_weights_test = _fixed_weight_mix(pred_test, mean_weights)
    arrays[f"{adaptive_prefix}_fixed_pred_val"] = fixed_pred_val
    arrays[f"{adaptive_prefix}_fixed_weights_val"] = fixed_weights_val
    arrays[f"{adaptive_prefix}_fixed_pred_test"] = fixed_pred_test
    arrays[f"{adaptive_prefix}_fixed_weights_test"] = fixed_weights_test

    fixed_model = f"{run_label}_{adaptive_prefix}_fixed"
    if fixed_model not in rows:
        val_metrics = _score_regression(y_val, fixed_pred_val)
        test_metrics = _score_regression(y_test, fixed_pred_test)
        rows[fixed_model] = {
            "model": fixed_model,
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "notes": "derived fixed hedge from mean validation router weights",
        }
    return "derived_from_val_mean_weights"


def _load_run_arrays(
    run_dir: Path,
    *,
    adaptive_prefix: str,
    label: str | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, np.ndarray], str, dict[str, str]]:
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
    arrays = np.load(artifact_path, allow_pickle=True)
    run_label = label or ("P0" if result_path.name == "p0_results.json" else "GraphDrone")
    array_dict = {name: arrays[name] for name in arrays.files}
    fixed_mode = _maybe_derive_fixed(
        rows=rows,
        arrays=array_dict,
        adaptive_prefix=adaptive_prefix,
        run_label=run_label,
    )
    quality_mode = _maybe_derive_quality(array_dict)
    provenance = {
        "fixed_mode": fixed_mode,
        "quality_mode": quality_mode,
    }
    return payload, rows, array_dict, run_label, provenance


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(x.mean())


def _safe_median(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.median(x))


def _safe_fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-8:
        return 0.0
    return float(num / den)


def _bucket_metric(values: np.ndarray, metric: np.ndarray, mask: np.ndarray) -> list[dict[str, float | int]]:
    if values.shape[0] != metric.shape[0] or values.shape[0] != mask.shape[0]:
        raise ValueError("Expected aligned arrays for bucketed metric summary")
    if not np.any(mask):
        return []
    return _bucket_summary(values[mask], metric[mask])


def analyze_run(run_dir: Path, *, adaptive_prefix: str = "router", label: str | None = None) -> dict[str, object]:
    _, rows, arrays, run_label, provenance = _load_run_arrays(
        run_dir,
        adaptive_prefix=adaptive_prefix,
        label=label,
    )

    view_names = [str(v) for v in arrays["view_names"].tolist()]
    if "FULL" not in view_names:
        raise ValueError(f"Expected FULL in view_names, got {view_names!r}")
    full_idx = view_names.index("FULL")

    y_test = arrays["y_test"].astype(np.float32)
    pred_views = arrays["pred_test"].astype(np.float32)
    adaptive_pred = arrays[f"{adaptive_prefix}_pred_test"].astype(np.float32)
    fixed_pred = arrays[f"{adaptive_prefix}_fixed_pred_test"].astype(np.float32)
    adaptive_weights = arrays[f"{adaptive_prefix}_weights_test"].astype(np.float32)
    fixed_weights = arrays[f"{adaptive_prefix}_fixed_weights_test"].astype(np.float32)
    quality = arrays["quality_test"].astype(np.float32)

    full_pred = pred_views[:, full_idx]
    abs_err_views = np.abs(pred_views - y_test[:, None])
    full_abs_err = abs_err_views[:, full_idx]
    oracle_idx = abs_err_views.argmin(axis=1)
    oracle_abs_err = abs_err_views[np.arange(len(y_test)), oracle_idx]
    adaptive_abs_err = np.abs(adaptive_pred - y_test)
    fixed_abs_err = np.abs(fixed_pred - y_test)

    full_oracle_mask = oracle_idx == full_idx
    non_full_oracle_mask = ~full_oracle_mask

    disagreement = pred_views.std(axis=1)
    adaptive_full_weight = adaptive_weights[:, full_idx]
    fixed_full_weight = fixed_weights[:, full_idx]
    adaptive_away_mass = 1.0 - adaptive_full_weight
    fixed_away_mass = 1.0 - fixed_full_weight
    adaptive_vs_fixed_away_delta = adaptive_away_mass - fixed_away_mass

    full_case_harm = adaptive_abs_err - full_abs_err
    fixed_full_case_harm = fixed_abs_err - full_abs_err
    non_full_potential = full_abs_err - oracle_abs_err
    adaptive_realized = full_abs_err - adaptive_abs_err
    fixed_realized = full_abs_err - fixed_abs_err

    non_full_potential_sum = float(non_full_potential[non_full_oracle_mask].sum())
    adaptive_realized_sum = float(adaptive_realized[non_full_oracle_mask].sum())
    fixed_realized_sum = float(fixed_realized[non_full_oracle_mask].sum())
    adaptive_clipped_sum = float(np.clip(adaptive_realized[non_full_oracle_mask], 0.0, None).sum())
    fixed_clipped_sum = float(np.clip(fixed_realized[non_full_oracle_mask], 0.0, None).sum())

    adaptive_model = f"{run_label}_{adaptive_prefix}"
    fixed_model = f"{adaptive_model}_fixed"
    full_model = f"{run_label}_FULL"

    summary = {
        "run_dir": str(run_dir),
        "adaptive_model": adaptive_model,
        "fixed_model": fixed_model,
        "full_model": full_model,
        "provenance": provenance,
        "n_rows": int(len(y_test)),
        "view_names": view_names,
        "global": {
            "test_rmse_full": float(rows[full_model]["test_rmse"]),
            "test_rmse_adaptive": float(rows[adaptive_model]["test_rmse"]),
            "test_rmse_fixed": float(rows[fixed_model]["test_rmse"]),
            "adaptive_minus_full_test_rmse": float(rows[full_model]["test_rmse"] - rows[adaptive_model]["test_rmse"]),
            "fixed_minus_full_test_rmse": float(rows[full_model]["test_rmse"] - rows[fixed_model]["test_rmse"]),
            "adaptive_minus_fixed_test_rmse": float(rows[fixed_model]["test_rmse"] - rows[adaptive_model]["test_rmse"]),
            "full_oracle_fraction": float(full_oracle_mask.mean()),
            "mean_adaptive_full_weight": float(adaptive_full_weight.mean()),
            "mean_fixed_full_weight": float(fixed_full_weight.mean()),
            "corr_adaptive_realized_vs_disagreement": _corr(adaptive_realized, disagreement),
            "corr_adaptive_realized_vs_away_mass": _corr(adaptive_realized, adaptive_away_mass),
            "corr_full_case_harm_vs_away_mass": _corr(full_case_harm[full_oracle_mask], adaptive_away_mass[full_oracle_mask]),
            "corr_non_full_realized_vs_away_mass": _corr(
                adaptive_realized[non_full_oracle_mask],
                adaptive_away_mass[non_full_oracle_mask],
            ),
            "corr_non_full_realized_vs_mean_j": _corr(
                adaptive_realized[non_full_oracle_mask],
                quality[non_full_oracle_mask, -1],
            ),
        },
        "full_oracle_case": {
            "n_rows": int(full_oracle_mask.sum()),
            "mean_adaptive_full_weight": _safe_mean(adaptive_full_weight[full_oracle_mask]),
            "mean_fixed_full_weight": _safe_mean(fixed_full_weight[full_oracle_mask]),
            "mean_adaptive_away_mass": _safe_mean(adaptive_away_mass[full_oracle_mask]),
            "mean_fixed_away_mass": _safe_mean(fixed_away_mass[full_oracle_mask]),
            "mean_adaptive_vs_fixed_away_delta": _safe_mean(adaptive_vs_fixed_away_delta[full_oracle_mask]),
            "false_diversion_mean_cost": _safe_mean(full_case_harm[full_oracle_mask]),
            "false_diversion_median_cost": _safe_median(full_case_harm[full_oracle_mask]),
            "false_diversion_positive_fraction": _safe_fraction(full_case_harm[full_oracle_mask] > 0.0),
            "false_diversion_gt_0_01_fraction": _safe_fraction(full_case_harm[full_oracle_mask] > 0.01),
            "fixed_false_diversion_mean_cost": _safe_mean(fixed_full_case_harm[full_oracle_mask]),
            "fixed_false_diversion_positive_fraction": _safe_fraction(fixed_full_case_harm[full_oracle_mask] > 0.0),
            "disagreement_buckets": _bucket_metric(disagreement, full_case_harm, full_oracle_mask),
        },
        "non_full_oracle_case": {
            "n_rows": int(non_full_oracle_mask.sum()),
            "oracle_view_fraction": {
                name: float(((oracle_idx == idx) & non_full_oracle_mask).mean())
                for idx, name in enumerate(view_names)
                if idx != full_idx
            },
            "mean_adaptive_full_weight": _safe_mean(adaptive_full_weight[non_full_oracle_mask]),
            "mean_fixed_full_weight": _safe_mean(fixed_full_weight[non_full_oracle_mask]),
            "high_adaptive_full_weight_fraction": _safe_fraction(adaptive_full_weight[non_full_oracle_mask] >= 0.80),
            "high_fixed_full_weight_fraction": _safe_fraction(fixed_full_weight[non_full_oracle_mask] >= 0.80),
            "mean_potential_gain": _safe_mean(non_full_potential[non_full_oracle_mask]),
            "mean_adaptive_realized_gain": _safe_mean(adaptive_realized[non_full_oracle_mask]),
            "mean_fixed_realized_gain": _safe_mean(fixed_realized[non_full_oracle_mask]),
            "adaptive_positive_gain_fraction": _safe_fraction(adaptive_realized[non_full_oracle_mask] > 0.0),
            "fixed_positive_gain_fraction": _safe_fraction(fixed_realized[non_full_oracle_mask] > 0.0),
            "adaptive_capture_ratio_total": _safe_ratio(adaptive_realized_sum, non_full_potential_sum),
            "fixed_capture_ratio_total": _safe_ratio(fixed_realized_sum, non_full_potential_sum),
            "adaptive_capture_ratio_clipped": _safe_ratio(adaptive_clipped_sum, non_full_potential_sum),
            "fixed_capture_ratio_clipped": _safe_ratio(fixed_clipped_sum, non_full_potential_sum),
            "missed_opportunity_mean_cost": _safe_mean(
                np.maximum(non_full_potential[non_full_oracle_mask] - adaptive_realized[non_full_oracle_mask], 0.0)
            ),
            "fixed_missed_opportunity_mean_cost": _safe_mean(
                np.maximum(non_full_potential[non_full_oracle_mask] - fixed_realized[non_full_oracle_mask], 0.0)
            ),
            "disagreement_buckets": _bucket_metric(disagreement, adaptive_realized, non_full_oracle_mask),
            "potential_gain_buckets": _bucket_metric(non_full_potential, adaptive_realized, non_full_oracle_mask),
        },
    }
    return summary


def write_summary(path: Path, summary: dict[str, object]) -> None:
    global_stats = summary["global"]
    full_case = summary["full_oracle_case"]
    non_full_case = summary["non_full_oracle_case"]
    lines = [
        "# FULL vs Adaptive Diagnosis",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- adaptive model: `{summary['adaptive_model']}`",
        f"- fixed model: `{summary['fixed_model']}`",
        f"- fixed comparator mode: `{summary['provenance']['fixed_mode']}`",
        f"- quality feature mode: `{summary['provenance']['quality_mode']}`",
        f"- FULL test RMSE: `{global_stats['test_rmse_full']:.4f}`",
        f"- adaptive test RMSE: `{global_stats['test_rmse_adaptive']:.4f}`",
        f"- fixed test RMSE: `{global_stats['test_rmse_fixed']:.4f}`",
        f"- adaptive minus FULL test RMSE: `{global_stats['adaptive_minus_full_test_rmse']:.4f}`",
        f"- adaptive minus fixed test RMSE: `{global_stats['adaptive_minus_fixed_test_rmse']:.4f}`",
        f"- FULL oracle fraction: `{global_stats['full_oracle_fraction']:.3f}`",
        "",
        "## When FULL Is Oracle-Best",
        "",
        f"- rows: `{full_case['n_rows']}`",
        f"- mean adaptive FULL weight: `{full_case['mean_adaptive_full_weight']:.3f}`",
        f"- mean fixed FULL weight: `{full_case['mean_fixed_full_weight']:.3f}`",
        f"- false diversion mean cost: `{full_case['false_diversion_mean_cost']:.4f}`",
        f"- false diversion positive fraction: `{full_case['false_diversion_positive_fraction']:.3f}`",
        "",
        "## When FULL Is Not Oracle-Best",
        "",
        f"- rows: `{non_full_case['n_rows']}`",
        f"- mean adaptive FULL weight: `{non_full_case['mean_adaptive_full_weight']:.3f}`",
        f"- mean fixed FULL weight: `{non_full_case['mean_fixed_full_weight']:.3f}`",
        f"- mean potential gain: `{non_full_case['mean_potential_gain']:.4f}`",
        f"- mean adaptive realized gain: `{non_full_case['mean_adaptive_realized_gain']:.4f}`",
        f"- adaptive capture ratio total: `{non_full_case['adaptive_capture_ratio_total']:.3f}`",
        f"- adaptive capture ratio clipped: `{non_full_case['adaptive_capture_ratio_clipped']:.3f}`",
        f"- missed opportunity mean cost: `{non_full_case['missed_opportunity_mean_cost']:.4f}`",
        "",
        "## Non-FULL Oracle View Fractions",
        "",
    ]
    for name, value in non_full_case["oracle_view_fraction"].items():
        lines.append(f"- {name}: `{value:.3f}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why FULL beats or loses to adaptive routing")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    parser.add_argument("--label", type=str, default="GraphDrone")
    args = parser.parse_args()

    label = "P0" if (args.run_dir / "p0_results.json").exists() else args.label
    summary = analyze_run(args.run_dir, adaptive_prefix=args.adaptive_prefix, label=label)
    out_json = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_full_regret_summary.json"
    out_md = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_full_regret_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_summary(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
