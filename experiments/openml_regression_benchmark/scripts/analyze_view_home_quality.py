from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from experiments.openml_regression_benchmark.scripts.analyze_router_full_regret import (
        _load_run_arrays,
        _safe_fraction,
        _safe_mean,
        _safe_ratio,
    )
except ModuleNotFoundError:
    from analyze_router_full_regret import (  # type: ignore
        _load_run_arrays,
        _safe_fraction,
        _safe_mean,
        _safe_ratio,
    )


def _rmse(errors: np.ndarray) -> float:
    if errors.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(errors**2)))


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

    abs_err_views = np.abs(pred_views - y_test[:, None])
    sq_err_views = (pred_views - y_test[:, None]) ** 2
    full_abs_err = abs_err_views[:, full_idx]
    adaptive_abs_err = np.abs(adaptive_pred - y_test)
    fixed_abs_err = np.abs(fixed_pred - y_test)
    oracle_idx = abs_err_views.argmin(axis=1)

    adaptive_model = f"{run_label}_{adaptive_prefix}"
    fixed_model = f"{adaptive_model}_fixed"
    full_model = f"{run_label}_FULL"

    per_view: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(view_names):
        mask = oracle_idx == idx
        n_rows = int(mask.sum())
        view_abs_err = abs_err_views[:, idx]
        view_sq_err = sq_err_views[:, idx]
        potential_gain = full_abs_err[mask] - view_abs_err[mask]
        adaptive_realized = full_abs_err[mask] - adaptive_abs_err[mask]
        fixed_realized = full_abs_err[mask] - fixed_abs_err[mask]
        potential_sum = float(potential_gain.sum())

        per_view[name] = {
            "n_rows": n_rows,
            "row_fraction": float(mask.mean()),
            "home_view_rmse": _rmse(np.sqrt(view_sq_err[mask])),
            "full_rmse_on_home": _rmse(full_abs_err[mask]),
            "adaptive_rmse_on_home": _rmse(adaptive_abs_err[mask]),
            "fixed_rmse_on_home": _rmse(fixed_abs_err[mask]),
            "mean_potential_gain_vs_full": _safe_mean(potential_gain),
            "mean_adaptive_realized_gain_vs_full": _safe_mean(adaptive_realized),
            "mean_fixed_realized_gain_vs_full": _safe_mean(fixed_realized),
            "adaptive_capture_ratio_total": _safe_ratio(float(adaptive_realized.sum()), potential_sum),
            "fixed_capture_ratio_total": _safe_ratio(float(fixed_realized.sum()), potential_sum),
            "adaptive_capture_ratio_clipped": _safe_ratio(
                float(np.clip(adaptive_realized, 0.0, None).sum()),
                potential_sum,
            ),
            "fixed_capture_ratio_clipped": _safe_ratio(
                float(np.clip(fixed_realized, 0.0, None).sum()),
                potential_sum,
            ),
            "capture_gap_vs_fixed": _safe_ratio(float(adaptive_realized.sum()), potential_sum)
            - _safe_ratio(float(fixed_realized.sum()), potential_sum),
            "adaptive_positive_gain_fraction": _safe_fraction(adaptive_realized > 0.0),
            "fixed_positive_gain_fraction": _safe_fraction(fixed_realized > 0.0),
            "mean_adaptive_view_weight": _safe_mean(adaptive_weights[mask, idx]),
            "mean_fixed_view_weight": _safe_mean(fixed_weights[mask, idx]),
            "mean_adaptive_full_weight": _safe_mean(adaptive_weights[mask, full_idx]),
            "mean_fixed_full_weight": _safe_mean(fixed_weights[mask, full_idx]),
        }

    non_full_names = [name for name in view_names if name != "FULL"]
    non_full_capture_gaps = {
        name: float(per_view[name]["capture_gap_vs_fixed"])
        for name in non_full_names
    }
    weakest_non_full = min(non_full_capture_gaps, key=non_full_capture_gaps.get) if non_full_capture_gaps else None
    strongest_non_full = max(non_full_capture_gaps, key=non_full_capture_gaps.get) if non_full_capture_gaps else None

    summary = {
        "run_dir": str(run_dir),
        "adaptive_model": adaptive_model,
        "fixed_model": fixed_model,
        "full_model": full_model,
        "provenance": provenance,
        "global": {
            "test_rmse_full": float(rows[full_model]["test_rmse"]),
            "test_rmse_adaptive": float(rows[adaptive_model]["test_rmse"]),
            "test_rmse_fixed": float(rows[fixed_model]["test_rmse"]),
            "adaptive_minus_full_test_rmse": float(rows[full_model]["test_rmse"] - rows[adaptive_model]["test_rmse"]),
            "adaptive_minus_fixed_test_rmse": float(rows[fixed_model]["test_rmse"] - rows[adaptive_model]["test_rmse"]),
        },
        "per_view_home_subset": per_view,
        "non_full_capture_gap": {
            "by_view": non_full_capture_gaps,
            "worst_view": weakest_non_full,
            "best_view": strongest_non_full,
        },
    }
    return summary


def write_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# View Home-Subset Quality",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- adaptive model: `{summary['adaptive_model']}`",
        f"- fixed comparator mode: `{summary['provenance']['fixed_mode']}`",
        f"- quality feature mode: `{summary['provenance']['quality_mode']}`",
        f"- adaptive minus FULL test RMSE: `{summary['global']['adaptive_minus_full_test_rmse']:.4f}`",
        f"- adaptive minus fixed test RMSE: `{summary['global']['adaptive_minus_fixed_test_rmse']:.4f}`",
        "",
    ]
    for name, view in summary["per_view_home_subset"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- rows: `{view['n_rows']}`",
                f"- row fraction: `{view['row_fraction']:.3f}`",
                f"- home view RMSE: `{view['home_view_rmse']:.4f}`",
                f"- FULL RMSE on home rows: `{view['full_rmse_on_home']:.4f}`",
                f"- adaptive RMSE on home rows: `{view['adaptive_rmse_on_home']:.4f}`",
                f"- fixed RMSE on home rows: `{view['fixed_rmse_on_home']:.4f}`",
                f"- mean potential gain vs FULL: `{view['mean_potential_gain_vs_full']:.4f}`",
                f"- mean adaptive realized gain: `{view['mean_adaptive_realized_gain_vs_full']:.4f}`",
                f"- adaptive capture ratio total: `{view['adaptive_capture_ratio_total']:.3f}`",
                f"- fixed capture ratio total: `{view['fixed_capture_ratio_total']:.3f}`",
                f"- capture gap vs fixed: `{view['capture_gap_vs_fixed']:.3f}`",
                f"- mean adaptive view weight: `{view['mean_adaptive_view_weight']:.3f}`",
                f"- mean adaptive FULL weight: `{view['mean_adaptive_full_weight']:.3f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-view quality on each view's oracle-home subset")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    parser.add_argument("--label", type=str, default="GraphDrone")
    args = parser.parse_args()

    label = "P0" if (args.run_dir / "p0_results.json").exists() else args.label
    summary = analyze_run(args.run_dir, adaptive_prefix=args.adaptive_prefix, label=label)
    out_json = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_view_home_summary.json"
    out_md = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_view_home_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_summary(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
