from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
ROUTER_SRC = REPO_ROOT / "experiments" / "tabpfn_view_router" / "src"
for path in (REPO_ROOT, ROUTER_SRC, THIS_FILE.parent):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.openml_regression_benchmark.scripts.analyze_router_full_regret import _load_run_arrays
from router import fixed_weight_mix, fit_soft_router, score_regression


def _pair_name(anchor: str, candidate: str) -> str:
    return f"{anchor}+{candidate}"


def analyze_run(
    run_dir: Path,
    *,
    adaptive_prefix: str = "router",
    label: str | None = None,
    seed: int = 42,
) -> dict[str, object]:
    payload, rows, arrays, run_label = _load_run_arrays(run_dir, adaptive_prefix=adaptive_prefix, label=label)

    view_names = [str(v) for v in arrays["view_names"].tolist()]
    if "FULL" not in view_names:
        raise ValueError(f"Expected FULL in view_names, got {view_names!r}")
    full_idx = view_names.index("FULL")

    pred_val = arrays["pred_val"].astype(np.float32)
    pred_test = arrays["pred_test"].astype(np.float32)
    quality_val = arrays["quality_val"].astype(np.float32)
    quality_test = arrays["quality_test"].astype(np.float32)
    y_val = arrays["y_val"].astype(np.float32)
    y_test = arrays["y_test"].astype(np.float32)

    adaptive_model = f"{run_label}_{adaptive_prefix}"
    fixed_model = f"{adaptive_model}_fixed"
    full_model = f"{run_label}_FULL"

    full_router_rmse = float(rows[adaptive_model]["test_rmse"])
    full_fixed_rmse = float(rows[fixed_model]["test_rmse"])
    full_expert_rmse = float(rows[full_model]["test_rmse"])

    candidates: dict[str, dict[str, object]] = {}
    for idx, name in enumerate(view_names):
        if idx == full_idx:
            continue
        chosen_idx = [full_idx, idx]
        pair_pred_val = pred_val[:, chosen_idx]
        pair_pred_test = pred_test[:, chosen_idx]

        pair_router = fit_soft_router(
            x_val=quality_val,
            pred_val=pair_pred_val,
            y_val=y_val,
            x_test=quality_test,
            pred_test=pair_pred_test,
            seed=seed,
        )
        pair_router_metrics = score_regression(y_test, pair_router.pred_test)
        pair_fixed_mean = pair_router.weights_val.mean(axis=0)
        pair_fixed_pred_test, pair_fixed_weights_test = fixed_weight_mix(pair_pred_test, pair_fixed_mean)
        pair_fixed_metrics = score_regression(y_test, pair_fixed_pred_test)

        candidates[name] = {
            "pair_name": _pair_name("FULL", name),
            "adaptive_test_rmse": pair_router_metrics["rmse"],
            "fixed_test_rmse": pair_fixed_metrics["rmse"],
            "adaptive_minus_pair_fixed": float(pair_fixed_metrics["rmse"] - pair_router_metrics["rmse"]),
            "adaptive_minus_full_router": float(full_router_rmse - pair_router_metrics["rmse"]),
            "adaptive_minus_full_fixed": float(full_fixed_rmse - pair_router_metrics["rmse"]),
            "adaptive_minus_full_expert": float(full_expert_rmse - pair_router_metrics["rmse"]),
            "mean_pair_weights_val": {
                "FULL": float(pair_router.weights_val[:, 0].mean()),
                name: float(pair_router.weights_val[:, 1].mean()),
            },
            "mean_pair_weights_test": {
                "FULL": float(pair_router.weights_test[:, 0].mean()),
                name: float(pair_router.weights_test[:, 1].mean()),
            },
            "fixed_pair_weights_test": {
                "FULL": float(pair_fixed_weights_test[:, 0].mean()),
                name: float(pair_fixed_weights_test[:, 1].mean()),
            },
        }

    best_candidate = max(candidates.items(), key=lambda item: item[1]["adaptive_minus_full_router"])[0]
    best_pair = candidates[best_candidate]

    return {
        "run_dir": str(run_dir),
        "adaptive_model": adaptive_model,
        "fixed_model": fixed_model,
        "full_model": full_model,
        "global_reference": {
            "full_expert_test_rmse": full_expert_rmse,
            "full_router_test_rmse": full_router_rmse,
            "full_fixed_test_rmse": full_fixed_rmse,
        },
        "best_candidate_view": best_candidate,
        "best_two_expert": best_pair,
        "candidates": candidates,
        "runtime_context": {
            "seed": seed,
            "n_views_total": len(view_names),
            "payload_runtime": payload.get("runtime", {}),
        },
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Two-Expert Competition Diagnosis",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- adaptive model: `{summary['adaptive_model']}`",
        f"- full expert test RMSE: `{summary['global_reference']['full_expert_test_rmse']:.4f}`",
        f"- full router test RMSE: `{summary['global_reference']['full_router_test_rmse']:.4f}`",
        f"- full fixed test RMSE: `{summary['global_reference']['full_fixed_test_rmse']:.4f}`",
        f"- best candidate view: `{summary['best_candidate_view']}`",
        "",
        "## Best Two-Expert Pair",
        "",
        f"- pair: `{summary['best_two_expert']['pair_name']}`",
        f"- adaptive test RMSE: `{summary['best_two_expert']['adaptive_test_rmse']:.4f}`",
        f"- fixed test RMSE: `{summary['best_two_expert']['fixed_test_rmse']:.4f}`",
        f"- gain vs full router: `{summary['best_two_expert']['adaptive_minus_full_router']:.4f}`",
        f"- gain vs full expert: `{summary['best_two_expert']['adaptive_minus_full_expert']:.4f}`",
        "",
        "## All Candidate Views",
        "",
    ]
    for name, candidate in summary["candidates"].items():
        lines.extend(
            [
                f"- {name}: two-expert adaptive `{candidate['adaptive_test_rmse']:.4f}`, "
                f"gain vs full router `{candidate['adaptive_minus_full_router']:.4f}`, "
                f"gain vs full expert `{candidate['adaptive_minus_full_expert']:.4f}`, "
                f"FULL weight `{candidate['mean_pair_weights_test']['FULL']:.3f}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose multi-view competition noise via FULL+one-view rerouting")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    parser.add_argument("--label", type=str, default="GraphDrone")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label = "P0" if (args.run_dir / "p0_results.json").exists() else args.label
    summary = analyze_run(run_dir=args.run_dir, adaptive_prefix=args.adaptive_prefix, label=label, seed=args.seed)
    out_json = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_two_expert_summary.json"
    out_md = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_two_expert_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_summary(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
