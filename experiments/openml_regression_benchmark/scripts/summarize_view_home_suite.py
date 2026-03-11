from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def summarize(root: Path, *, adaptive_prefix: str = "router") -> dict[str, object]:
    paths = sorted(root.glob(f"**/artifacts/{adaptive_prefix}_view_home_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No {adaptive_prefix}_view_home_summary.json files found under {root}")

    payloads = [json.loads(path.read_text()) for path in paths]
    view_names = payloads[0]["per_view_home_subset"].keys()

    per_view: dict[str, dict[str, object]] = {}
    for view_name in view_names:
        capture_gap = [payload["per_view_home_subset"][view_name]["capture_gap_vs_fixed"] for payload in payloads]
        potential_gain = [payload["per_view_home_subset"][view_name]["mean_potential_gain_vs_full"] for payload in payloads]
        adaptive_capture = [payload["per_view_home_subset"][view_name]["adaptive_capture_ratio_total"] for payload in payloads]
        adaptive_full_weight = [payload["per_view_home_subset"][view_name]["mean_adaptive_full_weight"] for payload in payloads]
        per_view[view_name] = {
            "capture_gap_vs_fixed": {
                **_mean_std(capture_gap),
                "positive_fraction": float((np.asarray(capture_gap) > 0.0).mean()),
            },
            "mean_potential_gain_vs_full": _mean_std(potential_gain),
            "adaptive_capture_ratio_total": _mean_std(adaptive_capture),
            "mean_adaptive_full_weight": _mean_std(adaptive_full_weight),
        }

    return {
        "root": str(root),
        "adaptive_prefix": adaptive_prefix,
        "n_runs": len(payloads),
        "per_view": per_view,
    }


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# View Home-Suite Summary",
        "",
        f"- root: `{summary['root']}`",
        f"- adaptive_prefix: `{summary['adaptive_prefix']}`",
        f"- n_runs: `{summary['n_runs']}`",
        "",
    ]
    for view_name, payload in summary["per_view"].items():
        lines.extend(
            [
                f"## {view_name}",
                "",
                f"- capture gap vs fixed: `{payload['capture_gap_vs_fixed']['mean']:.6f} ± {payload['capture_gap_vs_fixed']['std']:.6f}`",
                f"- capture gap positive fraction: `{payload['capture_gap_vs_fixed']['positive_fraction']:.3f}`",
                f"- mean potential gain vs FULL: `{payload['mean_potential_gain_vs_full']['mean']:.6f} ± {payload['mean_potential_gain_vs_full']['std']:.6f}`",
                f"- adaptive capture ratio total: `{payload['adaptive_capture_ratio_total']['mean']:.6f} ± {payload['adaptive_capture_ratio_total']['std']:.6f}`",
                f"- mean adaptive FULL weight: `{payload['mean_adaptive_full_weight']['mean']:.6f} ± {payload['mean_adaptive_full_weight']['std']:.6f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate view-home summaries over many runs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    args = parser.parse_args()

    summary = summarize(args.root, adaptive_prefix=args.adaptive_prefix)
    out_json = args.root / f"{args.adaptive_prefix}_view_home_suite_summary.json"
    out_md = args.root / f"{args.adaptive_prefix}_view_home_suite_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
