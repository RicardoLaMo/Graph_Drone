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
    paths = sorted(root.glob(f"**/artifacts/{adaptive_prefix}_signal_noise_tradeoff.json"))
    if not paths:
        raise FileNotFoundError(f"No {adaptive_prefix}_signal_noise_tradeoff.json files found under {root}")

    payloads = [json.loads(path.read_text()) for path in paths]
    classifications: dict[str, int] = {}
    best_views: dict[str, int] = {}
    for payload in payloads:
        classifications[payload["classification"]] = classifications.get(payload["classification"], 0) + 1
        best_views[payload["best_view"]] = best_views.get(payload["best_view"], 0) + 1

    return {
        "root": str(root),
        "adaptive_prefix": adaptive_prefix,
        "n_runs": len(payloads),
        "competition_noise_gain_vs_full_router": _mean_std(
            [payload["global"]["competition_noise_gain_vs_full_router"] for payload in payloads]
        ),
        "best_pair_gain_vs_full_expert": {
            **_mean_std([payload["global"]["best_pair_gain_vs_full_expert"] for payload in payloads]),
            "positive_fraction": float(
                (
                    np.asarray([payload["global"]["best_pair_gain_vs_full_expert"] for payload in payloads])
                    > 0.0
                ).mean()
            ),
        },
        "best_view_capture_gap_vs_fixed": _mean_std(
            [payload["best_view_tradeoff"]["capture_gap_vs_fixed"] for payload in payloads]
        ),
        "classification_counts": classifications,
        "best_view_counts": best_views,
    }


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Signal vs Noise Suite Summary",
        "",
        f"- root: `{summary['root']}`",
        f"- adaptive_prefix: `{summary['adaptive_prefix']}`",
        f"- n_runs: `{summary['n_runs']}`",
        f"- competition-noise gain vs full router: `{summary['competition_noise_gain_vs_full_router']['mean']:.6f} ± {summary['competition_noise_gain_vs_full_router']['std']:.6f}`",
        f"- best pair gain vs FULL expert: `{summary['best_pair_gain_vs_full_expert']['mean']:.6f} ± {summary['best_pair_gain_vs_full_expert']['std']:.6f}`",
        f"- positive fraction vs FULL expert: `{summary['best_pair_gain_vs_full_expert']['positive_fraction']:.3f}`",
        f"- best-view capture gap vs fixed: `{summary['best_view_capture_gap_vs_fixed']['mean']:.6f} ± {summary['best_view_capture_gap_vs_fixed']['std']:.6f}`",
        "",
        "## Classification Counts",
        "",
    ]
    for key, count in summary["classification_counts"].items():
        lines.append(f"- {key}: `{count}`")
    lines.extend(["", "## Best View Counts", ""])
    for key, count in summary["best_view_counts"].items():
        lines.append(f"- {key}: `{count}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate signal-vs-noise summaries")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    args = parser.parse_args()

    summary = summarize(args.root, adaptive_prefix=args.adaptive_prefix)
    out_json = args.root / f"{args.adaptive_prefix}_signal_noise_suite_summary.json"
    out_md = args.root / f"{args.adaptive_prefix}_signal_noise_suite_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
