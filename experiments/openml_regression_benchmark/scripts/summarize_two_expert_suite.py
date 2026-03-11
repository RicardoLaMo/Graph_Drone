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
    paths = sorted(root.glob(f"**/artifacts/{adaptive_prefix}_two_expert_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No {adaptive_prefix}_two_expert_summary.json files found under {root}")

    payloads = [json.loads(path.read_text()) for path in paths]
    best_gains = [payload["best_two_expert"]["adaptive_minus_full_router"] for payload in payloads]
    best_vs_full = [payload["best_two_expert"]["adaptive_minus_full_expert"] for payload in payloads]
    best_views = [payload["best_candidate_view"] for payload in payloads]

    counts: dict[str, int] = {}
    for view in best_views:
        counts[view] = counts.get(view, 0) + 1

    return {
        "root": str(root),
        "adaptive_prefix": adaptive_prefix,
        "n_runs": len(payloads),
        "best_two_expert_gain_vs_full_router": {
            **_mean_std(best_gains),
            "positive_fraction": float((np.asarray(best_gains) > 0.0).mean()),
        },
        "best_two_expert_gain_vs_full_expert": {
            **_mean_std(best_vs_full),
            "positive_fraction": float((np.asarray(best_vs_full) > 0.0).mean()),
        },
        "best_candidate_counts": counts,
    }


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Two-Expert Suite Summary",
        "",
        f"- root: `{summary['root']}`",
        f"- adaptive_prefix: `{summary['adaptive_prefix']}`",
        f"- n_runs: `{summary['n_runs']}`",
        f"- best two-expert gain vs full router: `{summary['best_two_expert_gain_vs_full_router']['mean']:.6f} ± {summary['best_two_expert_gain_vs_full_router']['std']:.6f}`",
        f"- positive fraction vs full router: `{summary['best_two_expert_gain_vs_full_router']['positive_fraction']:.3f}`",
        f"- best two-expert gain vs full expert: `{summary['best_two_expert_gain_vs_full_expert']['mean']:.6f} ± {summary['best_two_expert_gain_vs_full_expert']['std']:.6f}`",
        f"- positive fraction vs full expert: `{summary['best_two_expert_gain_vs_full_expert']['positive_fraction']:.3f}`",
        "",
        "## Best Candidate Counts",
        "",
    ]
    for view, count in summary["best_candidate_counts"].items():
        lines.append(f"- {view}: `{count}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate two-expert competition summaries")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    args = parser.parse_args()

    summary = summarize(args.root, adaptive_prefix=args.adaptive_prefix)
    out_json = args.root / f"{args.adaptive_prefix}_two_expert_suite_summary.json"
    out_md = args.root / f"{args.adaptive_prefix}_two_expert_suite_summary.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
