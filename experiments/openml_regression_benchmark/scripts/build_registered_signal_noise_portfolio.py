from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.scripts.analyze_router_full_regret import (
    analyze_run as analyze_full_regret_run,
)
from experiments.openml_regression_benchmark.scripts.analyze_signal_noise_tradeoff import (
    summarize_components,
)
from experiments.openml_regression_benchmark.scripts.analyze_two_expert_competition import (
    analyze_run as analyze_two_expert_run,
)
from experiments.openml_regression_benchmark.scripts.analyze_view_home_quality import (
    analyze_run as analyze_view_home_run,
)
from experiments.openml_regression_benchmark.scripts.summarize_signal_noise_suite import (
    summarize_payloads,
)


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def _fraction(values: list[bool]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def _load_catalog(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _analyze_run(run_dir: Path, *, adaptive_prefix: str) -> dict[str, object]:
    full_regret = analyze_full_regret_run(run_dir, adaptive_prefix=adaptive_prefix, label="GraphDrone")
    view_home = analyze_view_home_run(run_dir, adaptive_prefix=adaptive_prefix, label="GraphDrone")
    two_expert = analyze_two_expert_run(run_dir, adaptive_prefix=adaptive_prefix, label="GraphDrone", seed=42)
    signal_noise = summarize_components(
        run_dir=run_dir,
        full_regret=full_regret,
        view_home=view_home,
        two_expert=two_expert,
    )
    signal_noise["provenance"] = full_regret["provenance"]
    return signal_noise


def _derive_dataset_verdict(signal_noise_summary: dict[str, object]) -> str:
    competition_gain = signal_noise_summary["competition_noise_gain_vs_full_router"]["mean"]
    best_pair_gain = signal_noise_summary["best_pair_gain_vs_full_expert"]["mean"]
    if competition_gain > 0.0 and best_pair_gain > 0.0:
        return "useful_signal_obscured_by_competition"
    if competition_gain > 0.0 and best_pair_gain <= 0.0:
        return "competition_noise_plus_weak_expert"
    if competition_gain <= 0.0 and best_pair_gain <= 0.0:
        return "no_useful_non_full_signal"
    return "weak_competition_effect"


def build_portfolio(catalog: dict[str, object], *, adaptive_prefix: str) -> dict[str, object]:
    dataset_summaries: list[dict[str, object]] = []
    portfolio_best_views: dict[str, int] = {}
    portfolio_verdicts: dict[str, int] = {}

    for dataset in catalog["datasets"]:
        key = dataset["key"]
        run_dirs = [Path(path) for path in dataset["source_runs"]]
        run_payloads = [_analyze_run(path, adaptive_prefix=adaptive_prefix) for path in run_dirs]
        suite_summary = summarize_payloads(
            run_payloads,
            root_label=key,
            adaptive_prefix=adaptive_prefix,
        )

        best_view = max(suite_summary["best_view_counts"], key=suite_summary["best_view_counts"].get)
        verdict = _derive_dataset_verdict(suite_summary)
        fixed_modes: dict[str, int] = {}
        quality_modes: dict[str, int] = {}
        for payload in run_payloads:
            fixed_mode = payload["provenance"]["fixed_mode"]
            quality_mode = payload["provenance"]["quality_mode"]
            fixed_modes[fixed_mode] = fixed_modes.get(fixed_mode, 0) + 1
            quality_modes[quality_mode] = quality_modes.get(quality_mode, 0) + 1

        stability = None
        if "stability_root" in dataset:
            stability_root = Path(dataset["stability_root"])
            stability_payloads = [
                _analyze_run(path.parent, adaptive_prefix=adaptive_prefix)
                for path in sorted(stability_root.glob("seed*/houses__r0f0/graphdrone_results.json"))
            ]
            stability = summarize_payloads(
                stability_payloads,
                root_label=str(stability_root),
                adaptive_prefix=adaptive_prefix,
            )

        dataset_summary = {
            "dataset": key,
            "n_runs": len(run_payloads),
            "best_view": best_view,
            "verdict": verdict,
            "signal_noise": suite_summary,
            "provenance": {
                "fixed_modes": fixed_modes,
                "quality_modes": quality_modes,
            },
        }
        if stability is not None:
            dataset_summary["stability_probe"] = stability
        dataset_summaries.append(dataset_summary)
        portfolio_best_views[best_view] = portfolio_best_views.get(best_view, 0) + 1
        portfolio_verdicts[verdict] = portfolio_verdicts.get(verdict, 0) + 1

    best_pair_positive = [
        summary["signal_noise"]["best_pair_gain_vs_full_expert"]["mean"] > 0.0
        for summary in dataset_summaries
    ]
    competition_gain_positive = [
        summary["signal_noise"]["competition_noise_gain_vs_full_router"]["mean"] > 0.0
        for summary in dataset_summaries
    ]

    return {
        "portfolio_id": catalog["portfolio_id"],
        "adaptive_prefix": adaptive_prefix,
        "n_datasets": len(dataset_summaries),
        "dataset_summaries": dataset_summaries,
        "portfolio_rollup": {
            "best_view_counts": portfolio_best_views,
            "verdict_counts": portfolio_verdicts,
            "datasets_with_positive_best_pair_gain_fraction": _fraction(best_pair_positive),
            "datasets_with_positive_competition_gain_fraction": _fraction(competition_gain_positive),
            "competition_gain_dataset_mean": _mean_std(
                [summary["signal_noise"]["competition_noise_gain_vs_full_router"]["mean"] for summary in dataset_summaries]
            ),
            "best_pair_gain_dataset_mean": _mean_std(
                [summary["signal_noise"]["best_pair_gain_vs_full_expert"]["mean"] for summary in dataset_summaries]
            ),
        },
    }


def write_markdown(path: Path, portfolio: dict[str, object]) -> None:
    lines = [
        "# Registered Portfolio Signal vs Noise",
        "",
        f"- portfolio_id: `{portfolio['portfolio_id']}`",
        f"- adaptive_prefix: `{portfolio['adaptive_prefix']}`",
        f"- n_datasets: `{portfolio['n_datasets']}`",
        f"- datasets with positive best-pair gain vs FULL: `{portfolio['portfolio_rollup']['datasets_with_positive_best_pair_gain_fraction']:.3f}`",
        f"- datasets with positive competition-noise gain: `{portfolio['portfolio_rollup']['datasets_with_positive_competition_gain_fraction']:.3f}`",
        "",
        "| Dataset | Best View | Verdict | CompNoise Gain vs Router | Best Pair Gain vs FULL | Capture Gap vs Fixed | Fixed Mode | Quality Mode |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for summary in portfolio["dataset_summaries"]:
        signal = summary["signal_noise"]
        fixed_mode = max(summary["provenance"]["fixed_modes"], key=summary["provenance"]["fixed_modes"].get)
        quality_mode = max(summary["provenance"]["quality_modes"], key=summary["provenance"]["quality_modes"].get)
        lines.append(
            f"| {summary['dataset']} | {summary['best_view']} | {summary['verdict']} | "
            f"{signal['competition_noise_gain_vs_full_router']['mean']:.4f} | "
            f"{signal['best_pair_gain_vs_full_expert']['mean']:.4f} | "
            f"{signal['best_view_capture_gap_vs_fixed']['mean']:.4f} | "
            f"{fixed_mode} | {quality_mode} |"
        )
    lines.extend(["", "## Dataset Notes", ""])
    for summary in portfolio["dataset_summaries"]:
        signal = summary["signal_noise"]
        lines.append(f"### {summary['dataset']}")
        lines.append("")
        lines.append(f"- best view: `{summary['best_view']}`")
        lines.append(f"- verdict: `{summary['verdict']}`")
        lines.append(
            f"- competition-noise gain vs full router: `{signal['competition_noise_gain_vs_full_router']['mean']:.6f} ± {signal['competition_noise_gain_vs_full_router']['std']:.6f}`"
        )
        lines.append(
            f"- best pair gain vs FULL expert: `{signal['best_pair_gain_vs_full_expert']['mean']:.6f} ± {signal['best_pair_gain_vs_full_expert']['std']:.6f}`"
        )
        lines.append(
            f"- best-view capture gap vs fixed: `{signal['best_view_capture_gap_vs_fixed']['mean']:.6f} ± {signal['best_view_capture_gap_vs_fixed']['std']:.6f}`"
        )
        lines.append(f"- best view counts: `{json.dumps(signal['best_view_counts'], sort_keys=True)}`")
        if "stability_probe" in summary:
            stability = summary["stability_probe"]
            lines.append(
                f"- stability probe capture gap vs fixed: `{stability['best_view_capture_gap_vs_fixed']['mean']:.6f} ± {stability['best_view_capture_gap_vs_fixed']['std']:.6f}`"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build registered-dataset signal/noise portfolio summary")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "configs" / "registered_signal_noise_sources.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports_phase0f_registered_portfolio",
    )
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = _load_catalog(args.catalog)
    adaptive_prefix = args.adaptive_prefix or catalog.get("adaptive_prefix", "router")
    portfolio = build_portfolio(catalog, adaptive_prefix=adaptive_prefix)
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_json = args.output_root / "registered_signal_noise_portfolio.json"
    out_md = args.output_root / "registered_signal_noise_portfolio.md"
    out_json.write_text(json.dumps(portfolio, indent=2) + "\n")
    write_markdown(out_md, portfolio)
    print(json.dumps(portfolio, indent=2))


if __name__ == "__main__":
    main()
