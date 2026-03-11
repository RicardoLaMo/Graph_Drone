from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _classify(best_pair_gain_vs_full: float, competition_noise_gain: float) -> str:
    if competition_noise_gain <= 0.0 and best_pair_gain_vs_full <= 0.0:
        return "no_useful_non_full_signal"
    if competition_noise_gain > 0.0 and best_pair_gain_vs_full > 0.0:
        return "useful_signal_obscured_by_competition"
    if competition_noise_gain > 0.0 and best_pair_gain_vs_full <= 0.0:
        return "competition_noise_plus_weak_expert"
    return "weak_competition_effect"


def summarize_components(
    *,
    run_dir: Path,
    full_regret: dict[str, object],
    view_home: dict[str, object],
    two_expert: dict[str, object],
) -> dict[str, object]:
    best_view = str(two_expert["best_candidate_view"])
    best_pair = two_expert["best_two_expert"]
    best_home = view_home["per_view_home_subset"][best_view]

    competition_noise_gain = float(best_pair["adaptive_minus_full_router"])
    best_pair_gain_vs_full = float(best_pair["adaptive_minus_full_expert"])
    best_pair_gain_vs_fixed = float(best_pair["adaptive_minus_full_fixed"])

    return {
        "run_dir": str(run_dir),
        "best_view": best_view,
        "best_pair_name": best_pair["pair_name"],
        "classification": _classify(best_pair_gain_vs_full, competition_noise_gain),
        "global": {
            "full_oracle_fraction": float(full_regret["global"]["full_oracle_fraction"]),
            "full_router_test_rmse": float(two_expert["global_reference"]["full_router_test_rmse"]),
            "full_expert_test_rmse": float(two_expert["global_reference"]["full_expert_test_rmse"]),
            "best_two_expert_test_rmse": float(best_pair["adaptive_test_rmse"]),
            "competition_noise_gain_vs_full_router": competition_noise_gain,
            "best_pair_gain_vs_full_expert": best_pair_gain_vs_full,
            "best_pair_gain_vs_full_fixed": best_pair_gain_vs_fixed,
        },
        "best_view_tradeoff": {
            "row_fraction": float(best_home["row_fraction"]),
            "mean_potential_gain_vs_full": float(best_home["mean_potential_gain_vs_full"]),
            "adaptive_capture_ratio_total": float(best_home["adaptive_capture_ratio_total"]),
            "fixed_capture_ratio_total": float(best_home["fixed_capture_ratio_total"]),
            "capture_gap_vs_fixed": float(best_home["capture_gap_vs_fixed"]),
            "mean_adaptive_full_weight_on_home": float(best_home["mean_adaptive_full_weight"]),
            "mean_adaptive_view_weight_on_home": float(best_home["mean_adaptive_view_weight"]),
        },
        "full_home_damage_control": {
            "false_diversion_mean_cost": float(full_regret["full_oracle_case"]["false_diversion_mean_cost"]),
            "false_diversion_positive_fraction": float(full_regret["full_oracle_case"]["false_diversion_positive_fraction"]),
        },
        "non_full_home_harvest": {
            "mean_potential_gain": float(full_regret["non_full_oracle_case"]["mean_potential_gain"]),
            "mean_adaptive_realized_gain": float(full_regret["non_full_oracle_case"]["mean_adaptive_realized_gain"]),
            "adaptive_capture_ratio_total": float(full_regret["non_full_oracle_case"]["adaptive_capture_ratio_total"]),
            "fixed_capture_ratio_total": float(full_regret["non_full_oracle_case"]["fixed_capture_ratio_total"]),
        },
    }


def analyze_run(run_dir: Path, *, adaptive_prefix: str = "router") -> dict[str, object]:
    artifacts = run_dir / "artifacts"
    full_regret = _load_json(artifacts / f"{adaptive_prefix}_full_regret_summary.json")
    view_home = _load_json(artifacts / f"{adaptive_prefix}_view_home_summary.json")
    two_expert = _load_json(artifacts / f"{adaptive_prefix}_two_expert_summary.json")
    return summarize_components(
        run_dir=run_dir,
        full_regret=full_regret,
        view_home=view_home,
        two_expert=two_expert,
    )


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Signal vs Noise Tradeoff",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- best view: `{summary['best_view']}`",
        f"- best pair: `{summary['best_pair_name']}`",
        f"- classification: `{summary['classification']}`",
        f"- competition-noise gain vs full router: `{summary['global']['competition_noise_gain_vs_full_router']:.4f}`",
        f"- best pair gain vs FULL expert: `{summary['global']['best_pair_gain_vs_full_expert']:.4f}`",
        f"- full oracle fraction: `{summary['global']['full_oracle_fraction']:.3f}`",
        "",
        "## Best View Tradeoff",
        "",
        f"- row fraction: `{summary['best_view_tradeoff']['row_fraction']:.3f}`",
        f"- mean potential gain vs FULL: `{summary['best_view_tradeoff']['mean_potential_gain_vs_full']:.4f}`",
        f"- adaptive capture ratio: `{summary['best_view_tradeoff']['adaptive_capture_ratio_total']:.4f}`",
        f"- fixed capture ratio: `{summary['best_view_tradeoff']['fixed_capture_ratio_total']:.4f}`",
        f"- capture gap vs fixed: `{summary['best_view_tradeoff']['capture_gap_vs_fixed']:.4f}`",
        "",
        "## Mechanism Split",
        "",
        f"- FULL-home false diversion mean cost: `{summary['full_home_damage_control']['false_diversion_mean_cost']:.4f}`",
        f"- non-FULL mean potential gain: `{summary['non_full_home_harvest']['mean_potential_gain']:.4f}`",
        f"- non-FULL adaptive realized gain: `{summary['non_full_home_harvest']['mean_adaptive_realized_gain']:.4f}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize view signal vs competition noise for one run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adaptive-prefix", choices=["router", "crossfit"], default="router")
    args = parser.parse_args()

    summary = analyze_run(args.run_dir, adaptive_prefix=args.adaptive_prefix)
    out_json = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_signal_noise_tradeoff.json"
    out_md = args.run_dir / "artifacts" / f"{args.adaptive_prefix}_signal_noise_tradeoff.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(out_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
