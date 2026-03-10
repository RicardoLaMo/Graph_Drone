#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


DEFAULT_REPORTS = Path("experiments/tabpfn_view_router/reports")
DEFAULT_MODELS = ("P0_FULL", "P0_router", "P0_crossfit", "P0_uniform", "P0_sigma2", "P0_gora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P0 OpenML split-sweep results.")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORTS / "p0_openml_split_sweep.md",
    )
    return parser.parse_args()


def load_payloads(reports_root: Path, model_seed: int, split_seeds: list[int]) -> list[dict]:
    payloads = []
    for split_seed in split_seeds:
        path = reports_root / f"p0_openml_split{split_seed}_seed{model_seed}_h200" / "p0_results.json"
        payloads.append(json.loads(path.read_text()))
    return payloads


def summarize(payloads: list[dict], model_seed: int) -> str:
    refs = payloads[0]["references"]
    rows_by_model: dict[str, list[tuple[int, float]]] = {}
    for payload in payloads:
        split_seed = int(payload["split_seed"])
        for row in payload["results"]:
            rows_by_model.setdefault(row["model"], []).append((split_seed, float(row["test_rmse"])))

    lines = [
        "# P0 OpenML Split Sweep Summary",
        "",
        f"- dataset source: `{payloads[0]['dataset']['source']}`",
        f"- OpenML dataset id: `{payloads[0]['dataset']['openml_dataset_id']}`",
        f"- model_seed: `{model_seed}`",
        "",
        "## Split Table",
        "",
        "| Split Seed | P0_FULL | P0_router | P0_crossfit |",
        "|---|---:|---:|---:|",
    ]
    for payload in payloads:
        per_model = {row["model"]: float(row["test_rmse"]) for row in payload["results"]}
        lines.append(
            "| "
            f"{payload['split_seed']} | "
            f"{per_model['P0_FULL']:.4f} | "
            f"{per_model['P0_router']:.4f} | "
            f"{per_model['P0_crossfit']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Across-Split Means",
            "",
            "| Model | Mean RMSE | Std | Wins vs TabR | Wins vs TabPFN best |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    tabr = float(refs["tabr_rmse"])
    tabpfn_best = float(refs["tabpfn_full_best"])

    for model in DEFAULT_MODELS:
        vals = [rmse for _, rmse in rows_by_model[model]]
        mean = st.mean(vals)
        std = st.pstdev(vals)
        wins_tabr = sum(1 for value in vals if value < tabr)
        wins_tabpfn = sum(1 for value in vals if value < tabpfn_best)
        lines.append(
            f"| {model} | {mean:.4f} | {std:.4f} | "
            f"{wins_tabr}/{len(vals)} | "
            f"{wins_tabpfn}/{len(vals)} |"
        )

    lines.extend(
        [
            "",
            "## Reference Anchors",
            "",
            f"- TabR_on_our_split: `{refs['tabr_rmse']:.4f}`",
            f"- TabM_on_our_split: `{refs['tabm_rmse']:.4f}`",
            f"- TabPFN_full_best: `{refs['tabpfn_full_best']:.4f}`",
            f"- TabPFN_full_multiseed_mean: `{refs['tabpfn_full_mean']:.4f}`",
            f"- MV-TabR-GoRA A6f artifact: `{refs['a6f_artifact']:.4f}`",
            "",
            "## Notes",
            "",
            "- This sweep varies `split_seed` while keeping the model seed fixed.",
            "- Use this as a split-dependence check, not as a full statistical significance study.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payloads = load_payloads(args.reports_root, args.model_seed, args.split_seeds)
    markdown = summarize(payloads, args.model_seed)
    args.output.write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
