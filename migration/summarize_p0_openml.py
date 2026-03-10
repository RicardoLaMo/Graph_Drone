#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


DEFAULT_REPORTS = Path("experiments/tabpfn_view_router/reports")
DEFAULT_MODELS = ("P0_FULL", "P0_router", "P0_crossfit", "P0_uniform", "P0_sigma2", "P0_gora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P0 OpenML multiseed results.")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORTS / "p0_openml_multiseed.md",
    )
    return parser.parse_args()


def load_payloads(reports_root: Path, seeds: list[int]) -> list[dict]:
    payloads = []
    for seed in seeds:
        path = reports_root / f"p0_openml_seed{seed}_h200" / "p0_results.json"
        payloads.append(json.loads(path.read_text()))
    return payloads


def summarize(payloads: list[dict]) -> str:
    refs = payloads[0]["references"]
    rows_by_model: dict[str, list[tuple[int, float]]] = {}
    for payload in payloads:
        seed = int(payload["seed"])
        for row in payload["results"]:
            rows_by_model.setdefault(row["model"], []).append((seed, float(row["test_rmse"])))

    lines = [
        "# P0 OpenML Multiseed Summary",
        "",
        f"- dataset source: `{payloads[0]['dataset']['source']}`",
        f"- OpenML dataset id: `{payloads[0]['dataset']['openml_dataset_id']}`",
        f"- split_seed: `{payloads[0]['split_seed']}`",
        "",
        "## Seed Table",
        "",
        "| Seed | P0_FULL | P0_router | P0_crossfit |",
        "|---|---:|---:|---:|",
    ]
    for payload in payloads:
        per_model = {row["model"]: float(row["test_rmse"]) for row in payload["results"]}
        lines.append(
            "| "
            f"{payload['seed']} | "
            f"{per_model['P0_FULL']:.4f} | "
            f"{per_model['P0_router']:.4f} | "
            f"{per_model['P0_crossfit']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Multiseed Means",
            "",
            "| Model | Mean RMSE | Std | vs TabR | vs TabM | vs TabPFN best | vs TabPFN mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for model in DEFAULT_MODELS:
        vals = [rmse for _, rmse in rows_by_model[model]]
        mean = st.mean(vals)
        std = st.pstdev(vals)
        lines.append(
            f"| {model} | {mean:.4f} | {std:.4f} | "
            f"{mean - refs['tabr_rmse']:+.4f} | "
            f"{mean - refs['tabm_rmse']:+.4f} | "
            f"{mean - refs['tabpfn_full_best']:+.4f} | "
            f"{mean - refs['tabpfn_full_mean']:+.4f} |"
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
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payloads = load_payloads(args.reports_root, args.seeds)
    markdown = summarize(payloads)
    args.output.write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
