#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tab_foundation_compare.src.runtime_support import (
    ALIGNED_CANONICAL_SEED,
    seed_aware_run_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize aligned TabR/TabM/TabPFN split sweeps.")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--tab-foundation-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "tab_foundation_compare",
    )
    parser.add_argument(
        "--tabm-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "tabm_california_baseline",
    )
    parser.add_argument(
        "--p0-reports-root",
        type=Path,
        default=REPO_ROOT.parent / "p0-view-router" / "experiments" / "tabpfn_view_router" / "reports",
    )
    parser.add_argument("--p0-model-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "tab_foundation_compare" / "reports" / "foundation_split_sweep.md",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _tabr_report_path(root: Path, split_seed: int) -> Path:
    run_name = seed_aware_run_name(
        "tabr__0-evaluation__0",
        split_seed,
        canonical_seed=ALIGNED_CANONICAL_SEED,
        smoke=False,
    )
    return root / "reports" / f"{run_name}.json"


def _tabm_report_path(root: Path, split_seed: int) -> Path:
    run_name = seed_aware_run_name(
        "0-evaluation__0",
        split_seed,
        canonical_seed=ALIGNED_CANONICAL_SEED,
        smoke=False,
    )
    return root / "reports" / f"{run_name}.json"


def _tabpfn_report_path(root: Path, split_seed: int) -> Path:
    run_name = seed_aware_run_name(
        "tabpfn_aligned__full",
        split_seed,
        canonical_seed=ALIGNED_CANONICAL_SEED,
        smoke=False,
    )
    return root / "reports" / f"{run_name}.json"


def _load_p0_rows(root: Path, split_seeds: list[int], model_seed: int) -> dict[str, list[tuple[int, float]]]:
    rows: dict[str, list[tuple[int, float]]] = {}
    for split_seed in split_seeds:
        payload = _load_json(root / f"p0_openml_split{split_seed}_seed{model_seed}_h200" / "p0_results.json")
        for row in payload["results"]:
            rows.setdefault(row["model"], []).append((split_seed, float(row["test_rmse"])))
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    return st.mean(values), st.pstdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    tabr_rows: list[tuple[int, float]] = []
    tabm_rows: list[tuple[int, float]] = []
    tabpfn_rows: list[tuple[int, float]] = []
    for split_seed in args.split_seeds:
        tabr_rows.append((split_seed, float(_load_json(_tabr_report_path(args.tab_foundation_root, split_seed))["metrics"]["test"]["rmse"])))
        tabm_rows.append((split_seed, float(_load_json(_tabm_report_path(args.tabm_root, split_seed))["metrics"]["test"]["rmse"])))
        tabpfn_path = _tabpfn_report_path(args.tab_foundation_root, split_seed)
        if tabpfn_path.exists():
            tabpfn_rows.append((split_seed, float(_load_json(tabpfn_path)["test"]["rmse"])))

    p0_rows = _load_p0_rows(args.p0_reports_root, args.split_seeds, args.p0_model_seed)
    p0_router = p0_rows.get("P0_router", [])
    p0_full = p0_rows.get("P0_FULL", [])
    p0_crossfit = p0_rows.get("P0_crossfit", [])

    lines = [
        "# Foundation Split Sweep Summary",
        "",
        "| Split Seed | TabR | TabM | TabPFN_full | P0_router | P0_full | P0_crossfit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tabpfn_lookup = dict(tabpfn_rows)
    p0_router_lookup = dict(p0_router)
    p0_full_lookup = dict(p0_full)
    p0_crossfit_lookup = dict(p0_crossfit)
    for split_seed, tabr_rmse in tabr_rows:
        lines.append(
            f"| {split_seed} | "
            f"{tabr_rmse:.4f} | "
            f"{dict(tabm_rows)[split_seed]:.4f} | "
            f"{tabpfn_lookup.get(split_seed, float('nan')):.4f} | "
            f"{p0_router_lookup.get(split_seed, float('nan')):.4f} | "
            f"{p0_full_lookup.get(split_seed, float('nan')):.4f} | "
            f"{p0_crossfit_lookup.get(split_seed, float('nan')):.4f} |"
        )

    lines.extend(["", "## Means", "", "| Model | Mean RMSE | Std |", "|---|---:|---:|"])
    for label, rows in [
        ("TabR", tabr_rows),
        ("TabM", tabm_rows),
        ("TabPFN_full", tabpfn_rows),
        ("P0_router", p0_router),
        ("P0_full", p0_full),
        ("P0_crossfit", p0_crossfit),
    ]:
        if not rows:
            continue
        mean, std = _mean_std([value for _, value in rows])
        lines.append(f"| {label} | {mean:.4f} | {std:.4f} |")

    if p0_router:
        wins_vs_tabr = sum(
            1
            for split_seed, p0_rmse in p0_router
            if p0_rmse < dict(tabr_rows)[split_seed]
        )
        lines.extend(
            [
                "",
                "## Notes",
                "",
                f"- `P0_router` beats `TabR` on `{wins_vs_tabr}/{len(p0_router)}` matched split seeds.",
                "- This summary varies split seed while leaving each model family on its own fixed training seed/config path.",
            ]
        )

    markdown = "\n".join(lines) + "\n"
    args.output.write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
