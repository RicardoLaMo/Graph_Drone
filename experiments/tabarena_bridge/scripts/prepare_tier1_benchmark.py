from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tabarena_bridge.src.manifest import load_manifest, select_datasets

ARTIFACT_DIR = ROOT / "experiments" / "tabarena_bridge" / "artifacts"


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "priority_tier",
        "name",
        "openml_dataset_id",
        "openml_task_id",
        "rows",
        "features",
        "view_family",
        "integration_phase",
        "graph_drone_role",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_json(rows: list[dict], path: Path, split_seeds: list[int]) -> None:
    path.write_text(json.dumps({"split_seeds": split_seeds, "datasets": rows}, indent=2) + "\n")


def write_runbook(rows: list[dict], path: Path, split_seeds: list[int], baselines: list[str]) -> None:
    lines: list[str] = []
    lines.append("# Tier 1 TabArena Runbook")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Split seeds: `{', '.join(map(str, split_seeds))}`")
    lines.append(f"- Baselines: `{', '.join(baselines)}`")
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['name']}")
        lines.append(f"- OpenML dataset id: `{row['openml_dataset_id']}`")
        lines.append(f"- OpenML task id: `{row['openml_task_id']}`")
        lines.append(f"- View family: `{row['view_family']}`")
        lines.append(f"- Integration phase: `{row['integration_phase']}`")
        lines.append(f"- Role: `{row['graph_drone_role']}`")
        lines.append(f"- Why: {row['why']}")
        lines.append("")
    lines.append("## H200 Execution Order")
    lines.append("")
    lines.append("1. Reproduce external baselines on all Tier 1 datasets.")
    lines.append("2. Freeze split sweep outputs.")
    lines.append("3. Add Graph_Drone custom adapters only after the baseline protocol is stable.")
    lines.append("4. Start custom-model transfer on geo_housing datasets before generic_numeric datasets.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tier", type=int, default=1)
    parser.add_argument("--output-prefix", default="tier1")
    args = parser.parse_args()

    manifest = load_manifest()
    rows = select_datasets(manifest, args.max_tier)
    split_seeds = manifest["benchmark"]["split_policy"]["split_seeds"]
    baselines = manifest["benchmark"]["baseline_models"]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACT_DIR / f"{args.output_prefix}_matrix.csv"
    json_path = ARTIFACT_DIR / f"{args.output_prefix}_bundle.json"
    md_path = ARTIFACT_DIR / f"{args.output_prefix}_runbook.md"

    write_csv(rows, csv_path)
    write_json(rows, json_path, split_seeds)
    write_runbook(rows, md_path, split_seeds, baselines)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
