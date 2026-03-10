from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "experiments" / "tabarena_bridge" / "configs" / "benchmark_manifest.json"
ARTIFACT_DIR = ROOT / "experiments" / "tabarena_bridge" / "artifacts"
CSV_PATH = ARTIFACT_DIR / "benchmark_matrix.csv"
MD_PATH = ARTIFACT_DIR / "benchmark_matrix.md"


def load_manifest() -> dict:
    with MANIFEST_PATH.open() as f:
        return json.load(f)


def normalize_rows(manifest: dict) -> list[dict]:
    rows = manifest["benchmark"]["datasets"]
    return sorted(rows, key=lambda row: (row["priority_tier"], row["name"]))


def write_csv(rows: list[dict]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority_tier",
        "name",
        "openml_dataset_id",
        "openml_task_id",
        "problem_type",
        "rows",
        "features",
        "graph_drone_role",
        "why",
    ]
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_markdown(rows: list[dict], manifest: dict) -> None:
    lines = []
    lines.append("# Graph_Drone TabArena Benchmark Matrix")
    lines.append("")
    lines.append(f"- Source suite: `{manifest['benchmark']['source_suite']['name']}`")
    lines.append(f"- OpenML suite id: `{manifest['benchmark']['source_suite']['openml_suite_id']}`")
    lines.append(f"- Primary metric: `{manifest['benchmark']['primary_metric']}`")
    lines.append(f"- Split seeds: `{', '.join(map(str, manifest['benchmark']['split_policy']['split_seeds']))}`")
    lines.append("")
    lines.append("| Tier | Dataset | OpenML DID | OpenML TID | Rows | Features | Role | Why |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['priority_tier']} | {row['name']} | {row['openml_dataset_id']} | "
            f"{row['openml_task_id']} | {row['rows']} | {row['features']} | "
            f"{row['graph_drone_role']} | {row['why']} |"
        )
    MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    manifest = load_manifest()
    rows = normalize_rows(manifest)
    write_csv(rows)
    write_markdown(rows, manifest)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
