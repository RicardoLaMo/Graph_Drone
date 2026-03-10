from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "experiments" / "tabarena_bridge" / "configs" / "benchmark_manifest.json"


def load_manifest() -> dict:
    with MANIFEST_PATH.open() as f:
        return json.load(f)


def sorted_datasets(manifest: dict) -> list[dict]:
    return sorted(
        manifest["benchmark"]["datasets"],
        key=lambda row: (row["priority_tier"], row["name"]),
    )


def select_datasets(manifest: dict, max_tier: int) -> list[dict]:
    return [row for row in sorted_datasets(manifest) if row["priority_tier"] <= max_tier]
