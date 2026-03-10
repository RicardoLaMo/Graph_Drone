from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments" / "tabarena_bridge" / "configs" / "benchmark_manifest.json"
SCRIPT_PATH = ROOT / "experiments" / "tabarena_bridge" / "scripts" / "render_benchmark_plan.py"
CSV_PATH = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "benchmark_matrix.csv"
MD_PATH = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "benchmark_matrix.md"


def test_manifest_has_expected_suite_and_priority_datasets() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    benchmark = manifest["benchmark"]
    assert benchmark["source_suite"]["openml_suite_id"] == 457
    names = {row["name"] for row in benchmark["datasets"]}
    assert {"miami_housing", "houses", "diamonds"}.issubset(names)


def test_render_script_generates_csv_and_markdown() -> None:
    subprocess.run([sys.executable, str(SCRIPT_PATH)], check=True, cwd=ROOT)
    assert CSV_PATH.exists()
    assert MD_PATH.exists()
    assert "Graph_Drone TabArena Benchmark Matrix" in MD_PATH.read_text()
