from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments" / "tabarena_bridge" / "configs" / "benchmark_manifest.json"
SCRIPT_PATH = ROOT / "experiments" / "tabarena_bridge" / "scripts" / "render_benchmark_plan.py"
PREPARE_PATH = ROOT / "experiments" / "tabarena_bridge" / "scripts" / "prepare_tier1_benchmark.py"
RUNNER_PATH = ROOT / "experiments" / "tabarena_bridge" / "scripts" / "run_tier1_baselines.py"
CSV_PATH = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "benchmark_matrix.csv"
MD_PATH = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "benchmark_matrix.md"
TIER1_CSV = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "tier1_matrix.csv"
TIER1_JSON = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "tier1_bundle.json"
TIER1_MD = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "tier1_runbook.md"


def _python_executable() -> str:
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def test_manifest_has_expected_suite_and_priority_datasets() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    benchmark = manifest["benchmark"]
    assert benchmark["source_suite"]["openml_suite_id"] == 457
    names = {row["name"] for row in benchmark["datasets"]}
    assert {"miami_housing", "houses", "diamonds"}.issubset(names)
    assert all("view_family" in row for row in benchmark["datasets"])
    assert all("integration_phase" in row for row in benchmark["datasets"])


def test_render_script_generates_csv_and_markdown() -> None:
    subprocess.run([_python_executable(), str(SCRIPT_PATH)], check=True, cwd=ROOT)
    assert CSV_PATH.exists()
    assert MD_PATH.exists()
    assert "Graph_Drone TabArena Benchmark Matrix" in MD_PATH.read_text()


def test_prepare_script_generates_tier1_bundle() -> None:
    subprocess.run([_python_executable(), str(PREPARE_PATH)], check=True, cwd=ROOT)
    assert TIER1_CSV.exists()
    assert TIER1_JSON.exists()
    assert TIER1_MD.exists()
    bundle = json.loads(TIER1_JSON.read_text())
    assert bundle["split_seeds"] == [42, 43, 44, 45, 46]
    assert {row["name"] for row in bundle["datasets"]} == {"miami_housing", "houses", "diamonds"}


def test_runner_dry_smoke_houses() -> None:
    out_dir = ROOT / "experiments" / "tabarena_bridge" / "reports" / "tier1_smoke_test"
    subprocess.run(
        [
            _python_executable(),
            str(RUNNER_PATH),
            "--dataset",
            "houses",
            "--split-seed",
            "42",
            "--seed",
            "42",
            "--smoke",
            "--output",
            str(out_dir),
        ],
        check=True,
        cwd=ROOT,
    )
    result_path = out_dir / "houses__split42__seed42" / "results.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text())
    assert payload["dataset"] == "houses"
    models = {row["model"] for row in payload["results"]}
    assert {"HGBR", "TabPFN_full"} == models
