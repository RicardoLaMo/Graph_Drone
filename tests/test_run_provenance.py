from __future__ import annotations

import json
from pathlib import Path

from graphdrone_fit import run_provenance


def test_run_ledger_records_pid_and_events(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(run_provenance, "gpu_compute_apps_snapshot", lambda: [{"pid": 42, "used_gpu_memory_mb": 512.0}])
    monkeypatch.setattr(run_provenance, "git_sha", lambda root: "abc123")

    run_dir = tmp_path / "run"
    ledger_path = run_provenance.start_run_ledger(
        run_dir=run_dir,
        root=tmp_path,
        label="challenger",
        task="regression",
        cmd=["python", "scripts/run_geopoe_benchmark.py"],
        env={
            "CUDA_VISIBLE_DEVICES": "7",
            "GRAPHDRONE_PRESET": "afc_candidate",
            "GRAPHDRONE_VERSION_OVERRIDE": "test",
            "PYTHONPATH": "src",
        },
        child_pid=12345,
    )
    run_provenance.append_run_event(run_dir=run_dir, child_pid=12345, event="started")
    run_provenance.finalize_run_ledger(ledger_path=ledger_path, elapsed_seconds=12.5, returncode=0)

    ledger = json.loads(ledger_path.read_text())
    assert ledger["child_pid"] == 12345
    assert ledger["git_sha"] == "abc123"
    assert ledger["returncode"] == 0
    assert ledger["final_gpu_apps"][0]["pid"] == 42

    events = (run_dir / "run_events.jsonl").read_text().strip().splitlines()
    assert len(events) == 1
    event = json.loads(events[0])
    assert event["child_pid"] == 12345
    assert event["event"] == "started"
