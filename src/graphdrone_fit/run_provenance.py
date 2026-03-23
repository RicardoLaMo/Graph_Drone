from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None


def git_sha(root: Path) -> str | None:
    result = _safe_run(["git", "rev-parse", "HEAD"], cwd=root)
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip() or None


def gpu_compute_apps_snapshot() -> list[dict[str, Any]]:
    result = _safe_run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "pid": int(parts[0]) if parts[0].isdigit() else parts[0],
                "process_name": parts[1],
                "gpu_uuid": parts[2],
                "used_gpu_memory_mb": float(parts[3]) if parts[3].replace(".", "", 1).isdigit() else parts[3],
            }
        )
    return rows


def start_run_ledger(
    *,
    run_dir: Path,
    root: Path,
    label: str,
    task: str,
    cmd: list[str],
    env: dict[str, str],
    child_pid: int,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "run_ledger.json"
    ledger = {
        "label": label,
        "task": task,
        "started_at_utc": utc_now_iso(),
        "root": str(root),
        "hostname": socket.gethostname(),
        "launcher_pid": os.getpid(),
        "child_pid": child_pid,
        "git_sha": git_sha(root),
        "command": cmd,
        "env": {
            key: env.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "GRAPHDRONE_PRESET",
                "GRAPHDRONE_VERSION_OVERRIDE",
                "GRAPHDRONE_ROUTER_SEED",
                "PYTHONPATH",
            ]
        },
        "initial_gpu_apps": gpu_compute_apps_snapshot(),
    }
    ledger_path.write_text(json.dumps(ledger, indent=2))
    return ledger_path


def append_run_event(
    *,
    run_dir: Path,
    child_pid: int,
    event: str,
    returncode: int | None = None,
) -> None:
    event_path = run_dir / "run_events.jsonl"
    payload = {
        "timestamp_utc": utc_now_iso(),
        "event": event,
        "child_pid": child_pid,
        "returncode": returncode,
        "gpu_apps": gpu_compute_apps_snapshot(),
    }
    with event_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def finalize_run_ledger(
    *,
    ledger_path: Path,
    elapsed_seconds: float,
    returncode: int,
) -> None:
    ledger = json.loads(ledger_path.read_text())
    ledger.update(
        {
            "finished_at_utc": utc_now_iso(),
            "elapsed_seconds": elapsed_seconds,
            "returncode": returncode,
            "final_gpu_apps": gpu_compute_apps_snapshot(),
        }
    )
    ledger_path.write_text(json.dumps(ledger, indent=2))
