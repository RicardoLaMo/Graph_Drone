from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "tabarena_bridge" / "scripts" / "score_p0_position.py"
OUT_DIR = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "p0_positioning_test"


def test_p0_positioning_generates_leaderboard() -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--output", str(OUT_DIR)], check=True, cwd=ROOT)
    leaderboard = json.loads((OUT_DIR / "leaderboard.json").read_text())
    assert leaderboard["target_model"] == "P0_router"
    models = [row["model"] for row in leaderboard["leaderboard"]]
    assert "P0_router" in models
    assert "TabPFN_full" in models
