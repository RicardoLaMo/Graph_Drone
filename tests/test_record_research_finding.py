from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_research_finding.py"


def test_record_research_finding_rebuilds_current_map(tmp_path: Path):
    repo_root = tmp_path / "repo"
    docs_research = repo_root / "docs" / "research"
    docs_research.mkdir(parents=True)
    findings = docs_research / "findings.jsonl"
    findings.write_text("", encoding="utf-8")
    current_map = docs_research / "current_hypotheses.md"

    script_copy = repo_root / "scripts" / "record_research_finding.py"
    script_copy.parent.mkdir(parents=True)
    script_copy.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    cmd = [
        sys.executable,
        str(script_copy),
        "record",
        "--claim-id",
        "afc-b-reg-anchor-asymmetry",
        "--topic",
        "afc_phase_b",
        "--status",
        "partially_causal",
        "--question",
        "Was anchor contamination real?",
        "--conclusion",
        "Yes, partly causal.",
        "--summary",
        "Anchor exclusion improved the rotor branch.",
        "--branch",
        "exp/afc-b-reg-anchor-exclusion",
        "--commit",
        "0fbbe59",
        "--note-path",
        "docs/2026-03-23-afc-phase-b-anchor-exclusion.md",
        "--artifact-path",
        "eval/phaseb_reg_anchorfix_l001_mini/comparison/promotion_decision.json",
        "--next-check",
        "Test frozen-router rotor training.",
        "--tag",
        "afc",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    lines = [line for line in findings.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["claim_id"] == "afc-b-reg-anchor-asymmetry"
    assert record["status"] == "partially_causal"
    assert record["next_checks"] == ["Test frozen-router rotor training."]
    assert current_map.exists()
    content = current_map.read_text(encoding="utf-8")
    assert "Partially Causal" in content
    assert "afc-b-reg-anchor-asymmetry" in content
