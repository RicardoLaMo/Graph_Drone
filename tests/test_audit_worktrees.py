from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_worktrees.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_worktrees", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_worktree_porcelain_output():
    module = load_module()
    text = """worktree /repo
HEAD 1234567890abcdef
branch refs/heads/main

worktree /repo/.worktrees/exp
HEAD fedcba0987654321
branch refs/heads/feature/exp
"""
    records = module.parse_worktree_porcelain(text)
    assert len(records) == 2
    assert records[0]["path"] == "/repo"
    assert records[0]["head"] == "1234567890abcdef"
    assert records[0]["branch"] == "main"
    assert records[1]["branch"] == "feature/exp"


def test_detect_experiment_families():
    module = load_module()
    names = module.detect_experiment_families(
        [
            "experiments/gora_tabular",
            "experiments/head_routing_v5",
            "experiments/mq_gora_v4",
            "experiments/mv_tabr_gora",
        ]
    )
    assert names == [
        "gora_tabular",
        "head_routing_v5",
        "mq_gora_v4",
        "mv_tabr_gora",
    ]


def test_build_registry_marks_current_checkout():
    module = load_module()
    worktrees = [
        {"path": "/repo", "head": "111", "branch": "main"},
        {"path": "/repo/.worktrees/exp", "head": "222", "branch": "feature/exp"},
    ]
    families_by_path = {
        "/repo": ["gora_tabular", "head_routing_v5"],
        "/repo/.worktrees/exp": ["mv_tabr_gora"],
    }
    registry = module.build_registry(
        repo_root="/repo",
        current_path="/repo",
        current_branch="main",
        current_head="111",
        worktrees=worktrees,
        families_by_path=families_by_path,
    )
    assert registry["repo_root"] == "/repo"
    assert registry["current_checkout"]["path"] == "/repo"
    assert registry["current_checkout"]["repo_relative_path"] == "."
    assert registry["worktrees"][0]["is_current_checkout"] is True
    assert registry["worktrees"][0]["repo_relative_path"] == "."
    assert registry["worktrees"][0]["experiment_families"] == ["gora_tabular", "head_routing_v5"]
    assert registry["worktrees"][1]["repo_relative_path"] == ".worktrees/exp"
    assert registry["worktrees"][1]["experiment_families"] == ["mv_tabr_gora"]


def test_infer_primary_focus_from_branch_name():
    module = load_module()
    assert module.infer_primary_focus("feature/mv-tabr-gora-a7a-iterative-reindex", []) == "mv_tabr_gora_a7_followon"
    assert module.infer_primary_focus("feature/mv-tabr-gora", ["mv_tabr_gora"]) == "mv_tabr_gora_core"
    assert module.infer_primary_focus("feature/california-v35-routed-regression", []) == "california_v35_routed_regression"
    assert module.infer_primary_focus("feature/gora-v5-trust-routing", ["head_routing_v5"]) == "head_routing_v5"
