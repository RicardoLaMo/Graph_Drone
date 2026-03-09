from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


KNOWN_FAMILIES = [
    "gora_tabular",
    "head_routing_v5",
    "mq_gora_v4",
    "mv_tabr_gora",
]


def parse_worktree_porcelain(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, value = line.split(" ", 1)
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["head"] = value
        elif key == "branch":
            prefix = "refs/heads/"
            current["branch"] = value[len(prefix):] if value.startswith(prefix) else value
    if current:
        records.append(current)
    return records


def detect_experiment_families(experiment_dirs: list[str]) -> list[str]:
    found = {Path(p).name for p in experiment_dirs}
    return [name for name in KNOWN_FAMILIES if name in found]


def to_repo_relative_path(repo_root: str, path: str) -> str:
    try:
        relative = Path(path).resolve().relative_to(Path(repo_root).resolve())
    except ValueError:
        return path
    return "." if str(relative) == "." else str(relative)


def classify_lineage_role(path: str, branch: str, families: list[str], current_path: str) -> str:
    if path == current_path:
        return "current-checkout"
    if "funny-davinci" in path or branch == "claude/funny-davinci":
        return "early-v3-snapshot"
    if "mv_tabr_gora" in families and "a7" in path:
        return "mv-tabr-followon"
    if "mv_tabr_gora" in families:
        return "mv-tabr-core"
    if "head_routing_v5" in families:
        return "tracked-v5-family"
    if "mq_gora_v4" in families:
        return "tracked-v4-family"
    if "gora_tabular" in families:
        return "gora-tabular-line"
    return "other"


def infer_primary_focus(branch: str, families: list[str]) -> str:
    if branch == "feature/gora-v5-trust-routing" or "head_routing_v5" in families:
        return "head_routing_v5"
    if "mv-tabr-gora-a7" in branch:
        return "mv_tabr_gora_a7_followon"
    if branch == "feature/mv-tabr-gora" or branch == "feature/mv-tabr-gora-rerank" or "mv_tabr_gora" in families:
        return "mv_tabr_gora_core"
    if "california-v35-routed-regression" in branch:
        return "california_v35_routed_regression"
    if "california-tab-foundation-compare" in branch:
        return "california_tab_foundation_compare"
    if "tabr-california-baseline" in branch:
        return "tabr_california_baseline"
    if "tabr-mv-retrieval-bc" in branch:
        return "tabr_mv_retrieval_bc"
    if "shared-head-routing-backbone" in branch:
        return "shared_head_routing_backbone"
    if branch == "claude/funny-davinci":
        return "gora_tabular_v3_snapshot"
    if "gora_tabular" in families:
        return "gora_tabular"
    if "mq_gora_v4" in families:
        return "mq_gora_v4"
    return "other"


def classify_purpose(branch: str, families: list[str], path: str, current_path: str) -> str:
    if path == current_path:
        return "primary working checkout"
    if "mv_tabr_gora" in families and "a7" in path:
        return "mv-tabr retrieval follow-on experiment"
    if "mv_tabr_gora" in families:
        return "mv-tabr california core experiment"
    if "head_routing_v5" in families:
        return "tracked v5 trust-routing experiment family"
    if "mq_gora_v4" in families:
        return "tracked v4 split-track experiment family"
    if branch == "claude/funny-davinci":
        return "Claude Code v3 snapshot worktree"
    return "auxiliary worktree"


def build_registry(
    *,
    repo_root: str,
    current_path: str,
    current_branch: str,
    current_head: str,
    worktrees: list[dict[str, str]],
    families_by_path: dict[str, list[str]],
) -> dict:
    entries = []
    for item in worktrees:
        path = item["path"]
        branch = item.get("branch", "(detached)")
        families = families_by_path.get(path, [])
        entries.append(
            {
                "path": path,
                "repo_relative_path": to_repo_relative_path(repo_root, path),
                "branch": branch,
                "head": item.get("head", ""),
                "is_current_checkout": path == current_path,
                "experiment_families": families,
                "primary_focus": infer_primary_focus(branch, families),
                "lineage_role": classify_lineage_role(path, branch, families, current_path),
                "purpose": classify_purpose(branch, families, path, current_path),
                "status": "current" if path == current_path else "sibling-worktree",
            }
        )
    return {
        "repo_root": repo_root,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "current_checkout": {
            "path": current_path,
            "repo_relative_path": to_repo_relative_path(repo_root, current_path),
            "branch": current_branch,
            "head": current_head,
        },
        "worktrees": entries,
    }


def run_git(*args: str, cwd: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def list_experiment_dirs(worktree_path: str) -> list[str]:
    base = Path(worktree_path) / "experiments"
    if not base.exists():
        return []
    return [str(path.relative_to(Path(worktree_path))) for path in base.iterdir() if path.is_dir()]


def generate_registry(repo_root: str) -> dict:
    porcelain = run_git("worktree", "list", "--porcelain", cwd=repo_root)
    worktrees = parse_worktree_porcelain(porcelain)
    current_path = run_git("rev-parse", "--show-toplevel", cwd=repo_root)
    current_branch = run_git("branch", "--show-current", cwd=repo_root)
    current_head = run_git("rev-parse", "--short", "HEAD", cwd=repo_root)
    families_by_path = {
        item["path"]: detect_experiment_families(list_experiment_dirs(item["path"]))
        for item in worktrees
    }
    return build_registry(
        repo_root=repo_root,
        current_path=current_path,
        current_branch=current_branch,
        current_head=current_head,
        worktrees=worktrees,
        families_by_path=families_by_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit git worktrees and experiment lineage.")
    parser.add_argument(
        "--write",
        type=Path,
        help="Write the generated registry JSON to this path.",
    )
    args = parser.parse_args()

    repo_root = run_git("rev-parse", "--show-toplevel", cwd=str(Path.cwd()))
    registry = generate_registry(repo_root)

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(registry, indent=2) + "\n")

    print(f"repo_root: {registry['repo_root']}")
    print(
        "current_checkout:",
        registry["current_checkout"]["branch"],
        registry["current_checkout"]["head"],
        registry["current_checkout"]["path"],
    )
    for item in registry["worktrees"]:
        families = ",".join(item["experiment_families"]) or "-"
        print(
            f"- {item['branch']} | {item['head']} | {item['lineage_role']} | "
            f"{families} | {item['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
