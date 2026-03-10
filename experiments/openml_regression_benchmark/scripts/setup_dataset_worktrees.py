from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.src.dataset_manifests import (  # noqa: E402
    available_manifest_datasets,
    load_manifest_for_dataset,
)


def common_worktrees_root(repo_root: Path) -> Path:
    if repo_root.parent.name == ".worktrees":
        return repo_root.parent
    return repo_root / ".worktrees"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create independent GraphDrone dataset worktrees")
    parser.add_argument("--datasets", nargs="+", default=available_manifest_datasets())
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--root", type=Path, default=common_worktrees_root(REPO_ROOT))
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def branch_exists(branch_name: str) -> bool:
    return bool(git_output("branch", "--list", branch_name))


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        manifest = load_manifest_for_dataset(dataset)
        worktree_path = (args.root / manifest.worktree_name).resolve()
        if worktree_path.exists():
            print(f"exists {manifest.dataset}: {worktree_path}")
            continue
        command = ["git", "-C", str(REPO_ROOT), "worktree", "add"]
        if branch_exists(manifest.branch_name):
            command.extend([str(worktree_path), manifest.branch_name])
        else:
            command.extend(["-b", manifest.branch_name, str(worktree_path), args.base_ref])
        subprocess.run(command, check=True)
        print(f"created {manifest.dataset}: {worktree_path} ({manifest.branch_name})")


if __name__ == "__main__":
    main()
