#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_root() -> Path:
    return repo_root() / "skills"


def skill_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "SKILL.md").exists()
    )


def sync_one(src: Path, dest_root: Path) -> Path:
    dest = dest_root / src.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync GraphDrone repo skills into ~/.codex/skills.")
    parser.add_argument(
        "--dest-root",
        default=str(Path.home() / ".codex" / "skills"),
        help="Destination skill root. Defaults to ~/.codex/skills",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print source skill directories without copying them.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    src_root = source_root()
    repo_skill_dirs = skill_dirs(src_root)
    if not repo_skill_dirs:
        raise SystemExit(f"No repo skill directories found under {src_root}")

    if args.list_only:
        for path in repo_skill_dirs:
            print(path)
        return

    dest_root = Path(args.dest_root).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    for src in repo_skill_dirs:
        dest = sync_one(src, dest_root)
        print(f"synced {src.name} -> {dest}")


if __name__ == "__main__":
    main()
