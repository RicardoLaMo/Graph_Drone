#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def skills_root() -> Path:
    return repo_root() / "skills"


def claude_skills_root() -> Path:
    return repo_root() / ".claude" / "skills"


def skill_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "SKILL.md").exists())


def parse_frontmatter(skill_md: Path) -> dict[str, str]:
    content = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {"name": skill_md.parent.name, "description": ""}
    data = yaml.safe_load(match.group(1)) or {}
    return {
        "name": str(data.get("name", skill_md.parent.name)),
        "description": str(data.get("description", "")).strip(),
    }


def render_index(skill_paths: list[Path]) -> str:
    lines = [
        "# Claude Skill Bridge",
        "",
        "These skill packs are generated from the repo `skills/` directory.",
        "Treat `skills/` as the source of truth and `.claude/skills/` as the Claude-facing projection.",
        "",
        "## How To Use",
        "",
        "1. Read the relevant repo context in `CLAUDE.md`.",
        "2. Open this index.",
        "3. Pick the most relevant skill directory below and read its `SKILL.md`.",
        "4. Read any referenced files only when needed.",
        "",
        "## Available Skills",
        "",
    ]
    for skill_dir in skill_paths:
        meta = parse_frontmatter(skill_dir / "SKILL.md")
        lines.append(f"- `{meta['name']}`")
        lines.append(f"  path: `.claude/skills/{skill_dir.name}/SKILL.md`")
        if meta["description"]:
            lines.append(f"  description: {meta['description']}")
    lines.append("")
    return "\n".join(lines)


def export_skills(dest_root: Path) -> None:
    src_root = skills_root()
    dirs = skill_dirs(src_root)
    if not dirs:
        raise SystemExit(f"No skill directories found under {src_root}")

    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    for src in dirs:
        shutil.copytree(src, dest_root / src.name)

    (dest_root / "INDEX.md").write_text(render_index(dirs), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export repo skills into a Claude-friendly .claude/skills tree.")
    parser.add_argument(
        "--dest-root",
        default=str(claude_skills_root()),
        help="Destination for the Claude-facing skill tree. Defaults to .claude/skills under the repo root.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    export_skills(Path(args.dest_root).expanduser().resolve())
    print(Path(args.dest_root).expanduser().resolve())


if __name__ == "__main__":
    main()
