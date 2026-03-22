from __future__ import annotations

from pathlib import Path
import sys

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKTREE_ROOT / "src"
for path in (SRC_ROOT, WORKTREE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
