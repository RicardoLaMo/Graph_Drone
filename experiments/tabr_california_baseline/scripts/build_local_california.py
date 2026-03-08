from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tabr_california_baseline.src.data_bridge import write_california_dataset


def main():
    output_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO_ROOT / "experiments" / "tabr_california_baseline" / "artifacts" / "data" / "california_local"
    )
    write_california_dataset(output_dir, seed=0)


if __name__ == "__main__":
    main()
