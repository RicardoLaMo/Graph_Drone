from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tabr_california_baseline.src.upstream_refs import extract_upstream_california_refs
from experiments.tab_foundation_compare.src.runtime_support import default_tabr_upstream_root


def main():
    upstream_root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_tabr_upstream_root(REPO_ROOT)
    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else REPO_ROOT / "experiments" / "tabr_california_baseline" / "artifacts" / "upstream_reference_metrics.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = extract_upstream_california_refs(upstream_root)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
