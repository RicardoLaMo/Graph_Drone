from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tabm_california_baseline.src.upstream_refs import extract_upstream_california_refs
from experiments.tab_foundation_compare.src.runtime_support import default_tabm_upstream_root


def main() -> None:
    upstream_root = default_tabm_upstream_root(REPO_ROOT)
    df = extract_upstream_california_refs(upstream_root)
    output = REPO_ROOT / "experiments" / "tabm_california_baseline" / "artifacts" / "upstream_reference_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
