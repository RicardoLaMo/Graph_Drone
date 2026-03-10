from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def extract_upstream_california_refs(upstream_root: Path) -> pd.DataFrame:
    base = upstream_root / "exp" / "tabr" / "california"
    rows = []
    for report_path in sorted(base.glob("**/report.json")):
        config_dir = report_path.parent.relative_to(base)
        try:
            metrics = json.loads(report_path.read_text())["metrics"]["test"]
        except Exception:
            continue
        rows.append(
            {
                "family": "tabr",
                "dataset": "california",
                "config_name": str(config_dir),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )
    return pd.DataFrame(rows)

