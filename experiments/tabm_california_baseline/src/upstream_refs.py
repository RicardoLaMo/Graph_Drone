from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def extract_upstream_california_refs(upstream_root: Path) -> pd.DataFrame:
    rows = []
    for report_path in sorted(upstream_root.glob("exp/tabm/california/**/report.json")):
        obj = json.loads(report_path.read_text())
        metrics = obj.get("metrics", {}).get("test")
        if metrics is None:
            continue
        config_name = str(report_path.relative_to(upstream_root / "exp" / "tabm" / "california").parent)
        rows.append(
            {
                "family": "tabm",
                "dataset": "california",
                "config_name": config_name,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )
    return pd.DataFrame(rows)
