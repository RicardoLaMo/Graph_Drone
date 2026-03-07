"""Package init: expose step modules as s00–s07 for run_all.py."""

from experiments.california_curvature import (
    s00_data as s00,
    s01_graphs as s01,
    s02_curvature as s02,
    s03_baselines as s03,
    s04_graphsage as s04,
    s05_multiview as s05,
    s06_curvature_models as s06,
    s07_analysis as s07,
)

__all__ = ["s00", "s01", "s02", "s03", "s04", "s05", "s06", "s07"]
