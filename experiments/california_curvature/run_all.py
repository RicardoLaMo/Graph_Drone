"""
run_all.py — Single entrypoint. Run CD from project root:
  python experiments/california_curvature/run_all.py
"""

import time
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.california_curvature import (
    s00, s01, s02, s03, s04, s05, s06, s07,
)


def main():
    print("=" * 60)
    print("CALIFORNIA HOUSING: CURVATURE-AWARENESS EXPERIMENT")
    print("=" * 60)
    t0 = time.time()

    print("\n>>> STEP 00: Data Loading & Preprocessing")
    s00.load_and_preprocess()

    print("\n>>> STEP 01: Graph Construction")
    s01.build_graphs()

    print("\n>>> STEP 02: Curvature & Observer Features")
    s02.compute_all()

    print("\n>>> STEP 03: Baselines (Mean, MLP, HGBR)")
    baseline_results = s03.run_baselines()

    print("\n>>> STEP 04: GraphSAGE per View (M3, M4, M5)")
    sage_results = s04.run_graphsage()

    print("\n>>> STEP 05: Multi-View Combiners (M6, M7)")
    mv_results = s05.run_multiview()

    print("\n>>> STEP 06: Curvature Models (M8A, M8B, M9)")
    curv_results = s06.run_curvature_models()

    print("\n>>> STEP 07: Analysis, Figures & Report")
    s07.run_analysis()

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE in {total:.1f}s")
    print(f"  Report  : reports/california_curvature_experiment.md")
    print(f"  Metrics : artifacts/california_metrics.csv")
    print(f"  Figures : figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
