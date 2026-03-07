"""
run_experiment.py — Full MNIST geometry sanity check pipeline.

Usage:
  python experiments/mnist_geometry_sanity/scripts/run_experiment.py
  python experiments/mnist_geometry_sanity/scripts/run_experiment.py --full
"""

import sys
import time
import argparse
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.mnist_geometry_sanity.src import (
    s00_data, s01_graphs, s02_curvature, s03_baselines,
    s04_graphsage, s05_multiview, s06_curvature_models, s07_analysis,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use full 70k MNIST dataset")
    parser.add_argument("--n", type=int, default=10000, help="Subset size (ignored if --full)")
    args = parser.parse_args()
    n_subset = None if args.full else args.n

    print("=" * 65)
    print("  MNIST-784: GEOMETRY SANITY CHECK EXPERIMENT")
    print("  Branch: feature/mnist-geometry-sanity-check")
    print("=" * 65)
    t0 = time.time()

    print("\n>>> STEP 00: Data loading & preprocessing")
    s00_data.load_and_preprocess(n_subset=n_subset)

    print("\n>>> STEP 01: Graph construction (FULL, BLOCK, PCA)")
    s01_graphs.build_graphs()

    print("\n>>> STEP 02: Curvature & observer features")
    s02_curvature.compute_all()

    print("\n>>> STEP 03: Baselines (M0-M4: Majority, MLP, HGBR, XGBoost, TabPFN)")
    s03_baselines.run_baselines()

    print("\n>>> STEP 04: GraphSAGE per view (M5, M6, M7)")
    s04_graphsage.run_graphsage()

    print("\n>>> STEP 05: Multi-view combiners (M8 Uniform, M9 Learned)")
    s05_multiview.run_multiview()

    print("\n>>> STEP 06: Curvature models (M10 kappa feature, M11 observer combiner)")
    s06_curvature_models.run_curvature_models()

    print("\n>>> STEP 07: Analysis, figures & report")
    s07_analysis.run_analysis()

    total = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT COMPLETE in {total:.1f}s")
    print(f"  Report  : experiments/mnist_geometry_sanity/reports/mnist_geometry_sanity_report.md")
    print(f"  Metrics : experiments/mnist_geometry_sanity/artifacts/metrics.csv")
    print(f"  Figures : experiments/mnist_geometry_sanity/figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()
