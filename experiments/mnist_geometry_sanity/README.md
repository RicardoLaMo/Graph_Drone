# MNIST-784: Geometry Sanity Check Experiment

## Isolation Notice
> ⚠️ This experiment is **completely isolated** from the California Housing curvature experiment.
> No prior experiment files are modified. All outputs live exclusively under `experiments/mnist_geometry_sanity/`.

## Branch
`feature/mnist-geometry-sanity-check` (forked from `main`)

## Hypothesis
MNIST-784, when flattened into tabular form, retains hidden pixel adjacency geometry.
If our kNN + graph embedding + curvature-view pipeline cannot detect or exploit that hidden geometry,
that is a warning sign about the functionality of the mechanism.

This is **NOT** a claim that curvature-aware methods are better for image classification.
It is a **mechanism validation**: does our framework at least respond to known hidden geometry?

## Warning Signs
The following outcomes would weaken confidence in the mechanism:
- GraphSAGE views do not outperform MLP (H1 fails)
- Structured views are not complementary (H2 fails)
- Curvature proxy has low stability across scales (< 40% top-20% overlap)
- Curvature-aware model does not reduce error on high-kappa rows
- XGBoost or TabPFN dominate with no signal from graph methods

## Run Commands

```bash
# Activate environment
source .venv/bin/activate

# Run full experiment (default: 10k row subset)
python experiments/mnist_geometry_sanity/scripts/run_experiment.py

# Run analysis + report only (after experiment completes)
python experiments/mnist_geometry_sanity/scripts/run_report.py

# Run with full 70k dataset (slow, ~1-2 hrs)
python experiments/mnist_geometry_sanity/scripts/run_experiment.py --full
```

## Outputs
| Path | Contents |
|------|----------|
| `artifacts/metrics.csv` | All model metrics |
| `artifacts/curvature_bins.csv` | Per-row curvature + observer features |
| `figures/curvature_hist.png` | Curvature distribution |
| `figures/error_by_curvature_bin.png` | Accuracy by curvature bin per model |
| `figures/view_comparison.png` | Bar chart: per-view GraphSAGE vs baselines |
| `figures/stability_heatmap.png` | Curvature stability across k=10/20/30 |
| `reports/mnist_geometry_sanity_report.md` | Full experiment report |

## Models Compared
| ID | Description |
|----|-------------|
| M0 | Majority class baseline |
| M1 | MLP (3-layer, MPS) |
| M2 | HistGradientBoostingClassifier |
| M3 | XGBoost (multiclass, hist) |
| M4 | TabPFN v2.5 (subset ≤1024 train; documented limitation) |
| M5 | GraphSAGE on FULL graph |
| M6 | GraphSAGE on SPATIAL-BLOCK graph |
| M7 | GraphSAGE on PCA-50 graph |
| M8 | Uniform ensemble M5/M6/M7 |
| M9 | Learned combiner (no curvature) |
| M10 | kappa appended as node feature → GraphSAGE |
| M11 | Observer-driven per-row combiner [kappa, LID, LOF, density] |
