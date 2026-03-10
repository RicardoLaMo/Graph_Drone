# OpenML Regression Benchmark

This benchmark runs the current GraphDrone family against `TabPFN`, `TabR`, and
`TabM` on the shared H200 environment.

## Canonical Runtime

Use the shared H200 environment:

```bash
/home/wliu23/projects/Graph_Drone/.venv-h200
```

Recommended interpreter pattern:

```bash
/home/wliu23/projects/Graph_Drone/.venv-h200/bin/python ...
```

## Main Entry Points

- Full multi-model suite scheduler:
  `experiments/openml_regression_benchmark/scripts/run_openml_suite.py`
- Per-dataset manifest runner:
  `experiments/openml_regression_benchmark/scripts/run_dataset_manifest.py`
- Lane launchers:
  - `scripts/run_lane_a_small_datasets.sh`
  - `scripts/run_lane_b_diamonds.sh`
  - `scripts/run_portfolio_lanes.sh`
- GPU poller:
  `scripts/monitor_portfolio_lanes.sh`

## Standard Flow

1. Check GPUs:

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

2. Run the portfolio lanes:

```bash
cd /home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression
experiments/openml_regression_benchmark/scripts/run_portfolio_lanes.sh full
```

3. Run a standalone dataset when it is intentionally outside the active lanes:

```bash
cd /home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression
/home/wliu23/projects/Graph_Drone/.venv-h200/bin/python \
  experiments/openml_regression_benchmark/scripts/run_dataset_manifest.py \
  --dataset wine_quality --gpus 7
```

## Current Summary Artifacts

- Portfolio table:
  `experiments/openml_regression_benchmark/openml_portfolio_leaderboard.md`
- California-focused table:
  `experiments/openml_regression_benchmark/california_lineage_leaderboard.md`
- Earlier 3-fold benchmark summary:
  `experiments/openml_regression_benchmark/reports_h200_20260310_full/openml_benchmark_summary.md`
