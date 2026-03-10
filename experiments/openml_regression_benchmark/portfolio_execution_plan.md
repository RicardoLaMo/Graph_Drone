# OpenML Portfolio Execution Plan

Primary operating path:
- use the shared benchmark base at [graphdrone-openml-regression](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression)
- keep per-dataset worktrees only for true code divergence, not for normal execution
- use the shared H200 runtime at `/home/wliu23/projects/Graph_Drone/.venv-h200`

Current 8-dataset portfolio:
- `houses`
- `miami_housing`
- `diamonds`
- `healthcare_insurance_expenses`
- `concrete_compressive_strength`
- `airfoil_self_noise`
- `wine_quality`
- `used_fiat_500`

Lane policy:
- Lane A, `1 GPU per job`: `airfoil_self_noise`, `concrete_compressive_strength`, `healthcare_insurance_expenses`, `used_fiat_500`
- Lane B, `4 GPUs`: `diamonds` with `--exclusive-graphdrone`
- `wine_quality`, `houses`, and `miami_housing` stay on the normal 1-GPU path unless a targeted GraphDrone-only rerun is justified
- `wine_quality` is the standard standalone follow-up after lane completion when it has not been folded into the active lane set

Default GPU partition:
- Lane A: `3,2,1,0`
- Lane B: `7,6,5,4`

Run commands:

```bash
cd /home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression
experiments/openml_regression_benchmark/scripts/run_lane_a_small_datasets.sh full
```

```bash
cd /home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression
experiments/openml_regression_benchmark/scripts/run_lane_b_diamonds.sh full
```

```bash
cd /home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression
experiments/openml_regression_benchmark/scripts/run_portfolio_lanes.sh full
```

Standalone follow-up dataset:

```bash
cd /home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression
/home/wliu23/projects/Graph_Drone/.venv-h200/bin/python \
  experiments/openml_regression_benchmark/scripts/run_dataset_manifest.py \
  --dataset wine_quality --gpus 7
```

Print-only dry run:

```bash
cd /home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression
experiments/openml_regression_benchmark/scripts/run_portfolio_lanes.sh smoke --print-only
```

15-minute background monitor:

```bash
cd /home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-openml-regression
experiments/openml_regression_benchmark/scripts/monitor_portfolio_lanes.sh full --interval-seconds 900
```

Operational rule:
- if fewer than 4 clean GPUs are available, do not launch Lane B
- never put a 4-GPU GraphDrone run into the same mixed FIFO queue as the 1-GPU lane
