# M1 Smoke Report

- Status: `PASS`
- Objective: run a fixed shipped TabR California config end-to-end on Apple Silicon using CPU FAISS, before any tuning or architecture swaps
- Command:
  - `source .venv-tabr312/bin/activate && python experiments/tabr_california_baseline/scripts/run_tabr_california.py --config 0-evaluation/0 --smoke --output-root experiments/tabr_california_baseline`

## Result

- Smoke config: `0-evaluation/0`
- Best epoch: `2`
- Train RMSE: `0.4579`
- Val RMSE: `0.4664`
- Test RMSE: `0.4409`
- Wrapper wall clock: `79.188s`
- Upstream reported run time: `31.026681s`

## Interpretation

- The pipeline now runs to completion through the wrapper on this machine.
- This is a smoke run on a locally exported California split, not a fair comparison to the shipped upstream benchmark numbers yet.
- Even so, `0.4409` test RMSE is already competitive with this repo's current `B1_HGBR` reference band and materially better than several recent GoRA/route-heavy California results.

## Blockers Cleared

- Replaced ad hoc shell state with explicit wrapper runtime guards:
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `KMP_DUPLICATE_LIB_OK=TRUE`
  - `PYTHONFAULTHANDLER=1`
  - `TQDM_DISABLE=1`
- Wrapper now records:
  - local config path
  - run output path
  - upstream report path
  - parsed summary
  - parsed metrics

## Remaining Gap Before M2

- This smoke run uses a local California export, not the shipped upstream benchmark data bundle.
- M2 still needs one fixed-config non-smoke reproduction and an explicit comparison against the upstream reference band.

## Artifacts

- Wrapper report: [0-evaluation__0__smoke.json](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/reports/0-evaluation__0__smoke.json)
- Upstream run summary: [summary.json](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/configs/0-evaluation__0__smoke/summary.json)
- Metrics table: [metrics.csv](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/artifacts/metrics.csv)
- Runtime table: [runtime.csv](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/artifacts/runtime.csv)
