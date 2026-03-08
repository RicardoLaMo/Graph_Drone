# M2 Fixed-Config Report

- Status: `PASS`
- Objective: reproduce one shipped TabR California evaluation config without tuning and compare it against the upstream reference band
- Command:
  - `source .venv-tabr312/bin/activate && python experiments/tabr_california_baseline/scripts/run_tabr_california.py --config 0-evaluation/0 --output-root experiments/tabr_california_baseline`

## Result

- Config: `0-evaluation/0`
- Device: `cpu`
- Best epoch: `49`
- Train RMSE: `0.3645`
- Val RMSE: `0.4009`
- Test RMSE: `0.3949`
- Wrapper wall clock: `747.901s`
- Upstream reported run time: `0:11:44.988110`

## Comparison Against Upstream References

- Shipped upstream `0-evaluation/0` test RMSE: `0.4071`
- Shipped upstream `default-evaluation/0` test RMSE: `0.4106`
- Shipped upstream `2-plr-lite-evaluation/0` test RMSE: `0.4003`
- Local fixed-config `0-evaluation/0` test RMSE: `0.3949`

## Interpretation

- The Apple Silicon CPU reproduction is now in, and slightly better than, the upstream California reference band.
- This does not mean the local run is a directly fair apples-to-apples leaderboard claim, because the current path still uses a local California export rather than the full shipped benchmark bundle.
- It does mean the execution path is trustworthy enough to establish TabR as the current champion candidate for California in this repo.

## Milestone Conclusion

- `M0`: PASS
- `M1`: PASS
- `M2`: PASS

Current champion candidate:
- `TabR_CPU_fixed` on `0-evaluation/0`
- Test RMSE `0.3949`

## Next Step

- `M3` is optional.
- The more useful next move is `M4/M5`: lock this CPU champion and then align splits/preprocessing before challenger swaps with our architecture.

## Artifacts

- Wrapper report: [0-evaluation__0.json](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/reports/0-evaluation__0.json)
- Upstream summary: [summary.json](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/configs/0-evaluation__0/summary.json)
- Metrics table: [metrics.csv](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/artifacts/metrics.csv)
- Runtime table: [runtime.csv](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/artifacts/runtime.csv)
