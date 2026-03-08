# TabR California Baseline

This experiment tracks a CPU-first TabR baseline on California Housing in an
isolated worktree before any architecture swaps.

Scripts:
- `scripts/check_env.py`
- `scripts/export_upstream_refs.py`
- `scripts/build_local_california.py`
- `scripts/run_tabr_california.py`

Current status:
- `M0` environment capture: complete
- `M1` CPU smoke run: complete
- `M2` fixed-config CPU run: complete

Latest fixed-config result:
- config: `0-evaluation/0`
- device: `cpu`
- best epoch: `49`
- test RMSE: `0.3949403615264023`
