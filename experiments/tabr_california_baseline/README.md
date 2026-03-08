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

Latest smoke result:
- config: `0-evaluation/0`
- device: `cpu`
- test RMSE: `0.44085553303605535`
