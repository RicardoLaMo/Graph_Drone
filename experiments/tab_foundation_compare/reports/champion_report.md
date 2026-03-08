# M4 Champion Lock

## Decision

`TabR_CPU_fixed_upstream_style` remains the California champion candidate.

- Branch: `feature/tabr-california-baseline`
- Fixed config: `0-evaluation/0`
- Test RMSE: `0.3949`
- Val RMSE: `0.4009`
- Best epoch: `49`

## Why This Was Locked

- It was locally reproduced end-to-end on this Apple Silicon environment.
- It materially outperformed all prior repo California references.
- It did not require CUDA to be credible on this machine.

## TabM Status At M4

TabM was added as the second foundation baseline, but its M4 anchor is the upstream shipped California seed-0 reference rather than a separate local upstream-style rerun.

- Upstream seed-0 reference RMSE: `0.4418`

That is strong enough to justify inclusion, but not strong enough to displace TabR as champion.

## Immediate Consequence

All fair challenger work should be judged against:

1. `TabR_CPU_fixed_upstream_style` as the locked champion
2. `TabR_on_our_split` and `TabM_on_our_split` after alignment
