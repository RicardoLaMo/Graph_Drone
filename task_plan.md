# Task Plan

## Goal

Establish a disciplined California comparison track with:
- `M4` champion lock for TabR
- TabM added as a second foundation baseline
- `M5` aligned protocol runs for fair comparison against this repo's California experiments
- `C1` aligned foundation comparison
- `C2` decoder challenger on top of the strongest aligned foundation

## Current Branch

- Branch: `feature/california-tab-foundation-compare`
- Worktree: `/Volumes/MacMini/Projects/Graph_Drone/.worktrees/california-tab-foundation-compare`
- Base: `feature/tabr-california-baseline` at `fb92ce0`

## Phases

1. `DONE` - Verify inherited TabR baseline branch state and test baseline.
2. `DONE` - Inspect TabM upstream repo, California configs, and published California metrics.
3. `IN_PROGRESS` - Write M4/M5/C1/C2 comparison plan and persistent findings.
4. `PENDING` - Implement `M4` champion lock report for TabR.
5. `PENDING` - Implement TabM California baseline runner and smoke/fixed runs.
6. `PENDING` - Implement `M5` alignment audit and aligned data protocol.
7. `PENDING` - Run `C1` aligned foundation comparison: TabR vs TabM vs repo references.
8. `PENDING` - Implement `C2` decoder challenger on the strongest aligned foundation.

## Open Decisions

- Use CPU as the locked execution device unless an MPS run clearly matches metrics without extra fragility.
- Treat TabM as a comparison baseline first, not as the immediate champion favorite.
- Keep `C2` focused on decoder/readout changes, not encoder rewrites.

## Risks

- TabM paper reproduction may want a separate environment and dataset bundle.
- Alignment can easily become confounded if we do not pin one California split/protocol.
- `C2` can sprawl if it mixes retrieval, routing, and decoder changes in one step.
