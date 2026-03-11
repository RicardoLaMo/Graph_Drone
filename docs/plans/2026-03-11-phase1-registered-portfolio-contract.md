# Phase 1 Registered Portfolio Contract

## Purpose

Phase 1 measures GraphDrone across the registered dataset family before any per-dataset tuning or new architecture.

This phase answers:

- where GraphDrone wins
- where it loses
- whether those outcomes align with dataset properties and Phase 0 mechanism evidence

## Portfolio Rule

Use the registered dataset set as the primary benchmark surface.

California is mandatory in this phase and must be treated as a first-class dataset in the portfolio program, not a trailing appendix.

Source of truth:

- [graphdrone_registered_dataset_matrix.json](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-row-integration-plan/experiments/openml_regression_benchmark/configs/graphdrone_registered_dataset_matrix.json)

## Fixed-Setting Rule

Phase 1 uses:

- fixed hyperparameters
- fixed routing configuration
- fixed benchmark flow
- no per-dataset tuning

If a dataset needs special handling, that must be documented as a benchmark plumbing constraint, not hidden inside tuning.

## Required Outputs

1. Portfolio leaderboard
2. Per-dataset adaptive-minus-fixed deltas
3. Dataset property table
4. Win/loss clustering note
5. One summary answering whether GraphDrone behaves like a general row-level system or a regime-specific one

## Required Dataset Properties

For each dataset collect:

- row count
- feature count
- numeric vs categorical count
- target skew
- train/validation/test size
- ensemble disagreement summary
- routing entropy summary
- adaptive-minus-fixed delta

## Gates

### Gate P1.1: Portfolio Completion

Pass only if every dataset in the matrix has:

- a completed run
- the same model family comparisons
- summary artifacts written to the same surface

### Gate P1.2: California Inclusion

Pass only if California appears in the same portfolio table as the rest of the registered datasets.

### Gate P1.3: Non-Drift

Pass only if:

- no per-dataset tuning was introduced
- no hidden model variants were added mid-phase
- all deviations are documented explicitly

### Gate P1.4: Pattern Discovery

Pass only if the phase closes with one of:

- a real dataset-property hypothesis for when GraphDrone helps
- a clear conclusion that wins are scattered and the mechanism does not generalize cleanly

## Stop/Go Decision

Phase 2 may start only if Phase 1 ends with one of these explicit decisions:

- `GO_ROW_LEVEL_INTEGRATION`
  The mechanism appears real and portfolio-relevant.

- `GO_REGIME_SCOPED`
  The mechanism appears real but only for a subset of datasets.

- `NO_GO_MORE_ARCHITECTURE`
  The portfolio does not justify another integrated model branch yet.
