# v1.3 Regression Stability Lane

Branch:
- `exp/v13-reg-stability`

Base:
- `exp/v13-regression-program` at `f58b116`

Worktree:
- `/home/wliu23/projects/GraphDrone2/.worktrees/v13-reg-stability`

## Mission

Make regression router failures and fallbacks fully diagnosable.

## Starting evidence

Read first:
- `docs/2026-03-23-afc-phase-b-california-router-instability.md`
- `docs/2026-03-23-afc-phase-b-residual-usefulness.md`
- `docs/2026-03-23-v13-regression-handoff.md`

Relevant current signals:
- `router_training_nonfinite_anchor_only`
- `router_nonfinite_fallback`
- missing or `NaN` regression mechanism diagnostics on `california`

## First implementation targets

1. Classify regression fallback reasons explicitly.
2. Ensure every fallback path emits enough diagnostics to explain:
   - why training failed
   - whether tokens were finite
   - whether routing was skipped, forced anchor-only, or partially evaluated
3. Remove “finite predictions, unclear mechanism state” as an outcome class.

## First benchmark contract

Use:
- quick regression gate first
- then fold-0 mini-full regression

## Acceptance

This lane succeeds when a reviewer can explain every regression fallback case from the artifacts alone.
