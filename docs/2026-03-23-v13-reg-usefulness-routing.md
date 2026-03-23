# v1.3 Regression Usefulness-Routing Lane

Branch:
- `exp/v13-reg-usefulness-routing`

Base:
- `exp/v13-regression-program` at `f58b116`

Worktree:
- `/home/wliu23/projects/GraphDrone2/.worktrees/v13-reg-usefulness-routing`

## Mission

Turn positive available specialist value into positive realized specialist value.

## Starting evidence

Read first:
- `docs/2026-03-23-afc-phase-b-residual-usefulness.md`
- `docs/2026-03-23-afc-phase-b-anchor-exclusion.md`
- `docs/2026-03-23-v13-regression-handoff.md`

Current read:
- useful specialists exist
- realized attention-weighted specialist value is often negative
- the earlier residual-gap penalty formulation is already falsified

## First implementation targets

1. Add allocation-focused regression objectives rather than only final blended loss pressure.
2. Treat these as first-class diagnostics in every run:
   - `validation_best_specialist_advantage_score`
   - `validation_weighted_specialist_advantage_score`
   - `validation_defer_weighted_specialist_advantage_score`
   - `validation_positive_specialist_mass`
   - `validation_top_specialist_positive_rate`
   - `validation_residual_usefulness_gap`
3. Evaluate whether improved allocation quality translates to held-out RMSE/R².

## First benchmark contract

Use:
- quick regression gate first
- then fold-0 mini-full regression

## Acceptance

This lane succeeds when realized specialist value improves materially before or alongside RMSE/R² improvement.
