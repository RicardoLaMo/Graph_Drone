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

## First implemented change

The first usefulness-routing change adds a direct allocation-focused regression objective.

New config/control:
- `allocation_usefulness_lambda`
- env: `GRAPHDRONE_ALLOCATION_USEFULNESS_LAMBDA`

Current formulation:
- reward positive `validation_weighted_specialist_advantage_score`
- reward positive `validation_positive_specialist_mass`
- combined training-side allocation score:
  - `validation_allocation_usefulness_score`

This is intentionally different from the already falsified residual-gap penalty:
- the old formulation mainly punished missed opportunity after a poor allocation
- the new formulation tries to reward selective mass on helpful specialists directly

## First quick result

Quick contract:
- `eval/v13_reg_usefulness_quick_v2/report/results_granular.csv`
- `eval/v13_reg_usefulness_quick_v2/report/regression_route_state_summary.csv`

Setting:
- `GRAPHDRONE_ALLOCATION_USEFULNESS_LAMBDA=0.1`

Observed behavior:
- `california`
  - stayed `clean_routed`
  - weighted specialist advantage improved slightly
  - positive specialist mass improved slightly
  - defer dropped from the prior quick baseline
  - RMSE got slightly worse
- `cpu_act`
  - stayed `legitimacy_gate_anchor_only` at prediction time
  - but now preserves training-side usefulness diagnostics in the artifact row
  - weighted specialist advantage and positive specialist mass were both strong on the router-training split

Current interpretation:
- the direct allocation reward is mechanistically alive
- the new usefulness diagnostics are now visible even on early-exit rows
- end-to-end translation is still mixed, so the lane remains open

## Acceptance

This lane succeeds when realized specialist value improves materially before or alongside RMSE/R² improvement.
