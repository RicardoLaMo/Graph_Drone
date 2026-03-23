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

## First implemented change

The first v1.3 stability change adds explicit regression fallback cause labels.

New diagnostics now distinguish:
- training-stage fallback source:
  - `regression_router_fallback_stage`
  - `regression_router_fallback_reason`
- training-surface finiteness:
  - `validation_router_tokens_finite_flag`
  - `validation_router_predictions_finite_flag`
  - `validation_router_targets_finite_flag`
  - `validation_anchor_mse_finite_flag`
- prediction-stage finiteness when router outputs fail:
  - `prediction_router_tokens_finite_flag`
  - `prediction_router_weights_finite_flag`
  - `prediction_router_defer_finite_flag`

Current covered fallback reasons:
- `no_trainable_params`
- `nonfinite_loss`
- `nonfinite_gradients`
- `nonfinite_parameters`
- `nonfinite_router_output`

This does not solve the underlying regression router failures yet.
It makes them artifact-explainable, which is the prerequisite for the rest of the lane.

## First validation result

The first quick regression validation confirmed a real implementation hazard and then cleared it.

Observed issue:
- the residual-usefulness pass in `_fit_regression_router()` overwrote `_router_fit_diagnostics`
- that erased the new stability flags from clean routed cases
- `california` therefore looked like a healthy routed run in logs but still showed `NaN` training-surface diagnostics in the artifacts

Fix:
- merge residual-usefulness diagnostics into the existing router-fit diagnostics instead of replacing them

Validated on:
- `eval/v13_reg_stability_quick_v3/report/results_granular.csv`
- `eval/v13_reg_stability_quick_v3/cache/california__fold0__graphdrone.json`

Result:
- `california` now preserves the new validation finite flags on a clean routed row
- the cache keeps them under `diagnostics`
- the granular CSV exposes them directly

This closes the first stability-surface bug:
- regression diagnostics are no longer being silently erased by a later mechanism pass

## Second implemented change

The benchmark runner now emits a regression-level fallback summary:
- `eval/.../report/regression_fallback_summary.csv`
- mirrored in the regression section of `report.txt`

This matters because it separates:
- true regression router fallback or instability states
- safe non-fallback paths such as legitimacy-gate early exit

Quick validation on `california` + `cpu_act` showed:
- `california`: routed regression case with preserved training-side diagnostics
- `cpu_act`: legitimacy-gate anchor-only path, not a regression router fallback

So the lane now has the first dataset-level surface needed to distinguish:
- routing instability
- deliberate early exit
- clean routed behavior

## Third implemented change

The regression report now adds an explicit route-state summary:
- `eval/.../report/regression_route_state_summary.csv`

This is stronger than fallback-only reporting because it classifies every regression row into:
- `clean_routed`
- `legitimacy_gate_early_exit`
- `router_fallback`

Quick validation on `california` + `cpu_act` confirmed the intended separation:
- `california -> clean_routed`
- `cpu_act -> legitimacy_gate_early_exit`

That means the stability lane no longer has to infer route state indirectly from:
- `router_kind`
- `early_exit`
- `router_nonfinite_fallback`

The benchmark report now exposes the route-state layer directly.

## Cross-lane architecture read

Regression is not blocked from LMA or task-level learning by missing plumbing.

Current code already has generic task-prior hooks on the regression surface:
- the regression benchmark runner passes `task_prior_dataset_key`
- the model builds a task context from regression router tokens
- `_maybe_attach_task_prior_router()` is shared rather than classification-only

So yes, regression can adopt the same broad family of ideas used in the binary line:
- LMA / task-level learning
- smarter task-conditioned routing priors
- feedback-updated dataset memory

But the current v1.3 sequencing still makes sense:
- first stabilize and explain the local regression routing circuit
- then improve realized specialist value
- only then spend heavily on regression meta-priors or cross-dataset LMA

The reason is architectural, not implementation readiness:
- regression already has a place to plug in a task prior
- what it still lacks is a clearly trustworthy local routing target to teach that prior
## First benchmark contract

Use:
- quick regression gate first
- then fold-0 mini-full regression

## Acceptance

This lane succeeds when a reviewer can explain every regression fallback case from the artifacts alone.
