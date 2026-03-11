# Phase 0 Router Mechanism Readout

## Scope

Phase 0 was executed on the unified OpenML benchmark surface in the `codex/phase0-router-mechanism-proof` worktree.

Datasets covered:

- `california_housing_openml` as the primary win-case probe
- `houses` as the stability probe
- `miami_housing` as a loss case
- `diamonds` as a loss case

No new architecture was introduced. The only code added in this phase was benchmark-surface support and analysis tooling.

## Key Results

### California (repeat `0`, fold `0`)

- `GraphDrone_FULL = 0.4018`
- `GraphDrone_router = 0.3895`
- `GraphDrone_router_fixed = 0.3895`
- `GraphDrone_crossfit = 0.3895`
- `GraphDrone_crossfit_fixed = 0.3899`

Mechanism read:

- adaptive routing still beats `FULL`
- but adaptive barely beats the fixed hedge on this unified surface
- adaptive minus fixed for router is only about `+0.00002` RMSE
- the router still moves weights materially:
  - mean test weight L1 shift is about `0.116`
  - mean test weights are `FULL 0.766`, `GEO 0.217`, `SOCIO 0.008`, `LOWRANK 0.009`

Interpretation:

- the old California routed advantage does not transfer cleanly to this unified OpenML surface as a strong adaptive-over-fixed result
- there is real row-level movement, but not a material gain over a global hedge here

### Houses seed sweep (`seed/split_seed = 41..50`, fixed outer split)

Means and standard deviations:

- `GraphDrone_FULL = 0.20575 ± 0.00277`
- `GraphDrone_router = 0.20223 ± 0.00192`
- `GraphDrone_router_fixed = 0.20262 ± 0.00196`
- `GraphDrone_crossfit = 0.20223 ± 0.00193`
- `GraphDrone_crossfit_fixed = 0.20255 ± 0.00206`

Adaptive-over-fixed stability:

- router adaptive minus fixed mean: `+0.000393`
- router adaptive minus fixed std: `0.000209`
- router adaptive minus fixed positive fraction: `1.0`

Interpretation:

- houses is not noise in the simple sign sense; the router beats the fixed hedge on all ten seed settings
- but the effect size is still small
- this is stable small-signal evidence, not a large mechanism win

### Miami Housing loss case

- `GraphDrone_FULL = 82529.36`
- `GraphDrone_router = 83079.12`
- `GraphDrone_router_fixed = 88149.98`

Interpretation:

- adaptive routing clearly improves over the fixed hedge
- but both routed variants remain worse than `FULL`
- the mechanism is doing something real, but it is not enough to justify moving away from the monolithic full expert on this dataset

### Diamonds loss case

- `GraphDrone_FULL = 522.51`
- `GraphDrone_router = 539.34`
- `GraphDrone_router_fixed = 559.39`

Interpretation:

- same pattern as Miami
- adaptive routing improves materially over a fixed hedge
- but routed integration still loses badly to `FULL`

## Gate Status

### Gate P0.1: California Mechanism

Status: `FAILED`

Reason:

- adaptive routing does not beat the fixed hedge by a material amount on the unified OpenML California surface
- row-level weight movement is real, but the payoff is not material enough here

### Gate P0.2: Houses Stability

Status: `WEAK PASS`

Reason:

- adaptive routing beats the fixed hedge consistently across all ten seed settings
- but the magnitude is still small, so houses is evidence of stable small gain, not strong gain

### Gate P0.3: Loss-Case Explanation

Status: `PARTIAL`

Reason:

- Miami and Diamonds give a clear descriptive pattern:
  - adaptive movement helps relative to a fixed hedge
  - but not enough to beat `FULL`
- however, this is still descriptive rather than fully causal
- we do not yet know which row-level conditions make the routed movement net harmful versus net helpful

## Current Best Interpretation

The strongest current interpretation is:

- GraphDrone does perform real row-level movement
- that movement is not just a global hedge
- but the current row-level mechanism is not strong enough to justify a general architecture claim

More precisely:

- on some datasets, adaptive movement gives a stable but small gain over a fixed hedge
- on loss cases, adaptive movement rescues a bad fixed hedge somewhat, but still cannot compete with the `FULL` expert
- therefore the current mechanism looks like a weak row-level corrector, not yet a robust integrated decision system

## External Review

Claude review:

- [phase0-results-review](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T144336Z-phase0-results-review/analysis.claude.txt)

Claude’s strongest useful point:

- Phase 0 should not be treated as complete evidence for advancing to portfolio-scale claims or new architecture

Gemini review:

- [phase0-results-review](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T145000Z-phase0-results-review/20260311T144350Z/cleaned_output.txt)

Gemini’s strongest useful point:

- the current mechanism is functioning, but it still behaves more like a weak dynamic corrector than a strong integrator

## Decision

Do **not** start Phase 1 yet.

Instead, execute a short `Phase 0b`:

1. Repeat California on additional outer folds or matched repeats to determine whether the tiny adaptive-over-fixed gain is real or noise on the unified surface.
2. Extend loss-case analysis on Miami and Diamonds from descriptive summaries to row-level causal profiling.
3. Keep houses as stable small-signal evidence, but do not use it alone to justify architecture changes.

## Non-Decision

This phase does **not** justify:

- a new routing architecture branch
- dynamic kNN integration
- more view attachments
- per-dataset tuning

The evidence is useful, but it is not yet strong enough for that.
