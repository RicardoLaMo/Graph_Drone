# GraphDrone Row-Level Program

## Objective

GraphDrone should be developed as a row-level integration system, not as:

- a global blend with a routing label
- a growing list of attached experts or adapters
- a dataset-by-dataset tuning exercise

The near-term objective is to prove or falsify the claim that GraphDrone adds value by changing predictions on specific rows for interpretable reasons.

## Design Rules

1. Row-level, not global.
   Every new routing or integration idea must be compared against a fixed-weight or global baseline.

2. Integration, not attachment.
   A component is justified only if it changes the predictive decision path and we can explain what signal it contributes.

3. Portfolio before tuning.
   Registered datasets come before per-dataset optimization.

4. Mechanism before architecture.
   No new architecture branch should start until the current mechanism is explained on both a win case and a loss case.

5. Evidence before claims.
   Every phase needs fixed inputs, fixed artifacts, and explicit stop/go gates.

## Current Evidence

- California shows a real adaptive advantage over a fixed hedge.
- Houses shows only a small adaptive advantage and may still be partly noise.
- Prior TabPFN engineering-gap insertion did not recover the old routed effect by itself.
- The current GraphDrone win pattern is consistent with a conservative FULL-anchored hedge plus row-level correction, not hard expert switching.

This means the next problem is not "add more modules." The next problem is "prove what row-level signal is actually being used, and whether that generalizes across the registered portfolio."

## Program Structure

### Phase 0: Mechanism Proof

Purpose:
- determine whether adaptive routing is a real row-level mechanism or just a dressed-up global hedge

Scope:
- use the current diagnostics surfaces
- no new architecture
- no dynamic kNN or new views yet

Required outcome:
- one clear win-case explanation
- one clear non-win explanation
- one decision on whether houses is real evidence or noise

Frozen contract:
- [2026-03-11-phase0-router-mechanism-contract.md](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-row-integration-plan/docs/plans/2026-03-11-phase0-router-mechanism-contract.md)

### Phase 1: Registered Portfolio Baseline

Purpose:
- measure where GraphDrone helps across the registered dataset family before any per-dataset tuning

Scope:
- fixed hyperparameters
- fixed benchmark lane
- California included as a first-class required dataset

Required outcome:
- portfolio leaderboard
- dataset property table
- win/loss clustering hypothesis

Frozen contract:
- [2026-03-11-phase1-registered-portfolio-contract.md](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-row-integration-plan/docs/plans/2026-03-11-phase1-registered-portfolio-contract.md)
- [graphdrone_registered_dataset_matrix.json](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-row-integration-plan/experiments/openml_regression_benchmark/configs/graphdrone_registered_dataset_matrix.json)

### Phase 2: Architecture Decision

This phase does not start until Phase 0 and Phase 1 close.

Allowed outcomes:
- commit to a row-level integrated design because the mechanism is real and generalizes
- simplify toward global hedging if row-level effects do not justify the complexity
- explicitly scope GraphDrone to a dataset regime rather than pretending it is universal
- stop further architecture work if the mechanism remains unclear

## Codebase Discipline

To avoid drift:

- one worktree per phase
- one run contract per phase
- one summary artifact per phase
- no architecture branch without a closed prior phase
- no per-dataset tuning branch without explicit portfolio evidence

Suggested branch sequence:

- `phase0/router-mechanism-proof`
- `phase1/registered-portfolio-baseline`
- `phase2/row-level-integrated-architecture`

## Benchmark Discipline

For the current regression portfolio:

- primary metrics: `RMSE`, `MAE`, `R2`
- mechanism metrics: adaptive-minus-fixed delta, router entropy, oracle-match rate, weight variance, disagreement correlation
- portfolio metrics: win rate vs `TabPFN` and `AutoGluon`, dataset-rank profile, stability across seeds

Classification metrics such as `ROC-AUC`, `PR-AUC`, and `F1` belong in a separate classification benchmark lane, not in the current regression contract.

## Stop Conditions

Do not add new routing views, dynamic kNN, or new architecture branches until:

- California has a documented row-level mechanism explanation
- houses has been classified as real signal or noise
- at least one loss dataset has matching diagnostics
- the registered portfolio matrix has been run under fixed settings

## Program Question

The question for the next cycle is not:

- "What extra component can we attach to make the number better?"

It is:

- "What row-level decision signal is GraphDrone using, when does it help, and does that mechanism generalize across the registered portfolio?"
