# Phase 0 Router Mechanism Contract

## Purpose

Phase 0 exists to prove or falsify the claim that GraphDrone's advantage comes from row-level integration rather than a global hedge.

No new architecture work is allowed in this phase.

## Frozen Dataset Subset

Primary datasets:

- `california_housing_openml`
  Why: strongest current adaptive-over-fixed evidence
- `houses`
  Why: current adaptive gain is small and must be tested for noise
- `miami_housing`
  Why: strong loss case against foundation; useful for failure-mode profiling
- `diamonds`
  Why: another loss case with a different scale and structure than California

Source of truth:
- [phase0_router_mechanism_subset.json](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-row-integration-plan/experiments/openml_regression_benchmark/configs/phase0_router_mechanism_subset.json)

## Required Questions

1. Which rows benefit from adaptive routing relative to a fixed hedge?
2. Are those rows characterized by disagreement, uncertainty, density, target scale, or other measurable structure?
3. Is the houses adaptive gain stable across seeds or just evaluation noise?
4. Do loss datasets show the same routing behavior without payoff, or a qualitatively different behavior?

## Required Artifacts

For every Phase 0 dataset run:

- adaptive vs fixed test comparison
- `router_diagnostics.json`
- `router_fixed_diagnostics.json`
- row-level prediction bundle
- one short mechanism note

For houses specifically:

- repeated-seed stability summary
- mean and standard deviation of adaptive-minus-fixed test RMSE

## Allowed Metrics

Primary:

- `RMSE`
- `MAE`
- `R2`

Mechanism:

- adaptive-minus-fixed test delta
- top-weight vs oracle-best fraction
- anchor-oracle gap
- weight entropy
- per-row weight variance
- disagreement and uncertainty correlations

## Required Gates

### Gate P0.1: California Mechanism

Pass only if:

- adaptive beats fixed on test by a material amount
- and the rows receiving different weights have at least one interpretable measurable pattern

### Gate P0.2: Houses Stability

Pass only if one of the following is true:

- the adaptive-minus-fixed gain remains positive and stable across the seed sweep
- or the gain is declared statistically weak and houses is removed as mechanism evidence

### Gate P0.3: Loss-Case Explanation

Pass only if at least one loss dataset provides a concrete explanation for why the same routing machinery does not help.

## Forbidden Moves

Do not do any of the following during Phase 0:

- add dynamic kNN
- add new views
- add new experts
- tune per dataset
- use portfolio rank as a substitute for mechanism evidence

## Output

Phase 0 must end with a short decision note that answers:

- is GraphDrone genuinely row-level?
- on which rows?
- under what measurable conditions?
- is houses valid evidence or noise?
- what mechanism hypothesis will Phase 1 test at portfolio scale?
