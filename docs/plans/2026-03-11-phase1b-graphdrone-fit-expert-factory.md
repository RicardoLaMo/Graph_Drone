# Phase I-B GraphDrone.fit() Expert Factory

## Status

Phase I-B is implemented on `codex/graphdrone-fit-impl-plan`.

This phase does not introduce learned routing yet. It replaces manifest-only expert loading with a real portfolio-level expert-construction path inside `src/graphdrone_fit`, while keeping benchmark runners thin.

## What Was Implemented

Core package changes:

- `src/graphdrone_fit/expert_factory.py`
- `src/graphdrone_fit/model.py`
- `src/graphdrone_fit/config.py`
- `src/graphdrone_fit/__init__.py`

Benchmark bridge:

- `experiments/openml_regression_benchmark/src/graphdrone_fit_adapter.py`
- `experiments/openml_regression_benchmark/scripts/run_graphdrone_fit_openml.py`

Tests:

- `tests/graphdrone_fit/test_graphdrone_fit.py`
- `tests/graphdrone_fit/test_expert_factory.py`
- `tests/openml_regression_benchmark/test_graphdrone_fit_adapter.py`

## What The Phase Adds

- `GraphDrone.fit()` can now accept `expert_specs` and build a portfolio directly from training data.
- expert construction is expressed through typed `ExpertBuildSpec` objects rather than benchmark-local branching.
- the package now includes input adapters for:
  - identity subselect
  - PCA projection
- the benchmark layer now produces a reusable `GraphDroneBenchmarkExpertPlan` instead of fitting models inline.
- a thin benchmark runner now exercises the package end to end on OpenML data.

## Current Supported Expert Construction

Supported `model_kind` values in this phase:

- `constant`
- `linear`
- `tabpfn_regressor`

The implementation goal here is package/API consolidation, not final expert-family design.

## What Was Validated

Unit and adapter tests:

- `pytest -q tests/graphdrone_fit/test_graphdrone_fit.py tests/graphdrone_fit/test_expert_factory.py tests/openml_regression_benchmark/test_graphdrone_fit_adapter.py`
- result: `12 passed`

Smoke benchmark:

- runner: `experiments/openml_regression_benchmark/scripts/run_graphdrone_fit_openml.py`
- dataset: `houses`
- mode: `--smoke`
- output:
  - `experiments/openml_regression_benchmark/reports_fit_smoke/houses__r0f0__smoke/graphdrone_fit_results.json`
  - `experiments/openml_regression_benchmark/reports_fit_smoke/houses__r0f0__smoke/metrics.csv`

Observed smoke result:

- `GraphDroneFit_FULL` test RMSE: `0.2271`
- `GraphDroneFit_GEO` test RMSE: `0.3061`
- `GraphDroneFit_DOMAIN` test RMSE: `0.3427`
- `GraphDroneFit_LOWRANK` test RMSE: `0.4748`
- `GraphDroneFit_bootstrap` test RMSE: `0.2271`

This is expected for Phase I-B because the router is still explicit `bootstrap_full_only`.

## External Review

Gemini review:

- [phase1b Gemini review](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T202223Z/cleaned_output.txt)

Useful Gemini points:

- no confirmed blocker in the expert-factory or adapter split
- remaining risks are validation gaps:
  - persistence/serialization of spec-built portfolios
  - multi-device verification on the new factory path

Claude review:

- the Claude bundle was prepared twice, but the local CLI did not return a usable analysis artifact on this host
- the attempted bundles are:
  - `/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T202211Z-phase1b-code-review`
  - `/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T202353Z-phase1b-code-review-compact`

## Additional Guardrails Added

Phase I-B now has explicit negative-path coverage for:

- missing `y` when fitting from `expert_specs`
- unsupported `model_kind`

## What Is Still Not Done

- learned contextual routing
- support-aware token construction beyond the current placeholder path
- serialization of spec-built portfolios back to manifests or reusable artifacts
- multi-GPU validation of the new expert-factory path
- benchmark-wide portfolio runs through the new package

## Decision

Phase I-B is a valid checkpoint.

The package boundary now owns:

- expert construction
- expert descriptors
- expert batching
- model bootstrap integration

The benchmark layer no longer needs to define expert fitting logic directly.

The next implementation move is Phase I-C:

- connect `GraphDrone.fit()` to real portfolio expert construction on the registered benchmark surface
- keep runners thin
- still no learned contextual router yet
