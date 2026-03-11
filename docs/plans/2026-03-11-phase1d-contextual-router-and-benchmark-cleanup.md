# Phase I-D Contextual Router And Benchmark Cleanup

## Status

Phase I-D is implemented on `codex/graphdrone-fit-phase1d`.

This phase closes the transition from the bootstrap `GraphDrone.fit()` scaffold to a first real package-level benchmark path:

- the public benchmark surface now reports only `GraphDrone`
- internal expert rows are diagnostics-only
- the package now supports `fit_router()` with a learned contextual sparse router
- worktree-relative runtime assumptions were cleaned up for the suite and baseline runners

## What Was Implemented

Core package:

- `src/graphdrone_fit/config.py`
- `src/graphdrone_fit/set_router.py`
- `src/graphdrone_fit/model.py`

Benchmark bridge and public runner:

- `experiments/openml_regression_benchmark/src/graphdrone_fit_adapter.py`
- `experiments/openml_regression_benchmark/scripts/run_graphdrone_fit_openml.py`
- `experiments/openml_regression_benchmark/scripts/run_openml_suite.py`
- `experiments/openml_regression_benchmark/scripts/summarize_openml_suite.py`

Baseline runtime support:

- `experiments/tab_foundation_compare/src/runtime_support.py`

Tests:

- `tests/graphdrone_fit/test_graphdrone_fit.py`
- `tests/graphdrone_fit/test_set_router.py`
- `tests/openml_regression_benchmark/test_graphdrone_fit_adapter.py`
- `tests/openml_regression_benchmark/test_run_openml_suite.py`
- `tests/test_runtime_support.py`

## What Changed

### 1. Public benchmark cleanup

The benchmark suite no longer exposes internal GraphDrone expert names as public peer models.

Public model identity is now:

- `GraphDrone`

Internal experts are written only to:

- `graphdrone_internal_experts.json`

The smoke suite summary now compares:

- `GraphDrone`
- `TabPFN`
- `TabR`
- `TabM`

### 2. Generic internal expert identities

The benchmark adapter still builds experts from the registered dataset views, but it no longer passes view names through as public-facing expert ids.

Examples:

- `FULL` registry view -> internal `ANCHOR`
- domain-like registry views -> `SEMANTIC_1`, `SEMANTIC_2`
- low-rank transforms -> `SUBSPACE_1`

The original registry view names are preserved only in the internal expert-view map and descriptor tags.

### 3. First contextual sparse router

`GraphDrone.fit_router()` is now implemented at the package level.

The router:

- consumes per-row, per-expert tokens
- compares each expert token against the anchor token and global token context
- predicts sparse specialist weights
- predicts a defer-to-anchor probability
- is trained directly on validation labels with an internal early-stopping split

The current router kind is:

- `contextual_sparse_mlp`

This is not a fixed hedge and not runner-local logic. It is part of the `src/graphdrone_fit/` package surface.

### 4. Worktree-safe runtime path resolution

Two real path bugs were fixed:

- `run_openml_suite.py` now discovers the shared `.venv-h200` interpreter by walking the worktree parent chain
- `runtime_support.py` now resolves `.external/tabr` and `.external/tabm/paper` from the repo root or sibling worktrees

That keeps the benchmark runnable from research worktrees instead of only from one checkout layout.

## What Was Validated

Targeted tests:

- `pytest -q tests/graphdrone_fit/test_graphdrone_fit.py tests/graphdrone_fit/test_set_router.py tests/openml_regression_benchmark/test_graphdrone_fit_adapter.py tests/openml_regression_benchmark/test_run_openml_suite.py tests/test_runtime_support.py`
- result: `26 passed`

Package runner smoke:

- `python experiments/openml_regression_benchmark/scripts/run_graphdrone_fit_openml.py --dataset houses --fold 0 --repeat 0 --smoke ...`
- output:
  - `experiments/openml_regression_benchmark/reports_fit_phase1d_smoke/houses__r0f0__smoke/graphdrone_results.json`
  - `experiments/openml_regression_benchmark/reports_fit_phase1d_smoke/houses__r0f0__smoke/graphdrone_internal_experts.json`

Suite smoke:

- `python experiments/openml_regression_benchmark/scripts/run_openml_suite.py --datasets houses --folds 0 --models GraphDrone TabPFN TabR TabM --smoke ...`
- summary:
  - `experiments/openml_regression_benchmark/reports_fit_phase1d_suite_smoke_r3/openml_benchmark_summary.md`

Smoke summary values on `houses`:

- `GraphDrone = 0.2290`
- `TabPFN = 0.2391`
- `TabR = 0.3630`
- `TabM = 0.4157`

## External Review

Gemini review:

- [phase1d Gemini review](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T213645Z-phase1d-review/20260311T213731Z/cleaned_output.txt)

Gemini found no confirmed implementation blocker in the provided evidence. The useful remaining risks it called out are:

- full-portfolio router stability is still unproven
- sibling-worktree `.external/` discovery should be exercised in more layouts
- router fit-time cost is not yet summarized

Claude review:

- the Claude packet was created under `/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T213645Z-phase1d-review`
- this pass produced no usable analysis text, so it is treated as inconclusive rather than evidence

## Decision

Phase I-D is a valid implementation checkpoint.

The key architectural boundary is now correct:

- public benchmarks compare `GraphDrone` against external baselines
- internal experts are hidden in diagnostics
- routing lives inside `GraphDrone.fit()` instead of in benchmark-specific glue code

This does **not** prove that the new contextual router is the final GraphDrone design. It proves the package and benchmark surface are finally aligned well enough to continue Phase II work without falling back into `FULL/GEO/DOMAIN/LOWRANK` public benchmarking.
