# Phase I-A GraphDrone.fit() Skeleton

## Status

Phase I-A is implemented on `codex/graphdrone-fit-impl-plan`.

This phase does not implement learned routing yet. It establishes the core package boundary and a standalone bootstrap path so later phases can add expert-factory and contextual-router logic without pushing more model internals into benchmark scripts.

## What Was Implemented

New package:

- `src/graphdrone_fit/`

Core files:

- `config.py`
- `view_descriptor.py`
- `portfolio_loader.py`
- `expert_factory.py`
- `support_encoder.py`
- `token_builder.py`
- `set_router.py`
- `defer_integrator.py`
- `metrics.py`
- `model.py`

Benchmark bridge:

- `experiments/openml_regression_benchmark/src/graphdrone_fit_adapter.py`

Tests:

- `tests/graphdrone_fit/test_graphdrone_fit.py`
- `tests/openml_regression_benchmark/test_graphdrone_fit_adapter.py`

## What The Skeleton Does

- defines a public `GraphDrone.fit()` / `predict()` API
- loads a frozen portfolio from an explicit manifest
- validates typed expert descriptors
- builds per-row, per-expert tensor tokens
- runs an explicit Phase I-A bootstrap router
- applies an explicit defer-to-`FULL` integration path
- keeps benchmark-specific descriptor semantics out of the core package

## What Is Still Placeholder

- learned contextual routing
- support-summary encoding beyond zero-width placeholder tensors
- real portfolio expert construction from current benchmark artifacts
- benchmark runner wired to the new model package

## Validation

- `python -m py_compile ...`
- `pytest -q tests/graphdrone_fit/test_graphdrone_fit.py tests/openml_regression_benchmark/test_graphdrone_fit_adapter.py`
- result: `6 passed`

## External Review

Initial review:

- Claude identified one real blocker: benchmark-specific family hardcoding inside the adapter
- Gemini agreed the package split was directionally correct and called out the remaining adapter and placeholder risks

Fix applied:

- adapter family assignment now uses override-first semantics
- direct regression test added for overriding `LOWRANK`
- placeholder status for row-mean centering, zero-width support encoding, and bootstrap router is now explicit in code

Review artifacts:

- [phase1a code review Claude](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T194509Z-phase1a-code-review/analysis.claude.txt)
- [phase1a code review Gemini](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T194509Z-phase1a-code-review/20260311T194611Z/cleaned_output.txt)
- [phase1a postfix Claude](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T194802Z-phase1a-postfix-review/analysis.claude.txt)
- [phase1a postfix Gemini](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T194802Z-phase1a-postfix-review/20260311T194835Z/cleaned_output.txt)

## Decision

Phase I-A is now a valid checkpoint:

- package boundary exists
- benchmark coupling is reduced
- bootstrap path is explicit rather than hidden
- tests cover standalone loading and adapter override behavior

The next implementation move is Phase I-B:

- real portfolio expert-factory work
- no learned router yet
- no benchmark-script model growth
