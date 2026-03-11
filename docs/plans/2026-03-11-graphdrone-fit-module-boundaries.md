# GraphDrone.fit() Module Boundaries

## Purpose

This note maps current experiment-centric code to the target `GraphDrone.fit()` package boundaries so implementation can proceed without drifting back into runner-script logic.

## Current To Target Mapping

| Current Surface | Current Responsibility | Target Home | Keep / Move |
|---|---|---|---|
| `experiments/tabpfn_view_router/src/router.py` | scalar-quality routing, fixed hedges, regression scoring, diagnostics helpers | `src/graphdrone_fit/set_router.py`, `src/graphdrone_fit/defer_integrator.py`, `src/graphdrone_fit/metrics.py` | split |
| `experiments/tabpfn_view_router/src/data.py` | California-specific splits, views, kNN quality features | `src/graphdrone_fit/support_encoder.py`, benchmark-specific split code stays under experiments | split |
| `experiments/openml_regression_benchmark/src/openml_tasks.py` | dataset registry, dataset-specific view specs | `src/graphdrone_fit/expert_factory.py` consumes registry descriptors; registry itself stays under benchmark | bridge |
| `experiments/openml_regression_benchmark/scripts/run_graphdrone_openml.py` | full model assembly plus benchmark IO | `src/graphdrone_fit/model.py` plus thin benchmark adapter | shrink |

## New Package Layout

```text
src/graphdrone_fit/
  __init__.py
  config.py
  model.py
  portfolio_loader.py
  expert_factory.py
  view_descriptor.py
  token_builder.py
  support_encoder.py
  set_router.py
  defer_integrator.py
  metrics.py
```

## Boundary Rules

### `portfolio_loader.py`

- Loads frozen experts and descriptors from explicit config
- Must not assume the OpenML benchmark directory as the runtime root
- Must support a standalone smoke path outside the registered matrix

### `model.py`

- Owns the public object
- Exposes `fit()` and `predict()`
- Does not know OpenML CLI details

### `expert_factory.py`

- Builds experts for a dataset from descriptors
- Does not compute routing weights
- Does not own benchmark output formatting

### `view_descriptor.py`

- Normalizes expert metadata into a typed descriptor
- Must work when a dataset has no geography-specific expert

### `token_builder.py`

- Builds per-row, per-expert tensors
- Must accept variable expert counts
- May consume scalar priors as fields, but not as the whole representation

### `support_encoder.py`

- Converts local support statistics into tensor summaries
- Replaces the current "all information collapsed into sigma2/J" pattern

### `set_router.py`

- Contextual routing only
- No data loading
- No metrics IO
- No benchmark branching

### `defer_integrator.py`

- Applies sparse routing and defer-to-`FULL`
- Must emit diagnostics
- Must not silently collapse to `FULL`

### `metrics.py`

- Regression metrics now
- Classification metrics later
- Keeps metric logic out of router code

## Files That Must Stay Thin

- benchmark runners
- dataset-manifest utilities
- reporting scripts

If any of those files start owning token construction, sparse routing, or defer logic, the phase has drifted.
