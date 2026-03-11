# Phase I-C Token Builder And Support Encoder

## Status

Phase I-C is implemented on `codex/graphdrone-fit-impl-plan`.

This phase does not add a learned contextual router yet. It upgrades the package from placeholder token slots to a traceable token surface with:

- named per-expert quality fields
- pooled support summaries
- normalized descriptor features
- a compatibility bridge from legacy scalar priors into row-level token fields

## What Was Implemented

Core package changes:

- `src/graphdrone_fit/token_builder.py`
- `src/graphdrone_fit/support_encoder.py`
- `src/graphdrone_fit/model.py`
- `src/graphdrone_fit/__init__.py`

Benchmark bridge:

- `experiments/openml_regression_benchmark/src/graphdrone_fit_adapter.py`
- `experiments/openml_regression_benchmark/scripts/run_graphdrone_fit_openml.py`

Tests:

- `tests/graphdrone_fit/test_graphdrone_fit.py`
- `tests/graphdrone_fit/test_token_builder.py`
- `tests/openml_regression_benchmark/test_graphdrone_fit_adapter.py`

## What The Phase Adds

### 1. Named quality token fields

Legacy scalar priors are no longer only flat runner-local vectors.

They are now mapped into per-row, per-expert quality tokens:

- `quality_sigma2_self`
- `quality_sigma2_centered`
- `quality_pair_overlap_mean`
- `quality_pair_overlap_max`
- `quality_mean_J_global`

### 2. Support encoder with pooled summaries

The support path can now consume:

- explicit `SupportEncoding`
- 3D `[N, E, S]` support tensors
- 4D `[N, E, K, F]` support tensors

The 4D path is summarized into:

- support means
- support standard deviations
- support absolute maxima
- support count

### 3. Descriptor features suitable for later contextual routing

Descriptor fields are still explicit and named, but numeric descriptor channels are now normalized before concatenation with one-hot family and projection features.

### 4. Traceable token diagnostics

`GraphDrone.predict(..., return_diagnostics=True)` now records:

- token field slices
- token field names
- quality feature names
- support feature names

This makes later routing decisions auditable at the token level.

## What Was Validated

Targeted tests:

- `pytest -q tests/graphdrone_fit/test_graphdrone_fit.py tests/graphdrone_fit/test_expert_factory.py tests/graphdrone_fit/test_token_builder.py tests/openml_regression_benchmark/test_graphdrone_fit_adapter.py`
- result: `18 passed`

Smoke benchmark:

- runner: `experiments/openml_regression_benchmark/scripts/run_graphdrone_fit_openml.py`
- dataset: `houses`
- mode: `--smoke`
- output:
  - `experiments/openml_regression_benchmark/reports_fit_smoke/houses__r0f0__smoke/graphdrone_fit_results.json`
  - `experiments/openml_regression_benchmark/reports_fit_smoke/houses__r0f0__smoke/metrics.csv`

Important smoke diagnostics:

- quality token slice: `[3, 8]`
- descriptor token slice: `[8, 16]`
- descriptor family fields now include:
  - `descriptor_family_FULL`
  - `descriptor_family_domain_semantic`
  - `descriptor_family_structural_subspace`

The benchmark runner still does not provide explicit support tensors, so the smoke run correctly reports an empty support slice. The pooled support path is exercised in unit tests rather than this smoke artifact.

## External Review

Gemini review:

- [phase1c Gemini review](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T203331Z/cleaned_output.txt)

Useful Gemini points:

- the token-based architecture is now real rather than cosmetic
- the main remaining gap is that the benchmark runner still does not exercise the support branch in an end-to-end artifact

Claude review:

- [phase1c Claude review](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T203323Z-phase1c-code-review/analysis.claude.txt)

Useful Claude points that were applied:

- benchmark family mapping no longer collapses `GEO` and `DOMAIN` into a generic bootstrap family
- descriptor numeric channels are now normalized before later attention-style use

Remaining Claude point that is valid but not a blocker for this phase:

- the current router is still `bootstrap_full_only`, so the smoke artifact proves token construction and diagnostics, not contextual routing gains

## What Is Still Not Done

- learned contextual set-router
- sparse defer-to-`FULL` behavior beyond the bootstrap placeholder
- support tensors from the benchmark runner’s real neighborhood objects
- end-to-end verification that a learned router actually consumes the new token fields

## Decision

Phase I-C is a valid checkpoint.

The package now has:

- a portfolio-level expert factory
- a per-row, per-view token builder with named fields
- a support encoder that can summarize structured support inputs
- a legacy-prior bridge that feeds token subfields instead of runner-local scalar routing

The next implementation move is Phase I-D:

- implement the first lightweight contextual set-router over these tokens
- keep the defer-to-`FULL` path explicit
- require portfolio comparison against fixed-hedge baselines
