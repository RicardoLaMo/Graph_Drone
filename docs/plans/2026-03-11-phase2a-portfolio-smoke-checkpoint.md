# 2026-03-11 Phase II-A Portfolio Smoke Checkpoint

## Scope

Branch: `codex/graphdrone-fit-phase2a-portfolio`

Goal:
- extend `GraphDrone.fit()` with portfolio-generic geometry experts based on `LID` and `LOF`
- keep the public benchmark surface to `GraphDrone`, `TabPFN`, `TabR`, and `TabM`
- run a smoke portfolio pass across the registered regression datasets

## Implemented Changes

- Added `geometry_signal` expert family and `derived_features` projection kind in `src/graphdrone_fit/view_descriptor.py`.
- Added `GeometryFeatureAdapter` in `src/graphdrone_fit/expert_factory.py`.
  - builds normalized geometry-derived features from an anchor feature space
  - currently supports `lid`, `lof`, and `mean_knn_distance`
- Extended `experiments/openml_regression_benchmark/src/graphdrone_fit_adapter.py` so the internal expert plan now appends:
  - `GEOMETRY_1 = GEOMETRY_LID`
  - `GEOMETRY_2 = GEOMETRY_LOF`
- Expanded legacy quality encodings to expert-plan order so geometry experts inherit anchor/FULL quality channels instead of breaking token shape alignment.
- Then fixed the first integration mismatch:
  - geometry experts now receive geometry-specific quality channels:
    - `quality_geometry_lid`
    - `quality_geometry_lof`
    - `quality_geometry_mean_knn_distance`
  - router diagnostics now expose `active_specialist_ids` in addition to `active_specialist_indices`
- Hardened `GeometryFeatureAdapter` against degenerate LID cases where identical neighbor radii would otherwise create `inf`.
- Added Phase II-A portfolio config and launcher:
  - `experiments/openml_regression_benchmark/configs/phase2a_registered_portfolio.json`
  - `experiments/openml_regression_benchmark/scripts/run_phase2a_registered_portfolio.sh`

## Validation

- Targeted tests passed:
  - `31 passed`
- Smoke suite completed under:
  - `experiments/openml_regression_benchmark/reports_phase2a_smoke_portfolio_r4`
- Targeted post-fix rerun completed under:
  - `experiments/openml_regression_benchmark/reports_phase2a_geometry_graphdrone_rerun`

## Public Smoke Results

Smoke uses repeat `0`, fold `0`, and `--smoke` mode.

| Dataset | GraphDrone | TabPFN | TabR | TabM | Best |
|---|---:|---:|---:|---:|---|
| `airfoil_self_noise` | `1.1853` | `1.1007` | `4.6638` | `6.3665` | `TabPFN` |
| `california_housing_openml` | `0.4293` | `0.4655` | `0.6771` | `0.6735` | `GraphDrone` |
| `concrete_compressive_strength` | `4.2710` | `4.2364` | `11.7487` | `8.3370` | `TabPFN` |
| `diamonds` | `590.6060` | `636.8654` | `2366.5286` | `2166.6186` | `GraphDrone` |
| `healthcare_insurance_expenses` | `4145.7877` | `4133.0502` | `6891.5849` | `9558.1442` | `TabPFN` |
| `houses` | `0.2261` | `0.2391` | `0.3630` | `0.4157` | `GraphDrone` |
| `miami_housing` | `79436.9240` | `110047.5022` | `161070.3755` | `166404.9415` | `GraphDrone` |
| `used_fiat_500` | `698.3717` | `684.9147` | `1003.2854` | `1333.5767` | `TabPFN` |
| `wine_quality` | `0.6477` | `0.6590` | `0.6791` | `0.7133` | `GraphDrone` |

Smoke win count:
- `GraphDrone`: `6 / 9`
- `TabPFN`: `3 / 9`

## Internal Geometry Observations

The geometry experts are real and sometimes strong.

Datasets where a geometry expert beats the anchor:
- `concrete_compressive_strength`
  - anchor `4.2155`
  - best geometry `GEOMETRY_1 = 4.1581`
- `diamonds`
  - anchor `590.6060`
  - best geometry `GEOMETRY_1 = 589.8617`
- `healthcare_insurance_expenses`
  - anchor `4145.7877`
  - best geometry `GEOMETRY_1 = 4080.6529`
- `houses`
  - anchor `0.2271`
  - best geometry `GEOMETRY_2 = 0.2231`

Datasets where the anchor remains best:
- `airfoil_self_noise`
- `california_housing_openml`
- `miami_housing`
- `used_fiat_500`
- `wine_quality`

## Current Mechanism Risk

There is a real integration mismatch:
- some geometry experts outperform the anchor internally
- but the public `GraphDrone` row often remains at or very near the anchor

Examples:
- `concrete_compressive_strength`
  - `GraphDrone = 4.2710`
  - anchor `= 4.2155`
  - `GEOMETRY_1 = 4.1581`
- `healthcare_insurance_expenses`
  - `GraphDrone = 4145.7877`
  - anchor `= 4145.7877`
  - `GEOMETRY_1 = 4080.6529`
- `houses`
  - `GraphDrone = 0.2261`
  - anchor `= 0.2271`
  - `GEOMETRY_2 = 0.2231`

This means the new expert family is not merely cosmetic, but the current contextual routing/integration is not yet harvesting all of the available specialist gain.

## Post-Fix Rerun

After adding geometry-aware quality channels and active specialist id reporting, the targeted rerun on the four geometry-relevant smoke cases changed behavior as follows:

| Dataset | Old GraphDrone | New GraphDrone | Delta |
|---|---:|---:|---:|
| `houses` | `0.2261` | `0.2273` | `+0.0013` |
| `concrete_compressive_strength` | `4.2710` | `4.3173` | `+0.0462` |
| `healthcare_insurance_expenses` | `4145.7877` | `4101.8342` | `-43.9535` |
| `diamonds` | `590.6060` | `590.6060` | `+0.0000` |

What changed mechanistically:
- `houses`
  - router now reports active ids:
    - `SEMANTIC_1`
    - `SEMANTIC_2`
    - `GEOMETRY_1`
  - but `GEOMETRY_2` is still the strongest internal geometry expert and the public row regressed slightly
- `healthcare_insurance_expenses`
  - public GraphDrone improved materially after the fix
  - router now fully activates specialist routing with geometry ids visible
- `concrete_compressive_strength`
  - specialist routing became much more aggressive
  - but the public row regressed, which suggests the new geometry quality channels alone do not solve specialist selection quality

This updated the diagnosis:
- the original issue was real: copying anchor quality onto geometry experts was masking specialist signal
- fixing that issue helped observability and improved at least one dataset materially
- but the remaining gap is still architectural/integration quality, not just missing metadata

## Interpretation

What is supported by the smoke run:
- The public benchmark surface is now clean.
- Portfolio-generic geometry experts can carry real signal.
- The new `GraphDrone.fit()` path already beats `TabPFN` on several datasets in smoke mode.

What is not supported yet:
- A claim that the current contextual router is using geometry experts effectively.
- A claim that the geometry experts generalize in full-quality runs.
- A claim that the remaining gap is purely routing rather than quality-alignment between quality channels and geometry specialists.

## Next Checks

1. Review the implementation and smoke evidence with Gemini and Claude.
2. Decide whether the current mismatch is:
   - routing objective weakness
   - remaining token quality mismatch for geometry experts
   - or sparse-top-k behavior that suppresses useful specialists
3. If the code review is clean, commit the Phase II-A checkpoint and then run a fuller registered-portfolio evaluation.
