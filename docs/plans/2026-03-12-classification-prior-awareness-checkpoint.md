# 2026-03-12 Classification Prior-Awareness Checkpoint

## Goal

Port the regression-side prior-awareness pipeline into the `GraphDrone.fit()` classification path without drifting into dataset-specific view tuning.

## Scope Completed

- Ported richer support encoding into classification:
  - anchor-relative support moments
  - weighted support centroid features
  - support concentration and radius features
- Ported richer token semantics into classification:
  - anchor-relative quality channels
  - row-relative quality and support channels
- Ported router ingestion ablation into classification:
  - `field_aware`
  - `flat`
- Exposed the ablation through the public classification GraphDrone runner and suite queue.
- Updated classification tests to reflect the richer token/support schema and router diagnostics.

## Files Changed

- `src/graphdrone_fit/config.py`
- `src/graphdrone_fit/model.py`
- `src/graphdrone_fit/support_encoder.py`
- `src/graphdrone_fit/token_builder.py`
- `src/graphdrone_fit/set_router.py`
- `experiments/openml_classification_benchmark/scripts/run_graphdrone_fit_openml.py`
- `experiments/openml_classification_benchmark/scripts/run_openml_suite.py`
- `tests/graphdrone_fit/test_graphdrone_fit.py`
- `tests/graphdrone_fit/test_set_router.py`
- `tests/graphdrone_fit/test_token_builder.py`

## Validation

- Focused classification suite:
  - `22 passed`
- Gemini code review:
  - no concrete blockers
  - main caution was to confirm classification loss alignment and run both router token encoder modes end to end
- Real runner smoke on `diabetes`:
  - `field_aware`: accuracy `0.7734`, macro F1 `0.7398`, ROC-AUC `0.8477`, PR-AUC `0.7489`, log-loss `0.45520`
  - `flat`: accuracy `0.7734`, macro F1 `0.7398`, ROC-AUC `0.8477`, PR-AUC `0.7488`, log-loss `0.45521`

## Interpretation

This checkpoint validates the classification prior-awareness port as an implementation milestone, not a performance claim.

The next step is the portfolio-level `GraphDrone`-only ablation:

1. run the 9-dataset classification portfolio with `field_aware`
2. rerun the same portfolio with `flat`
3. compare on:
   - ROC-AUC
   - PR-AUC
   - macro F1
   - accuracy
   - log-loss

## Gate

Do not promote the classification `field_aware` router by default unless the portfolio ablation shows a consistent win over `flat`.
