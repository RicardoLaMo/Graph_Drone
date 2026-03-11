# GraphDrone.fit Classification Checkpoint

## Status

Classification is now implemented as a first-class `GraphDrone.fit()` path on branch `codex/graphdrone-fit-classification`.

## What Landed

- Core package classification support in `src/graphdrone_fit/`
- New OpenML classification benchmark lane in `experiments/openml_classification_benchmark/`
- Public benchmark surface restricted to:
  - `GraphDrone`
  - `TabPFN`
  - `TabR`
  - `TabM`
- Metrics standardized around:
  - `accuracy`
  - `macro F1`
  - `weighted F1`
  - `ROC-AUC`
  - `PR-AUC`
  - `log_loss`

## Important Implementation Decisions

1. `TabR` and `TabM` are scored from `predictions.npz`, not just upstream report summaries.
2. `TabR` numeric-only runs require explicit TOML null handling for `cat_policy`.
3. The classification summarizer no longer depends on `tabulate`.
4. Internal expert rows remain diagnostics-only and are written separately from the public benchmark summary.

## Validation Snapshot

- Targeted tests: `25 passed`
- Binary smoke:
  - `GraphDrone` on `diabetes`
  - `TabPFN` on `diabetes`
  - `TabR` on `diabetes`
  - `TabM` on `diabetes`
- Multiclass smoke:
  - `GraphDrone` on `maternal_health_risk`
- Public suite smoke:
  - `diabetes`, fold `0`, all public models

## Smoke Readout

`diabetes` suite smoke:

- `GraphDrone`: accuracy `0.7734`, macro F1 `0.7398`, ROC-AUC `0.8478`, PR-AUC `0.7490`
- `TabPFN`: accuracy `0.7734`, macro F1 `0.7398`, ROC-AUC `0.8480`, PR-AUC `0.7492`
- `TabM`: accuracy `0.7422`, macro F1 `0.7076`, ROC-AUC `0.7877`, PR-AUC `0.6632`
- `TabR`: accuracy `0.6641`, macro F1 `0.4397`, ROC-AUC `0.7539`, PR-AUC `0.6066`

`maternal_health_risk` GraphDrone smoke:

- accuracy `0.8432`
- macro F1 `0.8453`
- ROC-AUC OVR macro `0.9383`
- PR-AUC OVR macro `0.8951`

## External Review

- Gemini implementation review:
  - `.gemini-cross-checks/20260311T194500Z-classification-impl-review/20260311T231942Z/cleaned_output.txt`
- Main useful issue caught during live validation:
  - baseline bridge correctness for `TabR` null categorical policy
