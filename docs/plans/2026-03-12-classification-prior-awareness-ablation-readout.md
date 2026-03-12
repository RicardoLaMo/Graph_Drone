# 2026-03-12 Classification Prior-Awareness Ablation Readout

## Objective

Evaluate the classification `GraphDrone.fit()` prior-awareness port on the 9-dataset registered OpenML portfolio and isolate whether the new `field_aware` router ingestion helps relative to the safer `flat` token encoder.

## Public Evaluation Surface

- `GraphDrone`
- `TabPFN`
- `TabM`

For this ablation, only `GraphDrone` was rerun:

- `field_aware` token encoder
- `flat` token encoder

The public metrics are:

- ROC-AUC
- PR-AUC
- macro F1
- accuracy
- log-loss

## Portfolio Outputs

- field-aware run:
  - `experiments/openml_classification_benchmark/reports_prior_awareness_portfolio_field_r2/`
- flat run:
  - `experiments/openml_classification_benchmark/reports_prior_awareness_portfolio_flat_r1/`

## Headline Result

The richer prior-aware token/support pipeline is worth keeping, but the current `field_aware` router should **not** become the default classification path.

`flat` beats `field_aware` on most datasets and most public metrics.

### Metric-Level Win Counts

`flat` over `field_aware`:

- ROC-AUC: `6 / 9`
- PR-AUC: `5 / 9`
- macro F1: `5 / 9`
- accuracy: `7 / 9`
- log-loss: `6 / 9`

Relative to the baseline `GraphDrone` classification branch:

- `field_aware` improves on only a minority of datasets
- `flat` improves on a majority of datasets for every public metric except PR-AUC, where it still improves on `5 / 9`

### Representative Datasets

- `anneal`:
  - `field_aware` is worse than `flat` on ROC-AUC, PR-AUC, accuracy, and log-loss
- `bioresponse`:
  - `flat` improves all public metrics over `field_aware`
- `diabetes`:
  - `flat` recovers the baseline-level macro F1 and accuracy, while `field_aware` regresses
- `students_dropout_and_academic_success`:
  - both prior-aware variants improve PR-AUC and macro F1 over the baseline, but `flat` remains more balanced

## Interpretation

The classification readout matches the regression lesson:

1. keep richer prior-aware support and token semantics
2. do not promote the current `field_aware` router as default
3. treat expert quality and support modeling as the next likely bottleneck

This is an integration result, not a benchmark-surface regression:

- the public benchmark remains `GraphDrone` vs peer models
- the token encoder choice is an internal GraphDrone implementation detail

## Decision

Use the richer prior-aware support/token pipeline going forward, but default the classification router ingestion to `flat` until a stronger field-aware encoder can beat it cleanly across the portfolio.
