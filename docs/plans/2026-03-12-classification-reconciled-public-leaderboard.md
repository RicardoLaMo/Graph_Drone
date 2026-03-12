# 2026-03-12 Classification Reconciled Public Leaderboard

## Objective

Reconcile the classification branches into one public comparison surface:

- baseline `GraphDrone`
- prior-aware `GraphDrone` with default `flat`
- `TabPFN`
- `TabM`

## Sources

- baseline public summary:
  - `../graphdrone-fit-classification/experiments/openml_classification_benchmark/reports_phase2b_full_portfolio/openml_benchmark_summary.csv`
- prior-aware flat summary:
  - `experiments/openml_classification_benchmark/reports_prior_awareness_portfolio_flat_r1/openml_benchmark_summary.csv`

## High-Level Readout

The prior-aware classification branch improves `GraphDrone` over the baseline branch on most public metrics:

- ROC-AUC: `6 / 9`
- PR-AUC: `5 / 9`
- macro F1: `6 / 9`
- accuracy: `6 / 9`
- log-loss: `6 / 9`

But `TabPFN` remains the strongest overall classifier on this portfolio:

- prior-aware `GraphDrone` beats `TabPFN` on:
  - ROC-AUC: `1 / 9`
  - PR-AUC: `0 / 9`
  - macro F1: `2 / 9`
  - accuracy: `3 / 9`
  - log-loss: `2 / 9`

`GraphDrone` remains clearly stronger than `TabM`:

- ROC-AUC: `9 / 9`
- PR-AUC: `9 / 9`
- macro F1: `8 / 9`
- accuracy: `9 / 9`
- log-loss: `9 / 9`

## Primary ROC-AUC Table

| Dataset | GraphDrone Baseline | GraphDrone Prior Flat | TabPFN | TabM | Best ROC-AUC |
|---|---:|---:|---:|---:|---|
| `anneal` | 0.999145 | 0.999627 | 0.999779 | 0.977438 | TabPFN |
| `apsfailure` | 0.991590 | 0.991648 | 0.992324 | 0.985869 | TabPFN |
| `bank_customer_churn` | 0.870645 | 0.870505 | 0.872547 | 0.858892 | TabPFN |
| `bank_marketing` | 0.761346 | 0.761253 | 0.763381 | 0.757223 | TabPFN |
| `bioresponse` | 0.849326 | 0.849787 | 0.866898 | 0.829553 | TabPFN |
| `diabetes` | 0.845646 | 0.844838 | 0.847023 | 0.825420 | TabPFN |
| `maternal_health_risk` | 0.932357 | 0.932527 | 0.931446 | 0.820203 | GraphDrone Prior Flat |
| `students_dropout_and_academic_success` | 0.894402 | 0.895620 | 0.898157 | 0.879676 | TabPFN |
| `website_phishing` | 0.978803 | 0.979390 | 0.980122 | 0.901367 | TabPFN |

## Decision

The prior-aware classification branch should be treated as the current `GraphDrone` classification mainline because it improves the baseline `GraphDrone` on most portfolio metrics.

But the correct public claim remains narrower than “best overall”:

- `GraphDrone` improved materially over its own baseline
- `GraphDrone` remains clearly stronger than `TabM`
- `TabPFN` is still the strongest overall public baseline on this 9-dataset classification portfolio
