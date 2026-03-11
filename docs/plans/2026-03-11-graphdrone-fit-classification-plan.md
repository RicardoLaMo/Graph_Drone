# GraphDrone.fit Classification Plan

## Objective

Extend `GraphDrone.fit()` from regression-only benchmarking to a portfolio-level OpenML classification benchmark without drifting back to dataset-specific view benchmarking.

Public benchmark rows:
- `GraphDrone`
- `TabPFN`
- `TabR`
- `TabM`

Internal experts remain diagnostics-only.

## Classification Portfolio

The initial 9-dataset OpenML classification portfolio is:

1. `diabetes`
2. `anneal`
3. `maternal_health_risk`
4. `website_phishing`
5. `bioresponse`
6. `students_dropout_and_academic_success`
7. `bank_customer_churn`
8. `bank_marketing`
9. `apsfailure`

## Implementation Goals

1. Make classification first-class in `src/graphdrone_fit/`.
2. Add probability-aware metrics:
   - `ROC-AUC`
   - `PR-AUC`
   - `F1`
   - `accuracy`
   - `log_loss`
3. Keep the benchmark surface modular:
   - loader
   - adapter
   - public runner scripts
   - suite runner
   - summarizer
4. Compute public metrics consistently across all runners, including upstream baselines.

## Quality Gates

1. Targeted classification unit tests must pass.
2. `GraphDrone` binary smoke must pass.
3. `GraphDrone` multiclass smoke must pass.
4. `TabPFN`, `TabR`, and `TabM` classification wrappers must each pass at least one binary smoke.
5. Suite summarization must work on the shared H200 environment without optional extras such as `tabulate`.

## Agent Review

- Gemini is used as the external code review pass for implementation coherence and drift detection.
- Claude planning was attempted for this branch, but the local Claude CLI was rate-limited on this host during the planning window.
