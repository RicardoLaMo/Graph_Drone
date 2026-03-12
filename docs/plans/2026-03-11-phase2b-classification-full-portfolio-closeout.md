# 2026-03-11 Phase 2B Classification Full Portfolio Closeout

## Scope

Closed the OpenML classification portfolio on the public benchmark surface:

- `GraphDrone`
- `TabPFN`
- `TabM`

`TabR` was removed from the public classification comparison. The runner still exists for diagnostics, but it is no longer part of the default suite or default summary.

## Portfolio

OpenML classification portfolio:

1. `diabetes`
2. `anneal`
3. `maternal_health_risk`
4. `website_phishing`
5. `bioresponse`
6. `students_dropout_and_academic_success`
7. `bank_customer_churn`
8. `bank_marketing`
9. `apsfailure`

Each public model was evaluated on folds `0/1/2`.

## Recovery Notes

The first full suite surfaced three real benchmark-layer issues:

1. OpenML mixed-type repair was required for `bank_marketing`.
2. OpenML cache recovery was required for `apsfailure`.
3. `TabR` classification remained wrapper-fragile on several categorical datasets, so it was removed from the public classification surface instead of continuing to spend iteration budget on that path.

## Final Summary

Final public summary is in:

- `experiments/openml_classification_benchmark/reports_phase2b_full_portfolio/openml_benchmark_summary.md`
- `experiments/openml_classification_benchmark/reports_phase2b_full_portfolio/openml_benchmark_summary.csv`

High-level read:

- `GraphDrone` is competitive across the full 9-dataset classification portfolio.
- `GraphDrone` beats `TabM` broadly.
- `GraphDrone` is close to or better than `TabPFN` on several datasets, but does not dominate the portfolio.

## Code Surface Changed

- `experiments/openml_classification_benchmark/src/openml_tasks.py`
- `experiments/openml_classification_benchmark/src/foundation_config.py`
- `experiments/openml_classification_benchmark/scripts/run_tabr_openml.py`
- `experiments/openml_classification_benchmark/scripts/run_tabm_openml.py`
- `experiments/openml_classification_benchmark/scripts/run_tabpfn_openml.py`
- `experiments/openml_classification_benchmark/scripts/run_graphdrone_fit_openml.py`
- `experiments/openml_classification_benchmark/scripts/run_openml_suite.py`
- `experiments/openml_classification_benchmark/scripts/summarize_openml_suite.py`
- `experiments/tab_foundation_compare/scripts/run_tabr_aligned.py`

## Decision

Classification Phase 2B is complete enough to secure in git and push.
