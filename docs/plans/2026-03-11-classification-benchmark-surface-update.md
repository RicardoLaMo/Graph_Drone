# 2026-03-11 Classification Benchmark Surface Update

## Decision

Drop `TabR` from the public OpenML classification benchmark surface.

## Reason

The classification lane is meant to compare `GraphDrone.fit()` against stable public baselines, not spend iteration budget on wrapper-specific categorical compatibility work. `TabR` remained fragile on several classification datasets because its upstream categorical path is sensitive to:

- categorical-only datasets,
- unseen validation/test categories,
- environment-specific sklearn compatibility,
- and OpenML dataset formatting edge cases.

Those issues are benchmark-wrapper issues, not the current GraphDrone classification objective.

## Public Surface

The public classification benchmark now means:

- `GraphDrone`
- `TabPFN`
- `TabM`

`TabR` can still exist as an optional local runner for diagnostics, but it is no longer part of the default suite or default summary.

## Code Impact

- `experiments/openml_classification_benchmark/scripts/run_openml_suite.py`
  default public models no longer include `TabR`
- `experiments/openml_classification_benchmark/scripts/summarize_openml_suite.py`
  default summary filters to `GraphDrone`, `TabPFN`, and `TabM`

## Evaluation Impact

Historical `TabR` result files may remain on disk, but they are excluded from the rebuilt public classification leaderboard unless explicitly requested.
