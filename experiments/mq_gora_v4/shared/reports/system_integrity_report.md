# MQ-GoRA v4: System Integrity Report
*Branch: `feature/mq-gora-v4-split-track`*

## Verdict: ALL CHECKS PASS ✅

## Step 0 Self-Alignment
1. The confirmed kwargs and vectorisation fixes are already verified.
2. Those fixes do not explain current v3 regression differences because the recorded values did not drift.
3. Routing here means observer-driven view trust plus explicit isolation-vs-interaction control.
4. Routing is not post-hoc weighted ensembling or raw geometry appended to prediction features.
5. California and MNIST stay split because regression-safe and classification-friendly signals differ.
6. The v4 objective is integrity first, then architecture evaluation.

I confirm v4 will be evaluated under split-track logic.
I confirm geometry signals are routing priors, not appended prediction features.
I confirm known bug fixes are numerically invariant and will not be used as a false explanation for v3 model weakness.

## A. Interface Compatibility

| model                 | accepts_view_mask   | accepts_z_anc   | accepts_lbl_nei   | accepts_agree_score   | accepts_extra_kwargs   | status   | note        |
|:----------------------|:--------------------|:----------------|:------------------|:----------------------|:-----------------------|:---------|:------------|
| GoraTransformer       | Y                   | Y               | Y                 | Y                     | Y                      | PASS     | tuple_len=3 |
| StandardTransformer   | Y                   | Y               | Y                 | Y                     | Y                      | PASS     | tuple_len=3 |
| SingleViewTransformer | Y                   | Y               | Y                 | Y                     | Y                      | PASS     | tuple_len=3 |
| MQGoraV4              | Y                   | Y               | Y                 | Y                     | Y                      | PASS     | tuple_len=6 |

## B. Precompute Timing

| fn                         | shape        | has_nan   |   time_s | status   |
|:---------------------------|:-------------|:----------|---------:|:---------|
| compute_label_ctx_per_view | (500, 15, 3) | False     |   0.0001 | PASS     |
| normalise_lbl_nei          | (500, 15, 3) | False     |   0      | PASS     |

## C. Shape / Value Sanity

| model    | status   | note                                      |
|:---------|:---------|:------------------------------------------|
| MQGoraV4 | PASS     | pred=(8, 1) pi_std=0.0156 beta_std=0.0222 |

## D. Reference Reproduction

At report-write time the known bug fixes are numerically invariant. Reference reruns should be compared against the local v3 metrics CSVs; if they differ materially, treat that as a new path mismatch instead of blaming the old fixed bugs.

| model      | metric        | current              | reference           | delta   | status   |
|:-----------|:--------------|:---------------------|:--------------------|:--------|:---------|
| B1_HGBR    | rmse/accuracy | not-run-in-integrity | see v3 metrics CSVs | n/a     | PENDING  |
| G2_GoRA_v1 | rmse/accuracy | not-run-in-integrity | see v3 metrics CSVs | n/a     | PENDING  |
| G10_Full   | rmse/accuracy | not-run-in-integrity | see v3 metrics CSVs | n/a     | PENDING  |

## Conclusion
Interface compatibility, precompute shape/timing checks, and routing-shape sanity must pass before model-design conclusions are trusted.