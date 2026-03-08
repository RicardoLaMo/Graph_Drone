# Final Report: mnist
*2026-03-07*

## Executive Summary
Run mode: `smoke`.
Best MNIST v4 variant: `MN_v4a` with accuracy `0.8000`.
Saved v3 G10 reference accuracy: `0.9380`; saved G2 accuracy: `0.9300`.
MNIST remains a protection track: changes are only justified if specialization improves without materially losing the G10 accuracy gain.

## Integrity Confirmation
| model   | metric   |   current |   reference |      delta | status     |
|:--------|:---------|----------:|------------:|-----------:|:-----------|
| B1_HGBR | accuracy |  0.873333 |       0.958 | -0.0846667 | SMOKE_ONLY |
| G2_ref  | accuracy |  0.493333 |       0.93  | -0.436667  | SMOKE_ONLY |
| G10_ref | accuracy |  0.64     |       0.938 | -0.298     | SMOKE_ONLY |

## What Changed From v3
- G10 is rerun explicitly as a protected reference.
- v4 exposes explicit beta routing while preserving alpha-gate behavior.
- Diversity regularisation is treated as optional anti-collapse pressure, not assumed improvement.

## Root-Cause Audit
| root_cause                                               | status   | evidence                                                                                                                |
|:---------------------------------------------------------|:---------|:------------------------------------------------------------------------------------------------------------------------|
| RC1 — Router Information Starvation                      | PASS     | MN_v4a: routing_entropy active=True, beta active=True                                                                   |
| RC2 — No View-Discriminative Embedding                   | PASS     | view similarity off-diagonal min=0.450                                                                                  |
| RC3 — Meta-Learner Blind to Neighbourhood Content        | PASS     | beta regime span=0.069, dominant views=2                                                                                |
| RC4 — No Label / Local Predictiveness Context in Routing | PASS     | MNIST retains controlled label-context and teacher signal with alpha-gate protection rather than removing them blindly. |

## Routing Behavior Audit
|   head_idx |   routing_entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |   mean_beta |   beta_std |      tau | model   |
|-----------:|------------------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|------------:|-----------:|---------:|:--------|
|          0 |          0.983795 | PCA             |       0.281159 |             0    |        0.302441 |          0        |      0.4164   |        1        |    0.545694 |  0.034537  | 1.00356  | MN_v4a  |
|          1 |          0.989197 | PCA             |       0.280234 |             0    |        0.322336 |          0.06     |      0.397429 |        0.94     |    0.57512  |  0.0217148 | 0.999538 | MN_v4a  |
|          2 |          0.994923 | PCA             |       0.29013  |             0    |        0.341742 |          0.133333 |      0.368129 |        0.866667 |    0.51283  |  0.0171603 | 0.998924 | MN_v4a  |
|          3 |          0.99743  | BLOCK           |       0.318004 |             0.18 |        0.361655 |          0.82     |      0.320342 |        0        |    0.534253 |  0.0353248 | 0.999365 | MN_v4a  |

## Gate Results
| gate                                | status   | evidence                                                                                                 |
|:------------------------------------|:---------|:---------------------------------------------------------------------------------------------------------|
| S1 — Integrity Confirmed            | PARTIAL  | Smoke mode uses provisional reference comparison under current code path                                 |
| M1 — MNIST Gain Retention           | FAIL     | best MN_v4 accuracy=0.8000 vs saved G10=0.9380                                                           |
| M2 — MNIST Routing Quality          | PARTIAL  | dominant views across heads=2                                                                            |
| R1 — Rich Router Input Active       | PASS     | pi varies across heads and rows                                                                          |
| R2 — View-Discriminative Value Path | PASS     | dominant views across heads=2                                                                            |
| R3 — Mode Routing Active            | PASS     | beta std active=True                                                                                     |
| R4 — Complexity Justified           | PARTIAL  | Extra routing complexity is justified only if MNIST gains remain above G2 while specialization improves. |

## Predictive Results
| tag     | source      | best_ep   | stop_ep   | collapsed   | change                                     | interp                                                                 |   accuracy |   macro_f1 |   log_loss |
|:--------|:------------|:----------|:----------|:------------|:-------------------------------------------|:-----------------------------------------------------------------------|-----------:|-----------:|-----------:|
| G2_ref  | current_run | v1        | v1        | False       | GoRA carry-forward reference               | Reference GoRA path before v4 split-track changes                      |   0.493333 |   0.402972 |    1.99482 |
| G10_ref | current_run | v3        | v3        | False       | Exact v3 G10 reproduction                  | Protected baseline before any MNIST routing changes                    |   0.64     |   0.583525 |    1.93551 |
| MN_v4a  | current_run | 4         | 4         | True        | Protected v4 path matching G10 ingredients | Locked alpha-gate baseline under v4 routing path                       |   0.8      |   0.787748 |    1.23271 |
| MN_v4b  | current_run | 4         | 4         | True        | MN_v4a + mild diversity regulariser        | Incremental specialization pressure with alpha gate preserved          |   0.8      |   0.787294 |    1.19812 |
| MN_v4c  | current_run | 4         | 4         | True        | MN_v4b + stronger diversity                | More explicit anti-collapse pressure; only justified if accuracy holds |   0.773333 |   0.76066  |    1.22387 |
| MN_v4d  | current_run | 4         | 4         | True        | MN_v4c + label-ctx LayerNorm               | Optional anti-collapse variant with extra scale control                |   0.773333 |   0.751873 |    1.19286 |

## Performance Drop Triage
| affected_model   |   delta_vs_G10 | primary_bucket          | evidence                   | confidence   |
|:-----------------|---------------:|:------------------------|:---------------------------|:-------------|
| MN_v4a           |       0.138    | TRAINING DYNAMICS ISSUE | accuracy=0.8000, stop_ep=4 | medium       |
| MN_v4b           |       0.138    | TRAINING DYNAMICS ISSUE | accuracy=0.8000, stop_ep=4 | medium       |
| MN_v4c           |       0.164667 | TRAINING DYNAMICS ISSUE | accuracy=0.7733, stop_ep=4 | medium       |
| MN_v4d           |       0.164667 | TRAINING DYNAMICS ISSUE | accuracy=0.7733, stop_ep=4 | medium       |

## What Is Fixed
- The branch now preserves a protected G10 reproduction path.
- Explicit beta routing and isolated MNIST outputs are implemented.
- Best-checkpoint restore and teacher-query validation path mismatches are fixed in v4 training.

## What Remains
- Any MNIST accuracy loss versus G10 should block broader architectural changes from being treated as progress.
- Diversity pressure should be removed before v5 if it does not improve specialization without hurting metrics.

## Recommendation Before v5
| failed_gate                | observed_evidence                                                                                        | likely_cause              | minimal_next_fix                                                                    | priority   | should_fix_before_v5   |
|:---------------------------|:---------------------------------------------------------------------------------------------------------|:--------------------------|:------------------------------------------------------------------------------------|:-----------|:-----------------------|
| S1 — Integrity Confirmed   | Smoke mode uses provisional reference comparison under current code path                                 | model complexity mismatch | inspect protected alpha-gate path before adding more routing complexity             | medium     | no                     |
| M1 — MNIST Gain Retention  | best MN_v4 accuracy=0.8000 vs saved G10=0.9380                                                           | training dynamics issue   | revert to exact G10 semantics for any change that hurts accuracy materially         | high       | yes                    |
| M2 — MNIST Routing Quality | dominant views across heads=2                                                                            | training dynamics issue   | use the diversity regulariser only if specialization improves without accuracy loss | high       | yes                    |
| R4 — Complexity Justified  | Extra routing complexity is justified only if MNIST gains remain above G2 while specialization improves. | model complexity mismatch | inspect protected alpha-gate path before adding more routing complexity             | medium     | no                     |

## Final Verdict
v4 changed architecture but did not solve the main problems
