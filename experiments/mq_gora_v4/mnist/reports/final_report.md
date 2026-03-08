# Final Report: mnist
*2026-03-08*

## Executive Summary
Run mode: `full`.
Best MNIST v4 variant: `MN_v4a` with accuracy `0.9333`.
Saved v3 G10 reference accuracy: `0.9380`; saved G2 accuracy: `0.9300`.
MNIST remains a protection track: changes are only justified if specialization improves without materially losing the G10 accuracy gain.

## Integrity Confirmation
| model   | metric   |   current |   reference |        delta | status   |
|:--------|:---------|----------:|------------:|-------------:|:---------|
| B1_HGBR | accuracy |  0.957333 |       0.958 | -0.000666667 | MATCH    |
| G2_ref  | accuracy |  0.929333 |       0.93  | -0.000666667 | MATCH    |
| G10_ref | accuracy |  0.932    |       0.938 | -0.006       | MATCH    |

## What Changed From v3
- G10 is rerun explicitly as a protected reference.
- v4 exposes explicit beta routing while preserving alpha-gate behavior.
- Diversity regularisation is treated as optional anti-collapse pressure, not assumed improvement.

## Root-Cause Audit
| root_cause                                               | status   | evidence                                                                                                                |
|:---------------------------------------------------------|:---------|:------------------------------------------------------------------------------------------------------------------------|
| RC1 — Router Information Starvation                      | PASS     | MN_v4a: routing_entropy active=True, beta active=True                                                                   |
| RC2 — No View-Discriminative Embedding                   | PASS     | view similarity off-diagonal min=0.553                                                                                  |
| RC3 — Meta-Learner Blind to Neighbourhood Content        | PASS     | beta regime span=0.079, dominant views=1                                                                                |
| RC4 — No Label / Local Predictiveness Context in Routing | PASS     | MNIST retains controlled label-context and teacher signal with alpha-gate protection rather than removing them blindly. |

## Routing Behavior Audit
|   head_idx |   routing_entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |   mean_beta |   beta_std |      tau | model   |
|-----------:|------------------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|------------:|-----------:|---------:|:--------|
|          0 |          0.718449 | PCA             |       0.243761 |       0.0273333  |        0.105503 |         0.0646667 |      0.650736 |        0.908    |    0.761093 |   0.156392 | 1.02031  | MN_v4a  |
|          1 |          0.819039 | PCA             |       0.318031 |       0.0373333  |        0.13192  |         0.0713333 |      0.550048 |        0.891333 |    0.749188 |   0.123011 | 1.00225  | MN_v4a  |
|          2 |          0.78508  | PCA             |       0.225712 |       0.00666667 |        0.151008 |         0.0653333 |      0.62328  |        0.928    |    0.752334 |   0.152326 | 0.976908 | MN_v4a  |
|          3 |          0.920845 | PCA             |       0.340411 |       0.094      |        0.19026  |         0.0786667 |      0.469329 |        0.827333 |    0.795516 |   0.169151 | 1.00238  | MN_v4a  |

## Gate Results
| gate                                | status   | evidence                                                                                                 |
|:------------------------------------|:---------|:---------------------------------------------------------------------------------------------------------|
| S1 — Integrity Confirmed            | PASS     | B1/G2/G10 reference comparison under current code path                                                   |
| M1 — MNIST Gain Retention           | PARTIAL  | best MN_v4 accuracy=0.9333 vs saved G10=0.9380                                                           |
| M2 — MNIST Routing Quality          | PARTIAL  | dominant views across heads=1                                                                            |
| R1 — Rich Router Input Active       | PASS     | pi varies across heads and rows                                                                          |
| R2 — View-Discriminative Value Path | PARTIAL  | dominant views across heads=1                                                                            |
| R3 — Mode Routing Active            | PASS     | beta std active=True                                                                                     |
| R4 — Complexity Justified           | PASS     | Extra routing complexity is justified only if MNIST gains remain above G2 while specialization improves. |

## Predictive Results
| tag     | source      | best_ep   | stop_ep   | collapsed   | change                                     | interp                                                                 |   accuracy |   macro_f1 |   log_loss |
|:--------|:------------|:----------|:----------|:------------|:-------------------------------------------|:-----------------------------------------------------------------------|-----------:|-----------:|-----------:|
| G2_ref  | current_run | v1        | v1        | False       | GoRA carry-forward reference               | Reference GoRA path before v4 split-track changes                      |   0.929333 |   0.928317 |   0.272256 |
| G10_ref | current_run | v3        | v3        | False       | Exact v3 G10 reproduction                  | Protected baseline before any MNIST routing changes                    |   0.932    |   0.931026 |   0.25742  |
| MN_v4a  | current_run | 7         | 27        | True        | Protected v4 path matching G10 ingredients | Locked alpha-gate baseline under v4 routing path                       |   0.933333 |   0.932358 |   0.249675 |
| MN_v4b  | current_run | 7         | 27        | True        | MN_v4a + mild diversity regulariser        | Incremental specialization pressure with alpha gate preserved          |   0.932    |   0.931206 |   0.247637 |
| MN_v4c  | current_run | 5         | 25        | True        | MN_v4b + stronger diversity                | More explicit anti-collapse pressure; only justified if accuracy holds |   0.926    |   0.92509  |   0.262275 |
| MN_v4d  | current_run | 6         | 26        | True        | MN_v4c + label-ctx LayerNorm               | Optional anti-collapse variant with extra scale control                |   0.932    |   0.93102  |   0.236551 |

## Performance Drop Triage
| affected_model   |   delta_vs_G10 | primary_bucket          | evidence                    | confidence   |
|:-----------------|---------------:|:------------------------|:----------------------------|:-------------|
| MN_v4a           |     0.00466667 | TRAINING DYNAMICS ISSUE | accuracy=0.9333, stop_ep=27 | medium       |
| MN_v4b           |     0.006      | TRAINING DYNAMICS ISSUE | accuracy=0.9320, stop_ep=27 | medium       |
| MN_v4c           |     0.012      | TRAINING DYNAMICS ISSUE | accuracy=0.9260, stop_ep=25 | medium       |
| MN_v4d           |     0.006      | TRAINING DYNAMICS ISSUE | accuracy=0.9320, stop_ep=26 | medium       |

## What Is Fixed
- The branch now preserves a protected G10 reproduction path.
- Explicit beta routing and isolated MNIST outputs are implemented.
- Best-checkpoint restore and teacher-query validation path mismatches are fixed in v4 training.

## What Remains
- Any MNIST accuracy loss versus G10 should block broader architectural changes from being treated as progress.
- Diversity pressure should be removed before v5 if it does not improve specialization without hurting metrics.

## Recommendation Before v5
| failed_gate                         | observed_evidence                              | likely_cause              | minimal_next_fix                                                                    | priority   | should_fix_before_v5   |
|:------------------------------------|:-----------------------------------------------|:--------------------------|:------------------------------------------------------------------------------------|:-----------|:-----------------------|
| M1 — MNIST Gain Retention           | best MN_v4 accuracy=0.9333 vs saved G10=0.9380 | training dynamics issue   | revert to exact G10 semantics for any change that hurts accuracy materially         | high       | yes                    |
| M2 — MNIST Routing Quality          | dominant views across heads=1                  | training dynamics issue   | use the diversity regulariser only if specialization improves without accuracy loss | high       | yes                    |
| R2 — View-Discriminative Value Path | dominant views across heads=1                  | model complexity mismatch | inspect protected alpha-gate path before adding more routing complexity             | medium     | no                     |

## Final Verdict
v4 partially improved but key routing issues remain
