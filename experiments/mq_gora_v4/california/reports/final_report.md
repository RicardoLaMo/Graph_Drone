# Final Report: california
*2026-03-08*

## Executive Summary
Run mode: `full`.
Best California v4 variant: `CA_v4a` with RMSE `0.4719`.
Saved v3 G2 reference RMSE: `0.4546`; saved bad rich-context band best: `0.5099`.
Interpretation stays skeptical: the California question is whether regression-safe routing recovers toward G2, not whether extra components merely move numbers around.

## Integrity Confirmation
| model   | metric   |   current |   reference |      delta | status   |
|:--------|:---------|----------:|------------:|-----------:|:---------|
| B1_HGBR | rmse     |  0.443292 |    0.443292 |  0         | MATCH    |
| G2_ref  | rmse     |  0.451459 |    0.454643 | -0.0031838 | MATCH    |
| G10_ref | rmse     |  0.502292 |    0.520906 | -0.018614  | DRIFT    |

## What Changed From v3
- California v4 removes raw label-context dependence from the primary baseline path.
- Any label context that remains is normalised and train-masked.
- Teacher-lite drops centroid loss and uses longer training.
- v4 now exposes explicit beta mode routing rather than only pi/tau.

## Root-Cause Audit
| root_cause                                               | status   | evidence                                                                                                                                               |
|:---------------------------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| RC1 — Router Information Starvation                      | PASS     | CA_v4a: routing_entropy active=True, beta_std active=True                                                                                              |
| RC2 — No View-Discriminative Embedding                   | PASS     | view similarity off-diagonal min=0.298                                                                                                                 |
| RC3 — Meta-Learner Blind to Neighbourhood Content        | PASS     | beta regime span=0.194, dominant views=1                                                                                                               |
| RC4 — No Label / Local Predictiveness Context in Routing | PASS     | California v4 uses regression-safe track: no raw label-context variant is retained; label context is normalised and inference excludes label features. |

## Routing Behavior Audit
|   head_idx |   routing_entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |   mean_beta |   beta_std |      tau | model   |
|-----------:|------------------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|------------:|-----------:|---------:|:--------|
|          0 |          0.45469  | GEO             |      0.103055  |        0.0132429 |      0.778176 |        0.958979 |       0.0669699 |        0.0174419  |         0.0517993 |          0.0103359  |    0.283176 |  0.125891  | 0.902838 | CA_v4a  |
|          1 |          0.600129 | GEO             |      0.129308  |        0.0775194 |      0.67378  |        0.876292 |       0.0659346 |        0.0113049  |         0.130978  |          0.0348837  |    0.171816 |  0.119574  | 0.874031 | CA_v4a  |
|          2 |          0.786008 | GEO             |      0.250136  |        0.174742  |      0.410129 |        0.619832 |       0.0755714 |        0.00161499 |         0.264164  |          0.203811   |    0.105597 |  0.0885097 | 1.03295  | CA_v4a  |
|          3 |          0.46271  | GEO             |      0.0913863 |        0.0555556 |      0.765431 |        0.916021 |       0.0727616 |        0.0261628  |         0.0704216 |          0.00226098 |    0.155677 |  0.110113  | 0.996052 | CA_v4a  |

## Gate Results
| gate                                        | status   | evidence                                                                             |
|:--------------------------------------------|:---------|:-------------------------------------------------------------------------------------|
| S1 — Integrity Confirmed                    | PARTIAL  | B1/G2/G10 current-vs-v3 comparison plus shared shape/interface checks                |
| C1 — California Training Health             | PASS     | best CA_v4 stop epoch=99                                                             |
| C2 — California Regression-Safe Improvement | PASS     | best CA_v4 RMSE=0.4719 vs best bad-v3=0.5099                                         |
| C3 — California Toward G2                   | PARTIAL  | best CA_v4 RMSE=0.4719 vs G2=0.4546                                                  |
| R1 — Rich Router Input Active               | PASS     | pi entropy and row-sensitive routing are non-degenerate                              |
| R2 — View-Discriminative Value Path         | PARTIAL  | dominant views across heads=1                                                        |
| R3 — Mode Routing Active                    | PASS     | beta std active=True                                                                 |
| R4 — Complexity Justified                   | PASS     | v4 complexity is only justified if it improves over the failing rich-context v3 runs |

## Predictive Results
| tag     |     rmse | best_ep   | stop_ep   | collapsed   | interp                                                           |
|:--------|---------:|:----------|:----------|:------------|:-----------------------------------------------------------------|
| G2_ref  | 0.451459 | v1        | v1        | False       | Reference GoRA without v4 split-track changes                    |
| G10_ref | 0.502292 | v3        | v3        | False       | Used for integrity/reference comparison, not as a v4 fix         |
| CA_v4a  | 0.47187  | 34        | 74        | False       | Regression-safe structural routing only                          |
| CA_v4b  | 0.569631 | 39        | 79        | False       | Experimental target-derived context, normalised and train-masked |
| CA_v4c  | 0.532829 | 27        | 67        | False       | Adds router-side scale control to label context                  |
| CA_v4d  | 0.541853 | 43        | 83        | False       | Teacher-lite without centroid loss                               |
| CA_v4e  | 0.531636 | 59        | 99        | False       | Longer patience with cosine schedule                             |

## Performance Drop Triage
| affected_model   |   delta_vs_G2 | primary_bucket          | evidence                | confidence   |
|:-----------------|--------------:|:------------------------|:------------------------|:-------------|
| CA_v4a           |     0.0172273 | TRUE MODEL DESIGN ISSUE | rmse=0.4719, stop_ep=74 | low          |
| CA_v4b           |     0.114988  | TRUE MODEL DESIGN ISSUE | rmse=0.5696, stop_ep=79 | low          |
| CA_v4c           |     0.078186  | TRUE MODEL DESIGN ISSUE | rmse=0.5328, stop_ep=67 | low          |
| CA_v4d           |     0.0872102 | TRUE MODEL DESIGN ISSUE | rmse=0.5419, stop_ep=83 | low          |
| CA_v4e           |     0.076993  | TRUE MODEL DESIGN ISSUE | rmse=0.5316, stop_ep=99 | low          |

## What Is Fixed
- The branch now has explicit beta routing semantics and isolated California outputs.
- California label context is no longer used raw in the v4 path.
- v4 training restores the best checkpoint and validates teacher-query models with z_anc.

## What Remains
- California still needs to prove recovery toward or beyond G2 on a validated full run.
- Any gate marked PARTIAL/FAIL should be treated as unresolved before v5.

## Recommendation Before v5
| failed_gate                         | observed_evidence                                                     | likely_cause                 | minimal_next_fix                                                                       | priority   | should_fix_before_v5   |
|:------------------------------------|:----------------------------------------------------------------------|:-----------------------------|:---------------------------------------------------------------------------------------|:-----------|:-----------------------|
| S1 — Integrity Confirmed            | B1/G2/G10 current-vs-v3 comparison plus shared shape/interface checks | implementation/reporting gap | rerun full reference reproduction and investigate drift before architectural claims    | high       | yes                    |
| C3 — California Toward G2           | best CA_v4 RMSE=0.4719 vs G2=0.4546                                   | training dynamics issue      | simplify California path toward CA_v4a/CA_v4c and avoid unnecessary teacher complexity | high       | yes                    |
| R2 — View-Discriminative Value Path | dominant views across heads=1                                         | implementation/reporting gap | increase patience / inspect routing collapse                                           | medium     | no                     |

## Final Verdict
v4 partially improved but key routing issues remain
