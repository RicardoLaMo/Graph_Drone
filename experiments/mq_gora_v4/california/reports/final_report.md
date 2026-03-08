# Final Report: california
*2026-03-07*

## Executive Summary
Best California v4 variant: `CA_v4d` with RMSE `1.2790`.
Saved v3 G2 reference RMSE: `0.4546`; saved bad rich-context band best: `0.5099`.
Interpretation stays skeptical: the California question is whether regression-safe routing recovers toward G2, not whether extra components merely move numbers around.

## Integrity Confirmation
| model   | metric   |   current |   reference |    delta | status   |
|:--------|:---------|----------:|------------:|---------:|:---------|
| B1_HGBR | rmse     |  0.628481 |    0.443292 | 0.185189 | DRIFT    |
| G2_ref  | rmse     |  2.31796  |    0.454643 | 1.86331  | DRIFT    |
| G10_ref | rmse     |  2.32308  |    0.520906 | 1.80218  | DRIFT    |

## What Changed From v3
- California v4 removes raw label-context dependence from the primary baseline path.
- Any label context that remains is normalised and train-masked.
- Teacher-lite drops centroid loss and uses longer training.
- v4 now exposes explicit beta mode routing rather than only pi/tau.

## Root-Cause Audit
| root_cause                                               | status   | evidence                                                                                                                                               |
|:---------------------------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| RC1 — Router Information Starvation                      | PARTIAL  | CA_v4d: routing_entropy active=True, beta_std active=False                                                                                             |
| RC2 — No View-Discriminative Embedding                   | PASS     | view similarity off-diagonal min=-0.190                                                                                                                |
| RC3 — Meta-Learner Blind to Neighbourhood Content        | PASS     | beta regime span=0.047, dominant views=2                                                                                                               |
| RC4 — No Label / Local Predictiveness Context in Routing | PASS     | California v4 uses regression-safe track: no raw label-context variant is retained; label context is normalised and inference excludes label features. |

## Routing Behavior Audit
|   head_idx |   routing_entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |   mean_beta |   beta_std |      tau | model   |
|-----------:|------------------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|------------:|-----------:|---------:|:--------|
|          0 |          0.996781 | SOCIO           |       0.222661 |                0 |      0.236841 |               0 |        0.285187 |                 1 |          0.255311 |                   0 |    0.528642 | 0.00386217 | 1.00074  | CA_v4d  |
|          1 |          0.996951 | LOWRANK         |       0.24077  |                0 |      0.22386  |               0 |        0.249442 |                 0 |          0.285928 |                   1 |    0.496408 | 0.00418513 | 1.00105  | CA_v4d  |
|          2 |          0.994953 | SOCIO           |       0.21371  |                0 |      0.237171 |               0 |        0.294383 |                 1 |          0.254737 |                   0 |    0.52657  | 0.00377855 | 0.995686 | CA_v4d  |
|          3 |          0.990734 | SOCIO           |       0.199875 |                0 |      0.237416 |               0 |        0.310913 |                 1 |          0.251796 |                   0 |    0.483731 | 0.00506887 | 0.997849 | CA_v4d  |

## Gate Results
| gate                                        | status   | evidence                                                                             |
|:--------------------------------------------|:---------|:-------------------------------------------------------------------------------------|
| S1 — Integrity Confirmed                    | PARTIAL  | B1/G2/G10 current-vs-v3 comparison plus shared shape/interface checks                |
| C1 — California Training Health             | FAIL     | best CA_v4 stop epoch=4                                                              |
| C2 — California Regression-Safe Improvement | PARTIAL  | best CA_v4 RMSE=1.2790 vs best bad-v3=0.5099                                         |
| C3 — California Toward G2                   | FAIL     | best CA_v4 RMSE=1.2790 vs G2=0.4546                                                  |
| R1 — Rich Router Input Active               | PASS     | pi entropy and row-sensitive routing are non-degenerate                              |
| R2 — View-Discriminative Value Path         | PASS     | dominant views across heads=2                                                        |
| R3 — Mode Routing Active                    | FAIL     | beta std active=False                                                                |
| R4 — Complexity Justified                   | PARTIAL  | v4 complexity is only justified if it improves over the failing rich-context v3 runs |

## Predictive Results
| tag     |    rmse | best_ep   | stop_ep   | collapsed   | interp                                                           |
|:--------|--------:|:----------|:----------|:------------|:-----------------------------------------------------------------|
| G2_ref  | 2.31796 | v1        | v1        | False       | Reference GoRA without v4 split-track changes                    |
| G10_ref | 2.32308 | v3        | v3        | False       | Used for integrity/reference comparison, not as a v4 fix         |
| CA_v4a  | 1.82965 | 4         | 4         | True        | Regression-safe structural routing only                          |
| CA_v4b  | 1.50459 | 4         | 4         | True        | Experimental target-derived context, normalised and train-masked |
| CA_v4c  | 1.41697 | 4         | 4         | True        | Adds router-side scale control to label context                  |
| CA_v4d  | 1.27902 | 4         | 4         | True        | Teacher-lite without centroid loss                               |
| CA_v4e  | 1.39624 | 4         | 4         | True        | Longer patience with cosine schedule                             |

## Performance Drop Triage
| affected_model   |   delta_vs_G2 | primary_bucket          | evidence               | confidence   |
|:-----------------|--------------:|:------------------------|:-----------------------|:-------------|
| CA_v4a           |      1.37501  | TRAINING DYNAMICS ISSUE | rmse=1.8296, stop_ep=4 | medium       |
| CA_v4b           |      1.04994  | TRAINING DYNAMICS ISSUE | rmse=1.5046, stop_ep=4 | medium       |
| CA_v4c           |      0.962327 | TRAINING DYNAMICS ISSUE | rmse=1.4170, stop_ep=4 | medium       |
| CA_v4d           |      0.824377 | TRAINING DYNAMICS ISSUE | rmse=1.2790, stop_ep=4 | medium       |
| CA_v4e           |      0.941601 | TRAINING DYNAMICS ISSUE | rmse=1.3962, stop_ep=4 | medium       |

## What Is Fixed
- The branch now has explicit beta routing semantics and isolated California outputs.
- California label context is no longer used raw in the v4 path.
- v4 training restores the best checkpoint and validates teacher-query models with z_anc.

## What Remains
- California still needs to prove recovery toward or beyond G2 on a validated full run.
- Any gate marked PARTIAL/FAIL should be treated as unresolved before v5.

## Recommendation Before v5
| failed_gate                                 | observed_evidence                                                                    | likely_cause                 | minimal_next_fix                                                                       | priority   | should_fix_before_v5   |
|:--------------------------------------------|:-------------------------------------------------------------------------------------|:-----------------------------|:---------------------------------------------------------------------------------------|:-----------|:-----------------------|
| S1 — Integrity Confirmed                    | B1/G2/G10 current-vs-v3 comparison plus shared shape/interface checks                | implementation/reporting gap | rerun full reference reproduction and investigate drift before architectural claims    | high       | yes                    |
| C1 — California Training Health             | best CA_v4 stop epoch=4                                                              | training dynamics issue      | increase patience / inspect routing collapse                                           | medium     | no                     |
| C2 — California Regression-Safe Improvement | best CA_v4 RMSE=1.2790 vs best bad-v3=0.5099                                         | training dynamics issue      | remove or further constrain regression label-context / teacher dependence              | high       | yes                    |
| C3 — California Toward G2                   | best CA_v4 RMSE=1.2790 vs G2=0.4546                                                  | training dynamics issue      | simplify California path toward CA_v4a/CA_v4c and avoid unnecessary teacher complexity | high       | yes                    |
| R3 — Mode Routing Active                    | beta std active=False                                                                | implementation/reporting gap | strengthen or simplify beta gate so it varies meaningfully by regime                   | high       | yes                    |
| R4 — Complexity Justified                   | v4 complexity is only justified if it improves over the failing rich-context v3 runs | implementation/reporting gap | increase patience / inspect routing collapse                                           | medium     | no                     |

## Final Verdict
v4 changed architecture but did not solve the main problems
