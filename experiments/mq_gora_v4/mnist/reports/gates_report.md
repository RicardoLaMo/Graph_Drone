# Gates Report: mnist
*2026-03-07*

| gate                                | status   | evidence                                                                                                 |
|:------------------------------------|:---------|:---------------------------------------------------------------------------------------------------------|
| S1 — Integrity Confirmed            | PARTIAL  | B1/G2/G10 reference comparison under current code path                                                   |
| M1 — MNIST Gain Retention           | FAIL     | best MN_v4 accuracy=0.8000 vs saved G10=0.9380                                                           |
| M2 — MNIST Routing Quality          | PARTIAL  | dominant views across heads=2                                                                            |
| R1 — Rich Router Input Active       | PASS     | pi varies across heads and rows                                                                          |
| R2 — View-Discriminative Value Path | PASS     | dominant views across heads=2                                                                            |
| R3 — Mode Routing Active            | PASS     | beta std active=True                                                                                     |
| R4 — Complexity Justified           | PARTIAL  | Extra routing complexity is justified only if MNIST gains remain above G2 while specialization improves. |

## Failure-to-Fix Mapping

| failed_gate                | observed_evidence                                                                                        | likely_cause              | minimal_next_fix                                                                    | priority   | should_fix_before_v5   |
|:---------------------------|:---------------------------------------------------------------------------------------------------------|:--------------------------|:------------------------------------------------------------------------------------|:-----------|:-----------------------|
| S1 — Integrity Confirmed   | B1/G2/G10 reference comparison under current code path                                                   | model complexity mismatch | inspect protected alpha-gate path before adding more routing complexity             | medium     | no                     |
| M1 — MNIST Gain Retention  | best MN_v4 accuracy=0.8000 vs saved G10=0.9380                                                           | training dynamics issue   | revert to exact G10 semantics for any change that hurts accuracy materially         | high       | yes                    |
| M2 — MNIST Routing Quality | dominant views across heads=2                                                                            | training dynamics issue   | use the diversity regulariser only if specialization improves without accuracy loss | high       | yes                    |
| R4 — Complexity Justified  | Extra routing complexity is justified only if MNIST gains remain above G2 while specialization improves. | model complexity mismatch | inspect protected alpha-gate path before adding more routing complexity             | medium     | no                     |