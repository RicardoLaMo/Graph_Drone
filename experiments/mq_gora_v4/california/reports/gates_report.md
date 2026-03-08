# Gates Report: california
*2026-03-07*

| gate                                        | status   | evidence                                                                             |
|:--------------------------------------------|:---------|:-------------------------------------------------------------------------------------|
| S1 — Integrity Confirmed                    | PARTIAL  | Smoke mode uses provisional reference comparison plus shared shape/interface checks  |
| C1 — California Training Health             | FAIL     | best CA_v4 stop epoch=4                                                              |
| C2 — California Regression-Safe Improvement | PARTIAL  | best CA_v4 RMSE=1.2790 vs best bad-v3=0.5099                                         |
| C3 — California Toward G2                   | FAIL     | best CA_v4 RMSE=1.2790 vs G2=0.4546                                                  |
| R1 — Rich Router Input Active               | PASS     | pi entropy and row-sensitive routing are non-degenerate                              |
| R2 — View-Discriminative Value Path         | PASS     | dominant views across heads=2                                                        |
| R3 — Mode Routing Active                    | FAIL     | beta std active=False                                                                |
| R4 — Complexity Justified                   | PARTIAL  | v4 complexity is only justified if it improves over the failing rich-context v3 runs |

## Failure-to-Fix Mapping

| failed_gate                                 | observed_evidence                                                                    | likely_cause                 | minimal_next_fix                                                                       | priority   | should_fix_before_v5   |
|:--------------------------------------------|:-------------------------------------------------------------------------------------|:-----------------------------|:---------------------------------------------------------------------------------------|:-----------|:-----------------------|
| S1 — Integrity Confirmed                    | Smoke mode uses provisional reference comparison plus shared shape/interface checks  | implementation/reporting gap | rerun full reference reproduction and investigate drift before architectural claims    | high       | yes                    |
| C1 — California Training Health             | best CA_v4 stop epoch=4                                                              | training dynamics issue      | increase patience / inspect routing collapse                                           | medium     | no                     |
| C2 — California Regression-Safe Improvement | best CA_v4 RMSE=1.2790 vs best bad-v3=0.5099                                         | training dynamics issue      | remove or further constrain regression label-context / teacher dependence              | high       | yes                    |
| C3 — California Toward G2                   | best CA_v4 RMSE=1.2790 vs G2=0.4546                                                  | training dynamics issue      | simplify California path toward CA_v4a/CA_v4c and avoid unnecessary teacher complexity | high       | yes                    |
| R3 — Mode Routing Active                    | beta std active=False                                                                | implementation/reporting gap | strengthen or simplify beta gate so it varies meaningfully by regime                   | high       | yes                    |
| R4 — Complexity Justified                   | v4 complexity is only justified if it improves over the failing rich-context v3 runs | implementation/reporting gap | increase patience / inspect routing collapse                                           | medium     | no                     |