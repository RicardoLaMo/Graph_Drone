# Gates Report: california
*2026-03-08*

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

## Failure-to-Fix Mapping

| failed_gate                         | observed_evidence                                                     | likely_cause                 | minimal_next_fix                                                                       | priority   | should_fix_before_v5   |
|:------------------------------------|:----------------------------------------------------------------------|:-----------------------------|:---------------------------------------------------------------------------------------|:-----------|:-----------------------|
| S1 — Integrity Confirmed            | B1/G2/G10 current-vs-v3 comparison plus shared shape/interface checks | implementation/reporting gap | rerun full reference reproduction and investigate drift before architectural claims    | high       | yes                    |
| C3 — California Toward G2           | best CA_v4 RMSE=0.4719 vs G2=0.4546                                   | training dynamics issue      | simplify California path toward CA_v4a/CA_v4c and avoid unnecessary teacher complexity | high       | yes                    |
| R2 — View-Discriminative Value Path | dominant views across heads=1                                         | implementation/reporting gap | increase patience / inspect routing collapse                                           | medium     | no                     |