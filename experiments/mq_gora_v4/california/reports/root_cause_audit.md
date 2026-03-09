# Root Cause Audit
*2026-03-08*

Dataset: `california`
PASS / PARTIAL / FAIL reflects the v3 root-cause audit requirements.

| root_cause                                               | status   | evidence                                                                                                                                               |
|:---------------------------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| RC1 — Router Information Starvation                      | PASS     | CA_v4a: routing_entropy active=True, beta_std active=True                                                                                              |
| RC2 — No View-Discriminative Embedding                   | PASS     | view similarity off-diagonal min=0.298                                                                                                                 |
| RC3 — Meta-Learner Blind to Neighbourhood Content        | PASS     | beta regime span=0.194, dominant views=1                                                                                                               |
| RC4 — No Label / Local Predictiveness Context in Routing | PASS     | California v4 uses regression-safe track: no raw label-context variant is retained; label context is normalised and inference excludes label features. |