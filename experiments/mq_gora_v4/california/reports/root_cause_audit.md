# Root Cause Audit
*2026-03-07*

Dataset: `california`
PASS / PARTIAL / FAIL reflects the v3 root-cause audit requirements.

| root_cause                                               | status   | evidence                                                                                                                                               |
|:---------------------------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| RC1 — Router Information Starvation                      | PARTIAL  | CA_v4d: routing_entropy active=True, beta_std active=False                                                                                             |
| RC2 — No View-Discriminative Embedding                   | PASS     | view similarity off-diagonal min=-0.190                                                                                                                |
| RC3 — Meta-Learner Blind to Neighbourhood Content        | PASS     | beta regime span=0.047, dominant views=2                                                                                                               |
| RC4 — No Label / Local Predictiveness Context in Routing | PASS     | California v4 uses regression-safe track: no raw label-context variant is retained; label context is normalised and inference excludes label features. |