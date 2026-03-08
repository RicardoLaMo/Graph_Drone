# Root Cause Audit
*2026-03-07*

Dataset: `mnist`
PASS / PARTIAL / FAIL reflects the v3 root-cause audit requirements.

| root_cause                                               | status   | evidence                                                                                                                |
|:---------------------------------------------------------|:---------|:------------------------------------------------------------------------------------------------------------------------|
| RC1 — Router Information Starvation                      | PASS     | MN_v4a: routing_entropy active=True, beta active=True                                                                   |
| RC2 — No View-Discriminative Embedding                   | PASS     | view similarity off-diagonal min=0.450                                                                                  |
| RC3 — Meta-Learner Blind to Neighbourhood Content        | PASS     | beta regime span=0.069, dominant views=2                                                                                |
| RC4 — No Label / Local Predictiveness Context in Routing | PASS     | MNIST retains controlled label-context and teacher signal with alpha-gate protection rather than removing them blindly. |