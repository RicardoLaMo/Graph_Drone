# MNIST v4 Summary
*2026-03-07*

## Metrics
| tag         | source      | best_ep   | stop_ep   | collapsed   | change                                     | interp                                                                 |   accuracy |   macro_f1 |   log_loss |
|:------------|:------------|:----------|:----------|:------------|:-------------------------------------------|:-----------------------------------------------------------------------|-----------:|-----------:|-----------:|
| B1_HGBR     | current_run | —         | —         | False       | reference baseline                         | Strong tabular baseline for calibration                                |   0.873333 |   0.867219 | nan        |
| G2_ref      | current_run | v1        | v1        | False       | GoRA carry-forward reference               | Reference GoRA path before v4 split-track changes                      |   0.493333 |   0.402972 |   1.99482  |
| G10_ref     | current_run | v3        | v3        | False       | Exact v3 G10 reproduction                  | Protected baseline before any MNIST routing changes                    |   0.64     |   0.583525 |   1.93551  |
| MN_v4a      | current_run | 4         | 4         | True        | Protected v4 path matching G10 ingredients | Locked alpha-gate baseline under v4 routing path                       |   0.8      |   0.787748 |   1.23271  |
| MN_v4b      | current_run | 4         | 4         | True        | MN_v4a + mild diversity regulariser        | Incremental specialization pressure with alpha gate preserved          |   0.8      |   0.787294 |   1.19812  |
| MN_v4c      | current_run | 4         | 4         | True        | MN_v4b + stronger diversity                | More explicit anti-collapse pressure; only justified if accuracy holds |   0.773333 |   0.76066  |   1.22387  |
| MN_v4d      | current_run | 4         | 4         | True        | MN_v4c + label-ctx LayerNorm               | Optional anti-collapse variant with extra scale control                |   0.773333 |   0.751873 |   1.19286  |
| B1_HGBR     | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.958    |   0.957679 |   0.188216 |
| B2_TabPFN   | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   | nan        | nan        | nan        |
| G2_GoRA_v1  | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.93     |   0.929386 |   0.258996 |
| G7_RichCtx  | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.926667 |   0.925792 |   0.275473 |
| G8_LabelCtx | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.930667 |   0.930003 |   0.272681 |
| G9_Teacher  | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.934667 |   0.934031 |   0.279914 |
| G10_Full    | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.938    |   0.937367 |   0.237162 |

## Reference Reproduction
| model   | metric   |   current |   reference |      delta | status     |
|:--------|:---------|----------:|------------:|-----------:|:-----------|
| B1_HGBR | accuracy |  0.873333 |       0.958 | -0.0846667 | SMOKE_ONLY |
| G2_ref  | accuracy |  0.493333 |       0.93  | -0.436667  | SMOKE_ONLY |
| G10_ref | accuracy |  0.64     |       0.938 | -0.298     | SMOKE_ONLY |