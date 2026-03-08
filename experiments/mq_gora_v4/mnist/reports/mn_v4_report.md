# MNIST v4 Summary
*2026-03-08*

## Metrics
| tag         | source      | best_ep   | stop_ep   | collapsed   | change                                     | interp                                                                 |   accuracy |   macro_f1 |   log_loss |
|:------------|:------------|:----------|:----------|:------------|:-------------------------------------------|:-----------------------------------------------------------------------|-----------:|-----------:|-----------:|
| B1_HGBR     | current_run | —         | —         | False       | reference baseline                         | Strong tabular baseline for calibration                                |   0.957333 |   0.957061 | nan        |
| G2_ref      | current_run | v1        | v1        | False       | GoRA carry-forward reference               | Reference GoRA path before v4 split-track changes                      |   0.929333 |   0.928317 |   0.272256 |
| G10_ref     | current_run | v3        | v3        | False       | Exact v3 G10 reproduction                  | Protected baseline before any MNIST routing changes                    |   0.932    |   0.931026 |   0.25742  |
| MN_v4a      | current_run | 7         | 27        | True        | Protected v4 path matching G10 ingredients | Locked alpha-gate baseline under v4 routing path                       |   0.933333 |   0.932358 |   0.249675 |
| MN_v4b      | current_run | 7         | 27        | True        | MN_v4a + mild diversity regulariser        | Incremental specialization pressure with alpha gate preserved          |   0.932    |   0.931206 |   0.247637 |
| MN_v4c      | current_run | 5         | 25        | True        | MN_v4b + stronger diversity                | More explicit anti-collapse pressure; only justified if accuracy holds |   0.926    |   0.92509  |   0.262275 |
| MN_v4d      | current_run | 6         | 26        | True        | MN_v4c + label-ctx LayerNorm               | Optional anti-collapse variant with extra scale control                |   0.932    |   0.93102  |   0.236551 |
| B1_HGBR     | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.958    |   0.957679 |   0.188216 |
| B2_TabPFN   | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   | nan        | nan        | nan        |
| G2_GoRA_v1  | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.93     |   0.929386 |   0.258996 |
| G7_RichCtx  | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.926667 |   0.925792 |   0.275473 |
| G8_LabelCtx | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.930667 |   0.930003 |   0.272681 |
| G9_Teacher  | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.934667 |   0.934031 |   0.279914 |
| G10_Full    | saved_v3    | —         | —         | False       | saved reference                            | Historical v3 result                                                   |   0.938    |   0.937367 |   0.237162 |

## Reference Reproduction
| model   | metric   |   current |   reference |        delta | status   |
|:--------|:---------|----------:|------------:|-------------:|:---------|
| B1_HGBR | accuracy |  0.957333 |       0.958 | -0.000666667 | MATCH    |
| G2_ref  | accuracy |  0.929333 |       0.93  | -0.000666667 | MATCH    |
| G10_ref | accuracy |  0.932    |       0.938 | -0.006       | MATCH    |