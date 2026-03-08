# California v4 Summary
*2026-03-07*

## Metrics
| tag         | source      | best_ep   | stop_ep   | collapsed   | change                                    | interp                                                           |       rmse |        mae |          r2 |
|:------------|:------------|:----------|:----------|:------------|:------------------------------------------|:-----------------------------------------------------------------|-----------:|-----------:|------------:|
| B1_HGBR     | current_run | —         | —         | False       | reference baseline                        | Strong tabular baseline for calibration                          |   0.628481 |   0.433651 |   0.756047  |
| G2_ref      | current_run | v1        | v1        | False       | v3/v1 carry-forward                       | Reference GoRA without v4 split-track changes                    |   2.31796  |   1.90432  |  -2.31842   |
| G10_ref     | current_run | v3        | v3        | False       | Exact v3 rich-context reproduction        | Used for integrity/reference comparison, not as a v4 fix         |   2.32308  |   1.9578   |  -2.33312   |
| CA_v4a      | current_run | 4         | 4         | True        | Remove LabelContextEncoder entirely       | Regression-safe structural routing only                          |   1.82965  |   1.37416  |  -1.06755   |
| CA_v4b      | current_run | 4         | 4         | True        | CA_v4a + normalised label context         | Experimental target-derived context, normalised and train-masked |   1.50459  |   1.08725  |  -0.398153  |
| CA_v4c      | current_run | 4         | 4         | True        | CA_v4b + LayerNorm(label_ctx_vec)         | Adds router-side scale control to label context                  |   1.41697  |   1.03455  |  -0.24006   |
| CA_v4d      | current_run | 4         | 4         | True        | CA_v4c + teacher-lite (L_agree + L_label) | Teacher-lite without centroid loss                               |   1.27902  |   0.932883 |  -0.0103582 |
| CA_v4e      | current_run | 4         | 4         | True        | CA_v4d + healthier scheduler / patience   | Longer patience with cosine schedule                             |   1.39624  |   1.0045   |  -0.204048  |
| B1_HGBR     | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.443292 |   0.300658 |   0.85139   |
| B2_TabPFN   | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             | nan        | nan        | nan         |
| G2_GoRA_v1  | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.454643 |   0.314812 |   0.843683  |
| G7_RichCtx  | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.49288  |   0.344011 |   0.816283  |
| G8_LabelCtx | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.516909 |   0.368382 |   0.797934  |
| G9_Teacher  | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.50986  |   0.35959  |   0.803407  |
| G10_Full    | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.520906 |   0.36607  |   0.794797  |

## California Ablation Table
| tag     |    rmse | best_ep   | stop_ep   | collapsed   | interp                                                           |
|:--------|--------:|:----------|:----------|:------------|:-----------------------------------------------------------------|
| G2_ref  | 2.31796 | v1        | v1        | False       | Reference GoRA without v4 split-track changes                    |
| G10_ref | 2.32308 | v3        | v3        | False       | Used for integrity/reference comparison, not as a v4 fix         |
| CA_v4a  | 1.82965 | 4         | 4         | True        | Regression-safe structural routing only                          |
| CA_v4b  | 1.50459 | 4         | 4         | True        | Experimental target-derived context, normalised and train-masked |
| CA_v4c  | 1.41697 | 4         | 4         | True        | Adds router-side scale control to label context                  |
| CA_v4d  | 1.27902 | 4         | 4         | True        | Teacher-lite without centroid loss                               |
| CA_v4e  | 1.39624 | 4         | 4         | True        | Longer patience with cosine schedule                             |

## Reference Reproduction
| model   | metric   |   current |   reference |    delta | status     |
|:--------|:---------|----------:|------------:|---------:|:-----------|
| B1_HGBR | rmse     |  0.628481 |    0.443292 | 0.185189 | SMOKE_ONLY |
| G2_ref  | rmse     |  2.31796  |    0.454643 | 1.86331  | SMOKE_ONLY |
| G10_ref | rmse     |  2.32308  |    0.520906 | 1.80218  | SMOKE_ONLY |