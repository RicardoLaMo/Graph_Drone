# California v4 Summary
*2026-03-08*

## Metrics
| tag         | source      | best_ep   | stop_ep   | collapsed   | change                                    | interp                                                           |       rmse |        mae |         r2 |
|:------------|:------------|:----------|:----------|:------------|:------------------------------------------|:-----------------------------------------------------------------|-----------:|-----------:|-----------:|
| B1_HGBR     | current_run | —         | —         | False       | reference baseline                        | Strong tabular baseline for calibration                          |   0.443292 |   0.300658 |   0.85139  |
| G2_ref      | current_run | v1        | v1        | False       | v3/v1 carry-forward                       | Reference GoRA without v4 split-track changes                    |   0.451459 |   0.309318 |   0.845864 |
| G10_ref     | current_run | v3        | v3        | False       | Exact v3 rich-context reproduction        | Used for integrity/reference comparison, not as a v4 fix         |   0.502292 |   0.352318 |   0.8092   |
| CA_v4a      | current_run | 34        | 74        | False       | Remove LabelContextEncoder entirely       | Regression-safe structural routing only                          |   0.47187  |   0.325637 |   0.831612 |
| CA_v4b      | current_run | 39        | 79        | False       | CA_v4a + normalised label context         | Experimental target-derived context, normalised and train-masked |   0.569631 |   0.382897 |   0.754612 |
| CA_v4c      | current_run | 27        | 67        | False       | CA_v4b + LayerNorm(label_ctx_vec)         | Adds router-side scale control to label context                  |   0.532829 |   0.366378 |   0.785295 |
| CA_v4d      | current_run | 43        | 83        | False       | CA_v4c + teacher-lite (L_agree + L_label) | Teacher-lite without centroid loss                               |   0.541853 |   0.361263 |   0.777961 |
| CA_v4e      | current_run | 59        | 99        | False       | CA_v4d + healthier scheduler / patience   | Longer patience with cosine schedule                             |   0.531636 |   0.352731 |   0.786255 |
| B1_HGBR     | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.443292 |   0.300658 |   0.85139  |
| B2_TabPFN   | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             | nan        | nan        | nan        |
| G2_GoRA_v1  | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.454643 |   0.314812 |   0.843683 |
| G7_RichCtx  | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.49288  |   0.344011 |   0.816283 |
| G8_LabelCtx | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.516909 |   0.368382 |   0.797934 |
| G9_Teacher  | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.50986  |   0.35959  |   0.803407 |
| G10_Full    | saved_v3    | —         | —         | False       | saved reference                           | Historical v3 result                                             |   0.520906 |   0.36607  |   0.794797 |

## California Ablation Table
| tag     |     rmse | best_ep   | stop_ep   | collapsed   | interp                                                           |
|:--------|---------:|:----------|:----------|:------------|:-----------------------------------------------------------------|
| G2_ref  | 0.451459 | v1        | v1        | False       | Reference GoRA without v4 split-track changes                    |
| G10_ref | 0.502292 | v3        | v3        | False       | Used for integrity/reference comparison, not as a v4 fix         |
| CA_v4a  | 0.47187  | 34        | 74        | False       | Regression-safe structural routing only                          |
| CA_v4b  | 0.569631 | 39        | 79        | False       | Experimental target-derived context, normalised and train-masked |
| CA_v4c  | 0.532829 | 27        | 67        | False       | Adds router-side scale control to label context                  |
| CA_v4d  | 0.541853 | 43        | 83        | False       | Teacher-lite without centroid loss                               |
| CA_v4e  | 0.531636 | 59        | 99        | False       | Longer patience with cosine schedule                             |

## Reference Reproduction
| model   | metric   |   current |   reference |      delta | status   |
|:--------|:---------|----------:|------------:|-----------:|:---------|
| B1_HGBR | rmse     |  0.443292 |    0.443292 |  0         | MATCH    |
| G2_ref  | rmse     |  0.451459 |    0.454643 | -0.0031838 | MATCH    |
| G10_ref | rmse     |  0.502292 |    0.520906 | -0.018614  | DRIFT    |