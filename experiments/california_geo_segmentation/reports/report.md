# California Geo Segmentation Report

## Question
Can latitude/longitude-derived pseudo-neighborhood segments recover useful latent
community signal beyond raw continuous GEO features?

## Summary
Best model was `H2_all_geo_priors` at RMSE `0.4197`, improving over raw HGBR by `+0.0211`.

## Metrics
| model                        |     rmse |      mae |       r2 |   n_features | note                                                |
|:-----------------------------|---------:|---------:|---------:|-------------:|:----------------------------------------------------|
| H2_all_geo_priors            | 0.419664 | 0.277543 | 0.86681  |           26 | raw + all geo priors                                |
| H1_kmeans96_aug              | 0.423109 | 0.27705  | 0.864615 |           13 | raw + segment priors                                |
| H4_all_geo_target_stats_only | 0.42479  | 0.280477 | 0.863537 |           16 | raw + per-segment target mean/std only              |
| H5_all_geo_structure_only    | 0.425143 | 0.283719 | 0.86331  |           18 | raw + segment count/density/centroid structure only |
| H3_all_geo_priors_shuffled   | 0.426018 | 0.283331 | 0.862747 |           26 | raw + geo priors built from shuffled train targets  |
| H1_kmeans32_aug              | 0.42655  | 0.283483 | 0.862404 |           13 | raw + segment priors                                |
| H1_grid_coarse_aug           | 0.429077 | 0.285142 | 0.860768 |           12 | raw + segment priors                                |
| H1_grid_fine_aug             | 0.437708 | 0.291501 | 0.855111 |           12 | raw + segment priors                                |
| B1_HGBR_raw                  | 0.440782 | 0.297866 | 0.853068 |            8 | raw scaled features                                 |
| S0_kmeans96_mean             | 0.826919 | 0.595978 | 0.482879 |            1 | segment train mean only                             |
| S0_grid_fine_mean            | 0.828493 | 0.606161 | 0.480908 |            1 | segment train mean only                             |
| S0_grid_coarse_mean          | 0.877946 | 0.646558 | 0.41709  |            1 | segment train mean only                             |
| S0_kmeans32_mean             | 0.916316 | 0.678547 | 0.365024 |            1 | segment train mean only                             |

## Interpretation
- This experiment keeps the California split and preprocessing aligned with the v5/A6f line.
- Baseline `B1_HGBR_raw` already includes continuous latitude/longitude in the raw feature vector.
- Segment target priors are train-only. Validation/test rows receive statistics from train-fitted segments.
- `H3_all_geo_priors_shuffled` is a leakage control built from shuffled train targets only.
- Positive signal here justifies pushing geo segmentation into retrieval candidate pools or routing priors.
- Flat or negative signal suggests lat/lon continuous features already capture most of what these simple segmenters can recover.

## Segmentation note
Schemes tested: coarse/fine fixed grids over raw latitude-longitude and train-fitted KMeans(32/96) pseudo-communities.
