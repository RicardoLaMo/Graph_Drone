# P0 Positioning — california_anchor_benchmark

This is a provisional TabArena-style leaderboard for the California anchor benchmark.
It uses mean RMSE, normalized score, and paired win counts when split-wise runs exist.

- target model: `P0_router`

| Rank | Model | Coverage | Runs | Mean Test RMSE | Std | Normalized Score |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | P0_crossfit | matched_split_sweep | 3 | 0.3790 | 0.0016 | 1.0000 |
| 2 | P0_router | matched_split_sweep | 3 | 0.3790 | 0.0016 | 0.9983 |
| 3 | TabR | single_reference | 1 | 0.3829 | - | 0.9215 |
| 4 | P0_full | matched_split_sweep | 3 | 0.3932 | 0.0038 | 0.7165 |
| 5 | TabPFN_full | matched_split_sweep | 3 | 0.3932 | 0.0038 | 0.7165 |
| 6 | A6f_artifact | historical_artifact | 1 | 0.4063 | - | 0.4531 |
| 7 | TabM | single_reference | 1 | 0.4290 | - | 0.0000 |

## Paired Wins vs `P0_router`

| Model | Comparable Runs | Target Wins | Mean Test Delta (positive favors target) |
| --- | ---: | ---: | ---: |
| P0_full | 3 | 3 | 0.0141 |
| P0_crossfit | 3 | 1 | -0.0001 |
| TabPFN_full | 3 | 3 | 0.0141 |
| TabR | 0 | - | - |
| TabM | 0 | - | - |
| A6f_artifact | 0 | - | - |
