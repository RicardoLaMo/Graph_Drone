# California Housing: Routing Curvature Report
*2026-03-07* | Branch: `feature/routing-curvature-dual-datasets`

> Curvature used as ROUTING PRIOR only, not as direct predictor.

## Results

| model         |     rmse |      mae |       r2 |
|:--------------|---------:|---------:|---------:|
| C8_Routed     | 0.52501  | 0.373685 | 0.79155  |
| C5_GEO        | 0.526111 | 0.367417 | 0.790675 |
| C7_Learned    | 0.547883 | 0.379397 | 0.772992 |
| C9_RoutedGate | 0.625874 | 0.35396  | 0.703762 |
| C5_SOCIO      | 0.738488 | 0.476657 | 0.587567 |
| C5_FULL       | 0.747764 | 0.494852 | 0.577141 |
| C5_LOWRANK    | 0.814149 | 0.490501 | 0.498727 |

## Verdict: **LOCAL NON-FLATNESS HELPS ROUTING ON CALIFORNIA**

```
C2: N/A
C7_Learned=0.5479
C8_Routed=0.5250
C9_RoutedGate=0.6259
routing_helps=True
gate_helps=False
graph_beats_boost=True
```

## Curvature Stats
kappa mean=0.2849 std=0.0528

## Stability

|   spearman_k10_k20 |   pval_k10_k20 |   top20pct_overlap_k10_k20 |   spearman_k10_k30 |   pval_k10_k30 |   top20pct_overlap_k10_k30 |   spearman_k20_k30 |   pval_k20_k30 |   top20pct_overlap_k20_k30 |
|-------------------:|---------------:|---------------------------:|-------------------:|---------------:|---------------------------:|-------------------:|---------------:|---------------------------:|
|           0.261585 |              0 |                   0.290698 |           0.166195 |   9.97979e-128 |                   0.256298 |           0.474254 |              0 |                   0.380572 |

## Per-Bin RMSE

| model         | bin    |     rmse |   n_rows |
|:--------------|:-------|---------:|---------:|
| C5_FULL       | low    | 0.90348  |     1014 |
| C5_FULL       | medium | 0.690549 |     1078 |
| C5_FULL       | high   | 0.622756 |     1004 |
| C5_GEO        | low    | 0.574878 |     1014 |
| C5_GEO        | medium | 0.526187 |     1078 |
| C5_GEO        | high   | 0.471679 |     1004 |
| C5_SOCIO      | low    | 0.899639 |     1014 |
| C5_SOCIO      | medium | 0.666197 |     1078 |
| C5_SOCIO      | high   | 0.62272  |     1004 |
| C5_LOWRANK    | low    | 1.08845  |     1014 |
| C5_LOWRANK    | medium | 0.654806 |     1078 |
| C5_LOWRANK    | high   | 0.622149 |     1004 |
| C7_Learned    | low    | 0.615507 |     1014 |
| C7_Learned    | medium | 0.536089 |     1078 |
| C7_Learned    | high   | 0.484195 |     1004 |
| C8_Routed     | low    | 0.56812  |     1014 |
| C8_Routed     | medium | 0.525139 |     1078 |
| C8_Routed     | high   | 0.477386 |     1004 |
| C9_RoutedGate | low    | 0.854388 |     1014 |
| C9_RoutedGate | medium | 0.488205 |     1078 |
| C9_RoutedGate | high   | 0.46343  |     1004 |