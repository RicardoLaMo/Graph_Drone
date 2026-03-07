# MNIST-784: Routing Curvature Report
*2026-03-07* | Branch: `feature/routing-curvature-dual-datasets`

> Curvature used as ROUTING PRIOR only.

## Results

| model    |   accuracy |   macro_f1 |
|:---------|-----------:|-----------:|
| M5_PCA   |   0.942    |   0.941388 |
| M5_FULL  |   0.934    |   0.933184 |
| M5_BLOCK |   0.926667 |   0.925681 |

## Verdict: **WARNING SIGN: ROUTING FAILED TO EXPLOIT HIDDEN GEOMETRY MEANINGFULLY**

routing_helps=False, gate_helps=False, graph>mlp=False

## Curvature
kappa mean=0.4950 std=0.0695

## Stability

|   spearman_k10_k20 |   pval_k10_k20 |   top20pct_overlap_k10_k20 |   spearman_k10_k30 |   pval_k10_k30 |   top20pct_overlap_k10_k30 |   spearman_k20_k30 |   pval_k20_k30 |   top20pct_overlap_k20_k30 |
|-------------------:|---------------:|---------------------------:|-------------------:|---------------:|---------------------------:|-------------------:|---------------:|---------------------------:|
|           0.674682 |              0 |                     0.4975 |           0.618644 |              0 |                     0.4565 |           0.859386 |              0 |                      0.656 |

## Per-Bin Accuracy

| model    | bin    |   accuracy |   n_rows |
|:---------|:-------|-----------:|---------:|
| M5_FULL  | low    |   0.928144 |      501 |
| M5_FULL  | medium |   0.930894 |      492 |
| M5_FULL  | high   |   0.942801 |      507 |
| M5_BLOCK | low    |   0.934132 |      501 |
| M5_BLOCK | medium |   0.920732 |      492 |
| M5_BLOCK | high   |   0.925049 |      507 |
| M5_PCA   | low    |   0.936128 |      501 |
| M5_PCA   | medium |   0.936992 |      492 |
| M5_PCA   | high   |   0.952663 |      507 |