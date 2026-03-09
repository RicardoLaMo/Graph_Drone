# P0 TabPFN View Router

- seed: `42`
- n_estimators per expert: `1`

## Results

| Model | Test RMSE | Val RMSE | Notes |
|---|---:|---:|---|
| P0_FULL | 0.4813 | 0.5410 | single-view TabPFN expert |
| P0_GEO | 0.7666 | 0.7228 | single-view TabPFN expert |
| P0_SOCIO | 0.7134 | 0.7420 | single-view TabPFN expert |
| P0_LOWRANK | 0.6283 | 0.7307 | single-view TabPFN expert |
| P0_uniform | 0.5141 | 0.5648 | uniform mean over four TabPFN view experts |
| P0_sigma2 | 0.5142 | 0.5701 | inverse-sigma2 routing over view experts |
| P0_router | 0.4908 | 0.5071 | softmax router on sigma2_v + J_flat + mean_J (best_epoch=118) |

## Reference Anchors

- TabR_on_our_split: `0.3829`
- TabPFN_full_best: `0.3891`
- TabPFN_full_multiseed_mean: `0.3932`
- MV-TabR-GoRA A6f artifact: `0.4063`

## Learned Router Mean Weights (val / test)

- uniform: val `0.250` / test `0.250`
- sigma2_FULL: val `0.251` / test `0.251`
- sigma2_GEO: val `0.249` / test `0.250`
- sigma2_SOCIO: val `0.249` / test `0.249`
- sigma2_LOWRANK: val `0.251` / test `0.250`
- router_FULL: val `0.486` / test `0.499`
- router_GEO: val `0.292` / test `0.279`
- router_SOCIO: val `0.151` / test `0.152`
- router_LOWRANK: val `0.071` / test `0.070`
