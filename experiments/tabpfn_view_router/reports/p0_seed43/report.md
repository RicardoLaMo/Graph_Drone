# P0 TabPFN View Router

- seed: `43`
- n_estimators per expert: `1`

## Results

| Model | Test RMSE | Val RMSE | Notes |
|---|---:|---:|---|
| P0_FULL | 0.3922 | 0.4187 | single-view TabPFN expert |
| P0_GEO | 0.5306 | 0.5448 | single-view TabPFN expert |
| P0_SOCIO | 0.6753 | 0.7130 | single-view TabPFN expert |
| P0_LOWRANK | 0.5688 | 0.5967 | single-view TabPFN expert |
| P0_uniform | 0.4396 | 0.4741 | uniform mean over four TabPFN view experts |
| P0_sigma2 | 0.4395 | 0.4748 | inverse-sigma2 routing over view experts |
| P0_router | 0.3784 | 0.4046 | softmax router on sigma2_v + J_flat + mean_J (best_epoch=338) |

## Reference Anchors

- TabR_on_our_split: `0.3829`
- TabPFN_full_best: `0.3891`
- TabPFN_full_multiseed_mean: `0.3932`
- MV-TabR-GoRA A6f artifact: `0.4063`

## Learned Router Mean Weights (val / test)

- uniform: val `0.250` / test `0.250`
- sigma2_FULL: val `0.251` / test `0.251`
- sigma2_GEO: val `0.246` / test `0.246`
- sigma2_SOCIO: val `0.251` / test `0.251`
- sigma2_LOWRANK: val `0.251` / test `0.252`
- router_FULL: val `0.676` / test `0.678`
- router_GEO: val `0.268` / test `0.265`
- router_SOCIO: val `0.030` / test `0.030`
- router_LOWRANK: val `0.026` / test `0.026`
