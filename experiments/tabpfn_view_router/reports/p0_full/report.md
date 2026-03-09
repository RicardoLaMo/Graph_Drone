# P0 TabPFN View Router

- seed: `42`
- n_estimators per expert: `1`

## Results

| Model | Test RMSE | Val RMSE | Notes |
|---|---:|---:|---|
| P0_FULL | 0.3891 | 0.4111 | single-view TabPFN expert |
| P0_GEO | 0.5208 | 0.5422 | single-view TabPFN expert |
| P0_SOCIO | 0.6771 | 0.7121 | single-view TabPFN expert |
| P0_LOWRANK | 0.5622 | 0.5924 | single-view TabPFN expert |
| P0_uniform | 0.4359 | 0.4686 | uniform mean over four TabPFN view experts |
| P0_sigma2 | 0.4361 | 0.4694 | inverse-sigma2 routing over view experts |
| P0_router | 0.3773 | 0.3989 | softmax router on sigma2_v + J_flat + mean_J (best_epoch=399) |

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
- router_FULL: val `0.717` / test `0.720`
- router_GEO: val `0.235` / test `0.231`
- router_SOCIO: val `0.025` / test `0.025`
- router_LOWRANK: val `0.023` / test `0.023`
