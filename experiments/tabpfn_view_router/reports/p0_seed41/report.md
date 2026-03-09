# P0 TabPFN View Router

- seed: `41`
- n_estimators per expert: `1`

## Results

| Model | Test RMSE | Val RMSE | Notes |
|---|---:|---:|---|
| P0_FULL | 0.3982 | 0.4280 | single-view TabPFN expert |
| P0_GEO | 0.5302 | 0.5368 | single-view TabPFN expert |
| P0_SOCIO | 0.6764 | 0.7143 | single-view TabPFN expert |
| P0_LOWRANK | 0.5690 | 0.5911 | single-view TabPFN expert |
| P0_uniform | 0.4381 | 0.4710 | uniform mean over four TabPFN view experts |
| P0_sigma2 | 0.4383 | 0.4722 | inverse-sigma2 routing — no val labels used |
| P0_gora | 0.5220 | 0.5699 | GoRA analytical routing: softmax(-sigma2*tau), tau=1/(mean_J+eps) — zero params, no val labels |
| P0_router | 0.3812 | 0.4068 | softmax router on sigma2_v + J_flat + mean_J (best_epoch=399) |
| P0_crossfit | 0.3812 | 0.4078 | 5-fold OOF router — val RMSE is clean (unbiased); test uses router trained on all val (n_splits=5) |

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
- gora_FULL: val `0.201` / test `0.202`
- gora_GEO: val `0.225` / test `0.214`
- gora_SOCIO: val `0.322` / test `0.320`
- gora_LOWRANK: val `0.253` / test `0.264`
- router_FULL: val `0.653` / test `0.655`
- router_GEO: val `0.280` / test `0.276`
- router_SOCIO: val `0.034` / test `0.034`
- router_LOWRANK: val `0.034` / test `0.034`
- crossfit_FULL: val `0.664` / test `0.664`
- crossfit_GEO: val `0.273` / test `0.273`
- crossfit_SOCIO: val `0.031` / test `0.031`
- crossfit_LOWRANK: val `0.032` / test `0.032`
