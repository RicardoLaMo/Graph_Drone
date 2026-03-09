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
| P0_sigma2 | 0.4395 | 0.4748 | inverse-sigma2 routing — no val labels used |
| P0_gora | 0.5179 | 0.5674 | GoRA analytical routing: softmax(-sigma2*tau), tau=1/(mean_J+eps) — zero params, no val labels |
| P0_router | 0.3784 | 0.4044 | softmax router on sigma2_v + J_flat + mean_J (best_epoch=330) |
| P0_crossfit | 0.3781 | 0.4066 | 5-fold OOF router — val RMSE is clean (unbiased); test uses router trained on all val (n_splits=5) |

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
- router_FULL: val `0.696` / test `0.698`
- router_GEO: val `0.254` / test `0.251`
- router_SOCIO: val `0.028` / test `0.028`
- router_LOWRANK: val `0.022` / test `0.023`
- crossfit_FULL: val `0.684` / test `0.684`
- crossfit_GEO: val `0.254` / test `0.254`
- crossfit_SOCIO: val `0.034` / test `0.034`
- crossfit_LOWRANK: val `0.028` / test `0.028`
