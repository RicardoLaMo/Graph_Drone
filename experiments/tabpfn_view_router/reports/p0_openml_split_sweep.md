# P0 OpenML Split Sweep Summary

- dataset source: `openml`
- OpenML dataset id: `44024`
- model_seed: `42`

## Split Table

| Split Seed | P0_FULL | P0_router | P0_crossfit |
|---|---:|---:|---:|
| 42 | 0.3891 | 0.3777 | 0.3776 |
| 43 | 0.3992 | 0.3882 | 0.3880 |
| 44 | 0.3982 | 0.3857 | 0.3858 |
| 45 | 0.4123 | 0.4034 | 0.4034 |
| 46 | 0.4078 | 0.4058 | 0.4064 |

## Across-Split Means

| Model | Mean RMSE | Std | Wins vs TabR | Wins vs TabPFN best |
|---|---:|---:|---:|---:|
| P0_FULL | 0.4013 | 0.0081 | 0/5 | 1/5 |
| P0_router | 0.3922 | 0.0108 | 1/5 | 3/5 |
| P0_crossfit | 0.3922 | 0.0110 | 1/5 | 3/5 |
| P0_uniform | 0.4480 | 0.0097 | 0/5 | 0/5 |
| P0_sigma2 | 0.4483 | 0.0096 | 0/5 | 0/5 |
| P0_gora | 0.5330 | 0.0101 | 0/5 | 0/5 |

## Reference Anchors

- TabR_on_our_split: `0.3829`
- TabM_on_our_split: `0.4290`
- TabPFN_full_best: `0.3891`
- TabPFN_full_multiseed_mean: `0.3932`
- MV-TabR-GoRA A6f artifact: `0.4063`

## Notes

- This sweep varies `split_seed` while keeping the model seed fixed.
- Use this as a split-dependence check, not as a full statistical significance study.

