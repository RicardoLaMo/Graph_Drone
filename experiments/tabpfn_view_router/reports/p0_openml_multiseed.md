# P0 OpenML Multiseed Summary

- dataset source: `openml`
- OpenML dataset id: `44024`
- split_seed: `42`

## Seed Table

| Seed | P0_FULL | P0_router | P0_crossfit |
|---|---:|---:|---:|
| 41 | 0.3976 | 0.3807 | 0.3805 |
| 42 | 0.3891 | 0.3774 | 0.3776 |
| 43 | 0.3908 | 0.3775 | 0.3779 |

## Multiseed Means

| Model | Mean RMSE | Std | vs TabR | vs TabM | vs TabPFN best | vs TabPFN mean |
|---|---:|---:|---:|---:|---:|---:|
| P0_FULL | 0.3925 | 0.0037 | +0.0096 | -0.0365 | +0.0034 | -0.0007 |
| P0_router | 0.3786 | 0.0015 | -0.0043 | -0.0504 | -0.0105 | -0.0146 |
| P0_crossfit | 0.3787 | 0.0013 | -0.0042 | -0.0503 | -0.0104 | -0.0145 |
| P0_uniform | 0.4373 | 0.0013 | +0.0544 | +0.0083 | +0.0482 | +0.0441 |
| P0_sigma2 | 0.4375 | 0.0011 | +0.0546 | +0.0085 | +0.0484 | +0.0443 |
| P0_gora | 0.5198 | 0.0011 | +0.1369 | +0.0908 | +0.1307 | +0.1266 |

## Reference Anchors

- TabR_on_our_split: `0.3829`
- TabM_on_our_split: `0.4290`
- TabPFN_full_best: `0.3891`
- TabPFN_full_multiseed_mean: `0.3932`
- MV-TabR-GoRA A6f artifact: `0.4063`

