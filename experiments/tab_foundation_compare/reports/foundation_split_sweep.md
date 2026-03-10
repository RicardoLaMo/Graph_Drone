# Foundation Split Sweep Summary

| Split Seed | TabR | TabM | TabPFN_full | P0_router | P0_full | P0_crossfit |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.3906 | 0.4260 | 0.3920 | 0.3777 | 0.3891 | 0.3776 |
| 43 | 0.4017 | 0.4385 | 0.4002 | 0.3882 | 0.3992 | 0.3880 |
| 44 | 0.3979 | 0.4429 | 0.4039 | 0.3857 | 0.3982 | 0.3858 |
| 45 | 0.4203 | 0.4554 | 0.4180 | 0.4034 | 0.4123 | 0.4034 |
| 46 | 0.4084 | 0.4471 | 0.4127 | 0.4058 | 0.4078 | 0.4064 |

## Means

| Model | Mean RMSE | Std |
|---|---:|---:|
| TabR | 0.4038 | 0.0101 |
| TabM | 0.4420 | 0.0098 |
| TabPFN_full | 0.4054 | 0.0092 |
| P0_router | 0.3922 | 0.0108 |
| P0_full | 0.4013 | 0.0081 |
| P0_crossfit | 0.3922 | 0.0110 |

## Notes

- `P0_router` beats `TabR` on `5/5` matched split seeds.
- This summary varies split seed while leaving each model family on its own fixed training seed/config path.
