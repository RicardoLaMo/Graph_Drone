# TabPFN Full-Train Multi-Seed Check

| Seed | Test RMSE | Val RMSE | Duration s |
|---|---:|---:|---:|
| 41 | 0.3982 | 0.4280 | 70.7 |
| 42 | 0.3891 | 0.4111 | 69.3 |
| 43 | 0.3922 | 0.4187 | 69.0 |

- mean test RMSE: `0.3932`
- std test RMSE: `0.0038`
- mean val RMSE: `0.4193`
- std val RMSE: `0.0069`

Interpretation: the single best `0.3891` run is not pure noise, but the stable reference should be the multi-seed mean `0.3932`.
