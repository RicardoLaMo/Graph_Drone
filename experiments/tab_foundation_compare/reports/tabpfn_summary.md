# TabPFN Summary

| Run | Train rows | n_estimators | Test RMSE | Val RMSE | Duration s | Note |
|---|---:|---:|---:|---:|---:|---|
| TabPFN aligned (8k cap) | 8000 | 8 | 0.4067 | 0.4374 | n/a | legacy-style constrained CPU run |
| TabPFN aligned (full train) | 14448 | 8 | 0.3927 | 0.4202 | 480.1 | fair full-train run |
| TabPFN aligned (full train, e1) | 14448 | 1 | 0.3891 | 0.4111 | 71.0 | strongest measured TabPFN run |

Current best measured TabPFN number on this branch is `0.3891`.
