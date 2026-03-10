# OpenML Regression Benchmark Summary

## houses

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_crossfit | 0.1983 | 0.0049 | 0.1942 | 0.1300 | 0.8786 | 3 |
| GraphDrone_router | 0.1983 | 0.0049 | 0.1928 | 0.1301 | 0.8785 | 3 |
| GraphDrone_FULL | 0.2028 | 0.0051 | 0.1976 | 0.1316 | 0.8730 | 3 |
| TabPFN | 0.2037 | 0.0046 | 0.1988 | 0.1332 | 0.8718 | 3 |
| TabR | 0.2048 | 0.0033 | 0.1978 | 0.1361 | 0.8705 | 3 |
| TabM | 0.2290 | 0.0036 | 0.2222 | 0.1561 | 0.8380 | 3 |
| GraphDrone_gora | 0.2462 | 0.0044 | 0.2427 | 0.1774 | 0.8128 | 3 |
| GraphDrone_sigma2 | 0.2468 | 0.0047 | 0.2435 | 0.1819 | 0.8119 | 3 |
| GraphDrone_uniform | 0.2474 | 0.0047 | 0.2441 | 0.1825 | 0.8110 | 3 |
| GraphDrone_GEO | 0.2560 | 0.0060 | 0.2516 | 0.1803 | 0.7976 | 3 |
| GraphDrone_DOMAIN | 0.3203 | 0.0005 | 0.3162 | 0.2356 | 0.6830 | 3 |
| GraphDrone_LOWRANK | 0.4620 | 0.0053 | 0.4653 | 0.3698 | 0.3410 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.3201 | 0.2026 | 0.2604 | 0.4632 | 0.1995 | 0.2476 | 0.1995 | 0.2483 | 0.2488 | 0.2292 | 0.2041 | 0.2038 |
| 0 | 1 | 0.3200 | 0.1978 | 0.2492 | 0.4562 | 0.1929 | 0.2412 | 0.1929 | 0.2416 | 0.2422 | 0.2254 | 0.1989 | 0.2021 |
| 0 | 2 | 0.3210 | 0.2081 | 0.2585 | 0.4666 | 0.2024 | 0.2498 | 0.2026 | 0.2506 | 0.2511 | 0.2325 | 0.2082 | 0.2084 |

### GraphDrone Router Deltas

- vs TabR: `-0.0064` mean RMSE delta
- vs TabM: `-0.0307` mean RMSE delta
- vs TabPFN: `-0.0054` mean RMSE delta

## miami_housing

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 80132.0483 | 2656.8932 | 82974.2079 | 38061.4987 | 0.9364 | 3 |
| GraphDrone_FULL | 81717.8403 | 1269.6404 | 84734.2748 | 38763.9349 | 0.9339 | 3 |
| GraphDrone_crossfit | 82274.6784 | 1043.7627 | 86293.3594 | 41057.8164 | 0.9330 | 3 |
| GraphDrone_router | 82299.0846 | 1019.3351 | 84885.1305 | 41002.2331 | 0.9329 | 3 |
| TabR | 89461.7639 | 1920.6972 | 90352.1086 | 45695.1700 | 0.9206 | 3 |
| TabM | 89712.6471 | 1312.2093 | 93270.3333 | 44165.3633 | 0.9202 | 3 |
| GraphDrone_uniform | 101331.1553 | 2451.2787 | 106555.5074 | 49587.0117 | 0.8983 | 3 |
| GraphDrone_sigma2 | 101938.0481 | 2257.4117 | 107199.7437 | 49504.8464 | 0.8971 | 3 |
| GraphDrone_LOWRANK | 145912.5008 | 2818.8744 | 148877.5030 | 73968.3906 | 0.7890 | 3 |
| GraphDrone_gora | 147841.9300 | 4769.1877 | 155356.2992 | 67902.9688 | 0.7832 | 3 |
| GraphDrone_GEO | 158960.0919 | 5266.2944 | 162012.7946 | 78921.2708 | 0.7494 | 3 |
| GraphDrone_DOMAIN | 164749.1595 | 3927.8808 | 172499.0635 | 85071.9271 | 0.7311 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 162212.4095 | 82529.3618 | 160717.6099 | 146924.7304 | 83141.3332 | 142740.9061 | 83095.7408 | 104544.5632 | 104158.3435 | 89606.3255 | 82571.1899 | 87272.0291 |
| 0 | 1 | 169273.5938 | 82369.4639 | 153039.7850 | 142727.2823 | 82566.6943 | 152189.6114 | 82651.1218 | 100656.3955 | 99799.1455 | 88456.8331 | 80524.0055 | 90861.3839 |
| 0 | 2 | 162761.4751 | 80254.6954 | 163122.8806 | 148085.4897 | 81116.0078 | 148595.2725 | 81150.3912 | 100613.1856 | 100035.9769 | 91074.7828 | 77300.9495 | 90251.8787 |

### GraphDrone Router Deltas

- vs TabR: `-7162.6793` mean RMSE delta
- vs TabM: `-7413.5625` mean RMSE delta
- vs TabPFN: `+2167.0363` mean RMSE delta

