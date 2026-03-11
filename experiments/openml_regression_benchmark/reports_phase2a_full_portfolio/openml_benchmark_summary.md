# OpenML Regression Benchmark Summary

## airfoil_self_noise

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 1.2742 | 0.1372 | 1.3394 | 0.8267 | 0.9654 | 3 |
| GraphDrone | 1.3275 | 0.1232 | 1.4304 | 0.8743 | 0.9625 | 3 |
| TabM | 1.4153 | 0.1112 | 1.5161 | 0.9726 | 0.9573 | 3 |
| TabR | 1.6993 | 0.2096 | 1.6262 | 1.2072 | 0.9381 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 1.1853 | 1.2969 | 1.1200 | 1.4577 |
| 0 | 1 | 1.4025 | 1.5175 | 1.3198 | 1.8329 |
| 0 | 2 | 1.3946 | 1.4314 | 1.3828 | 1.8073 |

### GraphDrone Deltas

- vs TabR: `-0.3718` mean RMSE delta
- vs TabM: `-0.0878` mean RMSE delta
- vs TabPFN: `+0.0533` mean RMSE delta

## california_housing_openml

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone | 0.3943 | 0.0246 | 0.4271 | 0.2437 | 0.8815 | 3 |
| TabR | 0.4053 | 0.0192 | 0.4391 | 0.2589 | 0.8750 | 3 |
| TabPFN | 0.4076 | 0.0266 | 0.4408 | 0.2509 | 0.8734 | 3 |
| TabM | 0.4555 | 0.0138 | 0.4878 | 0.2912 | 0.8423 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.3874 | 0.4646 | 0.4045 | 0.3947 |
| 0 | 1 | 0.4217 | 0.4623 | 0.4356 | 0.4274 |
| 0 | 2 | 0.3739 | 0.4396 | 0.3826 | 0.3937 |

### GraphDrone Deltas

- vs TabR: `-0.0109` mean RMSE delta
- vs TabM: `-0.0612` mean RMSE delta
- vs TabPFN: `-0.0132` mean RMSE delta

## concrete_compressive_strength

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone | 4.3203 | 0.0830 | 4.1830 | 2.7754 | 0.9329 | 3 |
| TabPFN | 4.3306 | 0.1124 | 4.1992 | 2.7968 | 0.9326 | 3 |
| TabM | 5.1452 | 0.0626 | 4.7368 | 3.3940 | 0.9049 | 3 |
| TabR | 5.4709 | 0.3836 | 5.1425 | 3.8843 | 0.8922 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 4.2313 | 5.0891 | 4.2012 | 5.4819 |
| 0 | 1 | 4.3955 | 5.2126 | 4.4038 | 5.8489 |
| 0 | 2 | 4.3343 | 5.1339 | 4.3867 | 5.0820 |

### GraphDrone Deltas

- vs TabR: `-1.1506` mean RMSE delta
- vs TabM: `-0.8248` mean RMSE delta
- vs TabPFN: `-0.0102` mean RMSE delta

## diamonds

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 512.7602 | 2.1000 | 511.2449 | 248.4927 | 0.9835 | 3 |
| GraphDrone | 516.9677 | 5.0177 | 512.5327 | 252.5497 | 0.9832 | 3 |
| TabM | 532.0915 | 6.4088 | 525.3251 | 276.8310 | 0.9822 | 3 |
| TabR | 546.1741 | 6.0849 | 531.7461 | 284.4295 | 0.9813 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 522.5086 | 538.5349 | 515.0420 | 552.7999 |
| 0 | 1 | 512.7306 | 532.0218 | 510.9087 | 544.8861 |
| 0 | 2 | 515.6638 | 525.7179 | 512.3300 | 540.8363 |

### GraphDrone Deltas

- vs TabR: `-29.2064` mean RMSE delta
- vs TabM: `-15.1238` mean RMSE delta
- vs TabPFN: `+4.2075` mean RMSE delta

## healthcare_insurance_expenses

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone | 4510.7174 | 565.3983 | 4333.4612 | 1995.1890 | 0.8607 | 3 |
| TabPFN | 4540.5938 | 592.1278 | 4350.6442 | 2038.3326 | 0.8589 | 3 |
| TabM | 4591.8505 | 543.4953 | 4378.6510 | 2511.5497 | 0.8557 | 3 |
| TabR | 4978.0528 | 407.4173 | 4514.4162 | 2902.7545 | 0.8307 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 4145.7877 | 4246.1517 | 4142.2723 | 4574.8199 |
| 0 | 1 | 5162.0052 | 5218.3037 | 5221.0240 | 5389.5295 |
| 0 | 2 | 4224.3594 | 4311.0961 | 4258.4852 | 4969.8090 |

### GraphDrone Deltas

- vs TabR: `-467.3353` mean RMSE delta
- vs TabM: `-81.1331` mean RMSE delta
- vs TabPFN: `-29.8764` mean RMSE delta

## houses

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone | 0.1990 | 0.0055 | 0.1937 | 0.1303 | 0.8777 | 3 |
| TabPFN | 0.2037 | 0.0046 | 0.1988 | 0.1332 | 0.8718 | 3 |
| TabR | 0.2048 | 0.0033 | 0.1978 | 0.1361 | 0.8705 | 3 |
| TabM | 0.2290 | 0.0036 | 0.2222 | 0.1561 | 0.8380 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.2014 | 0.2292 | 0.2041 | 0.2038 |
| 0 | 1 | 0.1927 | 0.2254 | 0.1989 | 0.2021 |
| 0 | 2 | 0.2030 | 0.2325 | 0.2082 | 0.2084 |

### GraphDrone Deltas

- vs TabR: `-0.0057` mean RMSE delta
- vs TabM: `-0.0300` mean RMSE delta
- vs TabPFN: `-0.0047` mean RMSE delta

## miami_housing

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 80132.0483 | 2656.8932 | 82974.2079 | 38061.4987 | 0.9364 | 3 |
| GraphDrone | 81762.6177 | 1812.8705 | 83015.9367 | 39069.3893 | 0.9338 | 3 |
| TabR | 89187.9895 | 1395.7350 | 91170.2700 | 44290.7994 | 0.9211 | 3 |
| TabM | 89375.8604 | 1522.5388 | 92708.7826 | 44716.8203 | 0.9209 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 83143.3838 | 91047.6510 | 82571.1899 | 88396.1174 |
| 0 | 1 | 82434.8105 | 88068.8371 | 80524.0055 | 88368.2857 |
| 0 | 2 | 79709.6587 | 89011.0932 | 77300.9495 | 90799.5653 |

### GraphDrone Deltas

- vs TabR: `-7425.3718` mean RMSE delta
- vs TabM: `-7613.2428` mean RMSE delta
- vs TabPFN: `+1630.5694` mean RMSE delta

## used_fiat_500

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 732.0692 | 34.4613 | 787.4501 | 549.0113 | 0.8574 | 3 |
| GraphDrone | 734.4574 | 39.7127 | 787.1763 | 547.6829 | 0.8565 | 3 |
| TabM | 767.5875 | 28.4806 | 832.7882 | 586.1259 | 0.8433 | 3 |
| TabR | 780.3606 | 34.7720 | 834.0192 | 590.8100 | 0.8380 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 691.9654 | 735.7328 | 694.5983 | 740.3116 |
| 0 | 1 | 740.7729 | 776.4365 | 739.2060 | 797.9068 |
| 0 | 2 | 770.6340 | 790.5931 | 762.4034 | 802.8635 |

### GraphDrone Deltas

- vs TabR: `-45.9032` mean RMSE delta
- vs TabM: `-33.1300` mean RMSE delta
- vs TabPFN: `+2.3882` mean RMSE delta

## wine_quality

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 0.6465 | 0.0182 | 0.6459 | 0.4980 | 0.4514 | 3 |
| GraphDrone | 0.6515 | 0.0135 | 0.6466 | 0.4952 | 0.4427 | 3 |
| TabR | 0.6598 | 0.0162 | 0.6512 | 0.4970 | 0.4287 | 3 |
| TabM | 0.6748 | 0.0193 | 0.6703 | 0.5152 | 0.4024 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.6479 | 0.6648 | 0.6365 | 0.6502 |
| 0 | 1 | 0.6664 | 0.6970 | 0.6675 | 0.6785 |
| 0 | 2 | 0.6402 | 0.6624 | 0.6355 | 0.6506 |

### GraphDrone Deltas

- vs TabR: `-0.0083` mean RMSE delta
- vs TabM: `-0.0233` mean RMSE delta
- vs TabPFN: `+0.0050` mean RMSE delta

