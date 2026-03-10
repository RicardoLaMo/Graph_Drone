# OpenML Regression Benchmark Summary

## airfoil_self_noise

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_FOUNDATION | 1.2742 | 0.1372 | 1.3394 | 0.8267 | 0.9654 | 3 |
| TabPFN | 1.2742 | 0.1372 | 1.3394 | 0.8267 | 0.9654 | 3 |
| GraphDrone_foundation_crossfit | 1.2936 | 0.1355 | 1.3733 | 0.8633 | 0.9644 | 3 |
| GraphDrone_foundation_router | 1.3016 | 0.1411 | 1.2556 | 0.8750 | 0.9639 | 3 |
| GraphDrone_trust_gate | 1.3076 | 0.1236 | 1.3693 | 0.8662 | 0.9636 | 3 |
| GraphDrone_FULL | 1.3248 | 0.1210 | 1.4344 | 0.8685 | 0.9626 | 3 |
| GraphDrone_crossfit | 1.3450 | 0.1515 | 1.4354 | 0.8972 | 0.9614 | 3 |
| GraphDrone_router | 1.3538 | 0.1540 | 1.3770 | 0.9041 | 0.9609 | 3 |
| GraphDrone_LOWRANK | 1.3963 | 0.1578 | 1.4629 | 0.8845 | 0.9587 | 3 |
| TabM | 1.4153 | 0.1112 | 1.5161 | 0.9726 | 0.9573 | 3 |
| TabR | 1.6993 | 0.2096 | 1.6262 | 1.2072 | 0.9381 | 3 |
| GraphDrone_uniform | 2.2681 | 0.1145 | 2.2945 | 1.7407 | 0.8917 | 3 |
| GraphDrone_sigma2 | 2.2964 | 0.1191 | 2.3097 | 1.7691 | 0.8890 | 3 |
| GraphDrone_GEO | 2.8596 | 0.0679 | 2.7937 | 2.1328 | 0.8270 | 3 |
| GraphDrone_gora | 3.4926 | 0.2299 | 3.3764 | 2.4725 | 0.7433 | 3 |
| GraphDrone_DOMAIN | 6.2477 | 0.1661 | 6.2081 | 5.1472 | 0.1767 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FOUNDATION | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_foundation_crossfit | GraphDrone_foundation_router | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_trust_gate | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 6.1794 | 1.1200 | 1.1853 | 2.7812 | 1.2544 | 1.1712 | 1.1397 | 1.1400 | 3.4840 | 1.1768 | 2.2926 | 1.1649 | 2.2619 | 1.2969 | 1.1200 | 1.4577 |
| 0 | 1 | 6.1267 | 1.3198 | 1.4015 | 2.8984 | 1.3683 | 1.4151 | 1.3462 | 1.3646 | 3.2672 | 1.4278 | 2.1793 | 1.3805 | 2.1568 | 1.5175 | 1.3198 | 1.8329 |
| 0 | 2 | 6.4371 | 1.3828 | 1.3877 | 2.8991 | 1.5662 | 1.4488 | 1.3950 | 1.4001 | 3.7267 | 1.4569 | 2.4173 | 1.3774 | 2.3855 | 1.4314 | 1.3828 | 1.8073 |

### GraphDrone Router Deltas

- vs TabR: `-0.3455` mean RMSE delta
- vs TabM: `-0.0614` mean RMSE delta
- vs TabPFN: `+0.0796` mean RMSE delta

## concrete_compressive_strength

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_FOUNDATION | 4.3306 | 0.1124 | 4.1992 | 2.7968 | 0.9326 | 3 |
| TabPFN | 4.3306 | 0.1124 | 4.1992 | 2.7968 | 0.9326 | 3 |
| GraphDrone_trust_gate | 4.3470 | 0.1107 | 4.1338 | 2.7988 | 0.9320 | 3 |
| GraphDrone_FULL | 4.3695 | 0.1427 | 4.2029 | 2.7831 | 0.9313 | 3 |
| GraphDrone_foundation_crossfit | 4.4690 | 0.1461 | 4.2584 | 3.0371 | 0.9282 | 3 |
| GraphDrone_foundation_router | 4.4880 | 0.1442 | 3.9879 | 3.0413 | 0.9276 | 3 |
| GraphDrone_crossfit | 4.5075 | 0.0394 | 4.4494 | 3.0470 | 0.9270 | 3 |
| GraphDrone_router | 4.5440 | 0.0622 | 4.1404 | 3.0835 | 0.9258 | 3 |
| TabM | 5.1452 | 0.0626 | 4.7368 | 3.3940 | 0.9049 | 3 |
| TabR | 5.4709 | 0.3836 | 5.1425 | 3.8843 | 0.8922 | 3 |
| GraphDrone_uniform | 6.5870 | 0.3383 | 6.0607 | 4.9625 | 0.8441 | 3 |
| GraphDrone_sigma2 | 6.5892 | 0.3857 | 6.0603 | 4.9381 | 0.8439 | 3 |
| GraphDrone_gora | 7.8559 | 1.1136 | 7.7901 | 5.4602 | 0.7756 | 3 |
| GraphDrone_LOWRANK | 8.8597 | 0.9279 | 7.6935 | 6.2748 | 0.7159 | 3 |
| GraphDrone_DOMAIN | 10.1999 | 0.7868 | 9.7662 | 7.2708 | 0.6260 | 3 |
| GraphDrone_GEO | 11.4821 | 0.3365 | 11.4618 | 9.0767 | 0.5258 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FOUNDATION | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_foundation_crossfit | GraphDrone_foundation_router | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_trust_gate | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 11.0576 | 4.2012 | 4.2155 | 11.1850 | 8.7698 | 4.4689 | 4.4305 | 4.4812 | 8.1299 | 4.4921 | 6.7981 | 4.2302 | 6.7968 | 5.0891 | 4.2012 | 5.4819 |
| 0 | 1 | 9.5114 | 4.4038 | 4.3954 | 11.4138 | 7.9801 | 4.5477 | 4.3461 | 4.3474 | 6.6310 | 4.6130 | 6.1441 | 4.3605 | 6.1968 | 5.2126 | 4.4038 | 5.8489 |
| 0 | 2 | 10.0307 | 4.3867 | 4.4975 | 11.8476 | 9.8293 | 4.5060 | 4.6305 | 4.6355 | 8.8070 | 4.5269 | 6.8253 | 4.4503 | 6.7675 | 5.1339 | 4.3867 | 5.0820 |

### GraphDrone Router Deltas

- vs TabR: `-0.9269` mean RMSE delta
- vs TabM: `-0.6012` mean RMSE delta
- vs TabPFN: `+0.2134` mean RMSE delta

## diamonds

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_FOUNDATION | 523.7256 | 4.1723 | 521.5398 | 253.9760 | 0.9828 | 3 |
| TabPFN | 523.7256 | 4.1723 | 521.5398 | 253.9760 | 0.9828 | 3 |
| GraphDrone_foundation_router | 525.3961 | 6.0012 | 521.4492 | 257.4979 | 0.9827 | 3 |
| GraphDrone_foundation_crossfit | 527.1450 | 6.5995 | 524.1005 | 257.6547 | 0.9825 | 3 |
| GraphDrone_trust_gate | 527.8275 | 9.0317 | 523.6699 | 258.6218 | 0.9825 | 3 |
| GraphDrone_FULL | 528.8255 | 9.1308 | 524.9371 | 257.9534 | 0.9824 | 3 |
| TabM | 532.0915 | 6.4088 | 525.3251 | 276.8310 | 0.9822 | 3 |
| GraphDrone_router | 539.3399 | 10.4262 | 533.7548 | 290.4651 | 0.9817 | 3 |
| GraphDrone_crossfit | 539.5251 | 12.3508 | 535.9425 | 292.2800 | 0.9817 | 3 |
| TabR | 546.1741 | 6.0849 | 531.7461 | 284.4295 | 0.9813 | 3 |
| GraphDrone_LOWRANK | 660.7029 | 5.9458 | 642.8334 | 325.7516 | 0.9726 | 3 |
| GraphDrone_uniform | 1240.4935 | 23.5091 | 1226.5922 | 819.5831 | 0.9033 | 3 |
| GraphDrone_sigma2 | 1308.1576 | 21.3983 | 1294.3531 | 839.3734 | 0.8924 | 3 |
| GraphDrone_GEO | 1348.2688 | 30.3891 | 1351.6375 | 757.9259 | 0.8857 | 3 |
| GraphDrone_gora | 3110.4968 | 77.9138 | 3078.5861 | 1568.6736 | 0.3919 | 3 |
| GraphDrone_DOMAIN | 3845.6160 | 50.9239 | 3837.3349 | 2728.6335 | 0.0706 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FOUNDATION | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_foundation_crossfit | GraphDrone_foundation_router | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_trust_gate | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 3895.0171 | 527.7638 | 539.3645 | 1361.2683 | 657.5736 | 553.6081 | 533.0316 | 530.5601 | 3147.9576 | 551.3432 | 1314.0276 | 538.2040 | 1257.5668 | 538.5349 | 527.7638 | 552.7999 |
| 0 | 1 | 3848.5361 | 523.9821 | 523.8164 | 1369.9960 | 667.5598 | 530.5351 | 528.3927 | 526.8159 | 3162.6048 | 534.1422 | 1326.0083 | 523.5442 | 1250.2340 | 532.0218 | 523.9821 | 544.8861 |
| 0 | 2 | 3793.2949 | 519.4310 | 523.2955 | 1313.5422 | 656.9753 | 534.4322 | 520.0108 | 518.8123 | 3020.9280 | 532.5343 | 1284.4370 | 521.7343 | 1213.6797 | 525.7179 | 519.4310 | 540.8363 |

### GraphDrone Router Deltas

- vs TabR: `-6.8342` mean RMSE delta
- vs TabM: `+7.2484` mean RMSE delta
- vs TabPFN: `+15.6143` mean RMSE delta

## healthcare_insurance_expenses

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_router | 4478.7382 | 573.6306 | 4337.4457 | 2132.5563 | 0.8627 | 3 |
| GraphDrone_crossfit | 4481.8108 | 568.3337 | 4371.6513 | 2136.0833 | 0.8625 | 3 |
| GraphDrone_GEO | 4486.7132 | 538.4305 | 4374.6924 | 2152.7913 | 0.8623 | 3 |
| GraphDrone_foundation_router | 4494.0551 | 566.2946 | 4305.3298 | 2008.6199 | 0.8618 | 3 |
| GraphDrone_foundation_crossfit | 4495.7049 | 565.0334 | 4332.0833 | 2035.0382 | 0.8617 | 3 |
| GraphDrone_trust_gate | 4502.8562 | 551.8133 | 4344.4441 | 2016.2057 | 0.8612 | 3 |
| GraphDrone_FULL | 4510.7199 | 565.3964 | 4333.5244 | 1995.2655 | 0.8607 | 3 |
| GraphDrone_FOUNDATION | 4540.5938 | 592.1278 | 4350.6442 | 2038.3326 | 0.8589 | 3 |
| TabPFN | 4540.5938 | 592.1278 | 4350.6442 | 2038.3326 | 0.8589 | 3 |
| TabM | 4591.8505 | 543.4953 | 4378.6510 | 2511.5497 | 0.8557 | 3 |
| TabR | 4978.0528 | 407.4173 | 4514.4162 | 2902.7545 | 0.8307 | 3 |
| GraphDrone_uniform | 6985.0462 | 722.8892 | 6790.8786 | 5017.6748 | 0.6666 | 3 |
| GraphDrone_sigma2 | 7191.0471 | 622.9539 | 6925.4089 | 5121.1125 | 0.6467 | 3 |
| GraphDrone_gora | 8785.1727 | 818.1113 | 8484.3567 | 5800.3735 | 0.4686 | 3 |
| GraphDrone_LOWRANK | 11602.7969 | 710.1664 | 11206.6387 | 8343.1323 | 0.0790 | 3 |
| GraphDrone_DOMAIN | 12181.5527 | 627.9905 | 11433.3344 | 8657.3649 | -0.0152 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FOUNDATION | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_foundation_crossfit | GraphDrone_foundation_router | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_trust_gate | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 11557.5153 | 4142.2723 | 4145.7877 | 4108.6555 | 10797.3708 | 4097.4763 | 4108.3264 | 4103.1875 | 8040.4858 | 4089.7290 | 6559.2500 | 4145.7839 | 6371.6095 | 4246.1517 | 4142.2723 | 4574.8199 |
| 0 | 1 | 12813.4230 | 5221.0240 | 5162.0052 | 5103.1900 | 12138.9377 | 5134.6497 | 5144.0546 | 5143.4784 | 8654.1403 | 5137.5227 | 7804.7642 | 5138.4180 | 7782.0098 | 5218.3037 | 5221.0240 | 5389.5295 |
| 0 | 2 | 12173.7199 | 4258.4852 | 4224.3667 | 4248.2942 | 11872.0822 | 4213.3063 | 4234.7335 | 4235.4993 | 9660.8919 | 4208.9628 | 7209.1273 | 4224.3667 | 6801.5192 | 4311.0961 | 4258.4852 | 4969.8090 |

### GraphDrone Router Deltas

- vs TabR: `-499.3146` mean RMSE delta
- vs TabM: `-113.1123` mean RMSE delta
- vs TabPFN: `-61.8557` mean RMSE delta

## houses

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_foundation_router | 0.1974 | 0.0048 | 0.1918 | 0.1293 | 0.8797 | 3 |
| GraphDrone_foundation_crossfit | 0.1974 | 0.0048 | 0.1933 | 0.1294 | 0.8796 | 3 |
| GraphDrone_router | 0.1983 | 0.0048 | 0.1927 | 0.1300 | 0.8786 | 3 |
| GraphDrone_crossfit | 0.1983 | 0.0049 | 0.1942 | 0.1300 | 0.8786 | 3 |
| GraphDrone_trust_gate | 0.1988 | 0.0049 | 0.1941 | 0.1297 | 0.8780 | 3 |
| GraphDrone_FULL | 0.2028 | 0.0051 | 0.1976 | 0.1316 | 0.8730 | 3 |
| GraphDrone_FOUNDATION | 0.2037 | 0.0046 | 0.1988 | 0.1332 | 0.8718 | 3 |
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

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FOUNDATION | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_foundation_crossfit | GraphDrone_foundation_router | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_trust_gate | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.3201 | 0.2041 | 0.2026 | 0.2604 | 0.4632 | 0.1996 | 0.1989 | 0.1989 | 0.2476 | 0.1995 | 0.2483 | 0.2001 | 0.2488 | 0.2292 | 0.2041 | 0.2038 |
| 0 | 1 | 0.3200 | 0.1989 | 0.1978 | 0.2492 | 0.4562 | 0.1929 | 0.1921 | 0.1920 | 0.2412 | 0.1930 | 0.2416 | 0.1934 | 0.2422 | 0.2254 | 0.1989 | 0.2021 |
| 0 | 2 | 0.3210 | 0.2082 | 0.2081 | 0.2585 | 0.4666 | 0.2024 | 0.2013 | 0.2013 | 0.2498 | 0.2024 | 0.2506 | 0.2029 | 0.2511 | 0.2325 | 0.2082 | 0.2084 |

### GraphDrone Router Deltas

- vs TabR: `-0.0065` mean RMSE delta
- vs TabM: `-0.0308` mean RMSE delta
- vs TabPFN: `-0.0055` mean RMSE delta

## miami_housing

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_FOUNDATION | 80132.0483 | 2656.8932 | 82974.2079 | 38061.4987 | 0.9364 | 3 |
| TabPFN | 80132.0483 | 2656.8932 | 82974.2079 | 38061.4987 | 0.9364 | 3 |
| GraphDrone_foundation_crossfit | 80721.8209 | 2059.2225 | 83016.6375 | 38287.3398 | 0.9355 | 3 |
| GraphDrone_foundation_router | 80750.9044 | 1958.9572 | 83166.6009 | 38292.1549 | 0.9355 | 3 |
| GraphDrone_FULL | 81717.8403 | 1269.6404 | 84734.2748 | 38763.9349 | 0.9339 | 3 |
| GraphDrone_trust_gate | 82255.5651 | 1036.8860 | 86116.0151 | 40973.3919 | 0.9330 | 3 |
| GraphDrone_crossfit | 82255.5744 | 1036.8899 | 86116.0348 | 40973.5977 | 0.9330 | 3 |
| GraphDrone_router | 82489.3723 | 1092.9329 | 85131.1983 | 41502.4544 | 0.9326 | 3 |
| TabR | 89187.9895 | 1395.7350 | 91170.2700 | 44290.7994 | 0.9211 | 3 |
| TabM | 89375.8604 | 1522.5388 | 92708.7826 | 44716.8203 | 0.9209 | 3 |
| GraphDrone_uniform | 101331.1553 | 2451.2787 | 106555.5074 | 49587.0117 | 0.8983 | 3 |
| GraphDrone_sigma2 | 101938.0481 | 2257.4117 | 107199.7437 | 49504.8464 | 0.8971 | 3 |
| GraphDrone_LOWRANK | 145912.5008 | 2818.8744 | 148877.5030 | 73968.3906 | 0.7890 | 3 |
| GraphDrone_gora | 147841.9300 | 4769.1877 | 155356.2992 | 67902.9688 | 0.7832 | 3 |
| GraphDrone_GEO | 158960.0919 | 5266.2944 | 162012.7946 | 78921.2708 | 0.7494 | 3 |
| GraphDrone_DOMAIN | 164749.1595 | 3927.8808 | 172499.0635 | 85071.9271 | 0.7311 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FOUNDATION | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_foundation_crossfit | GraphDrone_foundation_router | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_trust_gate | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 162212.4095 | 82571.1899 | 82529.3618 | 160717.6099 | 146924.7304 | 82978.0146 | 82571.1434 | 82570.7838 | 142740.9061 | 83475.6112 | 104544.5632 | 82977.9930 | 104158.3435 | 91047.6510 | 82571.1899 | 88396.1174 |
| 0 | 1 | 169273.5938 | 80524.0055 | 82369.4639 | 153039.7850 | 142727.2823 | 82721.2162 | 81091.5327 | 81004.3831 | 152189.6114 | 82678.1635 | 100656.3955 | 82721.2162 | 99799.1455 | 88068.8371 | 80524.0055 | 88368.2857 |
| 0 | 2 | 162761.4751 | 77300.9495 | 80254.6954 | 163122.8806 | 148085.4897 | 81067.4923 | 78502.7865 | 78677.5463 | 148595.2725 | 81314.3424 | 100613.1856 | 81067.4860 | 100035.9769 | 89011.0932 | 77300.9495 | 90799.5653 |

### GraphDrone Router Deltas

- vs TabR: `-6698.6171` mean RMSE delta
- vs TabM: `-6886.4881` mean RMSE delta
- vs TabPFN: `+2357.3240` mean RMSE delta

## used_fiat_500

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_FOUNDATION | 732.0692 | 34.4613 | 787.4501 | 549.0113 | 0.8574 | 3 |
| TabPFN | 732.0692 | 34.4613 | 787.4501 | 549.0113 | 0.8574 | 3 |
| GraphDrone_FULL | 735.3961 | 39.9691 | 786.4971 | 548.1123 | 0.8561 | 3 |
| GraphDrone_foundation_router | 735.7657 | 37.0676 | 775.6378 | 555.8378 | 0.8560 | 3 |
| GraphDrone_foundation_crossfit | 735.7735 | 41.9432 | 782.3535 | 555.7939 | 0.8559 | 3 |
| GraphDrone_trust_gate | 739.6910 | 41.1500 | 783.4517 | 552.9332 | 0.8544 | 3 |
| GraphDrone_GEO | 741.9727 | 35.8773 | 787.0529 | 556.5066 | 0.8535 | 3 |
| GraphDrone_LOWRANK | 753.0895 | 24.8532 | 817.6420 | 578.2295 | 0.8491 | 3 |
| GraphDrone_router | 757.3655 | 36.3929 | 797.0409 | 579.1076 | 0.8474 | 3 |
| GraphDrone_crossfit | 759.5949 | 38.3780 | 794.3000 | 581.3964 | 0.8465 | 3 |
| TabM | 767.5875 | 28.4806 | 832.7882 | 586.1259 | 0.8433 | 3 |
| TabR | 780.3606 | 34.7720 | 834.0192 | 590.8100 | 0.8380 | 3 |
| GraphDrone_uniform | 838.7982 | 24.0559 | 887.7429 | 652.9824 | 0.8128 | 3 |
| GraphDrone_sigma2 | 855.0434 | 34.5822 | 904.9136 | 654.0017 | 0.8055 | 3 |
| GraphDrone_gora | 941.0032 | 74.2070 | 996.4440 | 685.8877 | 0.7636 | 3 |
| GraphDrone_DOMAIN | 1770.4928 | 28.0985 | 1857.5930 | 1425.4213 | 0.1659 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FOUNDATION | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_foundation_crossfit | GraphDrone_foundation_router | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_trust_gate | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 1754.5074 | 694.5983 | 691.9654 | 706.4424 | 731.3585 | 716.0271 | 692.7407 | 697.6236 | 917.7394 | 715.7033 | 818.6369 | 692.3974 | 823.6744 | 735.7328 | 694.5983 | 740.3116 |
| 0 | 1 | 1802.9370 | 739.2060 | 743.5889 | 741.2885 | 747.7220 | 788.3971 | 738.0451 | 738.0173 | 1024.0549 | 782.9548 | 887.4542 | 759.3617 | 826.1824 | 776.4365 | 739.2060 | 797.9068 |
| 0 | 2 | 1754.0341 | 762.4034 | 770.6340 | 778.1872 | 780.1878 | 774.3606 | 776.5348 | 771.6562 | 881.2154 | 773.4385 | 859.0391 | 767.3139 | 866.5377 | 790.5931 | 762.4034 | 802.8635 |

### GraphDrone Router Deltas

- vs TabR: `-22.9951` mean RMSE delta
- vs TabM: `-10.2219` mean RMSE delta
- vs TabPFN: `+25.2963` mean RMSE delta

## wine_quality

| Model | Mean Test RMSE | Std | Mean Val RMSE | Mean Test MAE | Mean Test R2 | Runs |
|---|---:|---:|---:|---:|---:|---:|
| GraphDrone_FOUNDATION | 0.6465 | 0.0182 | 0.6459 | 0.4980 | 0.4514 | 3 |
| TabPFN | 0.6465 | 0.0182 | 0.6459 | 0.4980 | 0.4514 | 3 |
| GraphDrone_foundation_crossfit | 0.6476 | 0.0173 | 0.6490 | 0.4981 | 0.4496 | 3 |
| GraphDrone_foundation_router | 0.6477 | 0.0173 | 0.6439 | 0.4980 | 0.4495 | 3 |
| GraphDrone_trust_gate | 0.6492 | 0.0146 | 0.6482 | 0.4920 | 0.4467 | 3 |
| GraphDrone_FULL | 0.6494 | 0.0143 | 0.6486 | 0.4912 | 0.4464 | 3 |
| GraphDrone_router | 0.6507 | 0.0142 | 0.6467 | 0.4961 | 0.4443 | 3 |
| GraphDrone_crossfit | 0.6509 | 0.0147 | 0.6510 | 0.4966 | 0.4439 | 3 |
| TabR | 0.6598 | 0.0162 | 0.6512 | 0.4970 | 0.4287 | 3 |
| TabM | 0.6748 | 0.0193 | 0.6703 | 0.5152 | 0.4024 | 3 |
| GraphDrone_GEO | 0.6816 | 0.0222 | 0.6755 | 0.5199 | 0.3903 | 3 |
| GraphDrone_uniform | 0.6862 | 0.0168 | 0.6883 | 0.5389 | 0.3821 | 3 |
| GraphDrone_sigma2 | 0.6863 | 0.0167 | 0.6882 | 0.5391 | 0.3820 | 3 |
| GraphDrone_gora | 0.6969 | 0.0167 | 0.6953 | 0.5455 | 0.3628 | 3 |
| GraphDrone_LOWRANK | 0.7685 | 0.0218 | 0.7745 | 0.6134 | 0.2253 | 3 |
| GraphDrone_DOMAIN | 0.7955 | 0.0143 | 0.8008 | 0.6270 | 0.1697 | 3 |

### Per Fold Test RMSE

| repeat | fold | GraphDrone_DOMAIN | GraphDrone_FOUNDATION | GraphDrone_FULL | GraphDrone_GEO | GraphDrone_LOWRANK | GraphDrone_crossfit | GraphDrone_foundation_crossfit | GraphDrone_foundation_router | GraphDrone_gora | GraphDrone_router | GraphDrone_sigma2 | GraphDrone_trust_gate | GraphDrone_uniform | TabM | TabPFN | TabR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.7795 | 0.6365 | 0.6430 | 0.6670 | 0.7450 | 0.6419 | 0.6357 | 0.6362 | 0.6832 | 0.6419 | 0.6714 | 0.6426 | 0.6712 | 0.6648 | 0.6365 | 0.6502 |
| 0 | 1 | 0.8071 | 0.6675 | 0.6658 | 0.7072 | 0.7879 | 0.6679 | 0.6675 | 0.6675 | 0.7155 | 0.6671 | 0.7043 | 0.6659 | 0.7043 | 0.6970 | 0.6675 | 0.6785 |
| 0 | 2 | 0.7998 | 0.6355 | 0.6394 | 0.6706 | 0.7726 | 0.6430 | 0.6397 | 0.6393 | 0.6918 | 0.6432 | 0.6832 | 0.6392 | 0.6833 | 0.6624 | 0.6355 | 0.6506 |

### GraphDrone Router Deltas

- vs TabR: `-0.0091` mean RMSE delta
- vs TabM: `-0.0240` mean RMSE delta
- vs TabPFN: `+0.0042` mean RMSE delta

