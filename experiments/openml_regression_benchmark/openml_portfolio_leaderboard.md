# OpenML Portfolio Leaderboard

Mean test RMSE by dataset and model family.

Note:
- `california_housing_openml` is the matched split-sweep California row from the
  aligned OpenML California benchmark, not a 3-fold row from the OpenML portfolio lane.
- The other rows are the current 3-fold dataset means from the H200 benchmark runs.

| Dataset | GraphDrone_FULL | GraphDrone_router | GraphDrone_crossfit | TabPFN | TabR | TabM | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| california_housing_openml | 0.4013 | 0.3922 | 0.3922 | 0.4054 | 0.4038 | 0.4420 | GraphDrone_router |
| houses | 0.2028 | 0.1983 | 0.1983 | 0.2037 | 0.2048 | 0.2290 | GraphDrone_crossfit |
| miami_housing | 81717.8403 | 82299.0846 | 82274.6784 | 80132.0483 | 89461.7639 | 89712.6471 | TabPFN |
| diamonds | 528.8255 | 538.5515 | 539.9814 | 523.7256 | 546.1741 | 532.0915 | TabPFN |
| healthcare_insurance_expenses | 4510.7199 | 4482.8546 | 4481.0295 | 4540.5938 | 4978.0528 | 4591.8505 | GraphDrone_crossfit |
| concrete_compressive_strength | 4.3695 | 4.5318 | 4.5791 | 4.3306 | 5.4709 | 5.1452 | TabPFN |
| airfoil_self_noise | 1.3248 | 1.3493 | 1.3440 | 1.2742 | 1.6993 | 1.4153 | TabPFN |
| used_fiat_500 | 735.3961 | 754.3571 | 755.4140 | 732.0692 | 780.3606 | 767.5875 | TabPFN |
| wine_quality | 0.6494 | 0.6509 | 0.6504 | 0.6465 | 0.6598 | 0.6748 | TabPFN |
