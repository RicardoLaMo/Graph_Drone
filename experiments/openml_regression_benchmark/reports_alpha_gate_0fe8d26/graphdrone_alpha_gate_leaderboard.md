# GraphDrone Alpha-Gate Leaderboard

| Dataset | GraphDrone_FULL | GraphDrone_router | GraphDrone_crossfit | GraphDrone_trust_gate | TabPFN | TabR | TabM | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| diamonds | 528.8255 | 539.5695 | 538.8079 | 528.2833 | 523.7256 | 546.1741 | 532.0915 | TabPFN |
| houses | 0.2028 | 0.1983 | 0.1983 | 0.1987 | 0.2037 | 0.2048 | 0.2290 | GraphDrone_router |
| miami_housing | 81717.8403 | 82228.8692 | 82225.1277 | 82091.7011 | 80132.0483 | 89187.9895 | 89375.8604 | TabPFN |
| wine_quality | 0.6494 | 0.6504 | 0.6509 | 0.6493 | 0.6465 | 0.6598 | 0.6748 | TabPFN |
| healthcare_insurance_expenses | 4510.7199 | 4480.1349 | 4483.1113 | 4499.3093 | 4540.5938 | 4978.0528 | 4591.8505 | GraphDrone_router |
| concrete_compressive_strength | 4.3695 | 4.5410 | 4.5700 | 4.3565 | 4.3306 | 5.4709 | 5.1452 | TabPFN |
| airfoil_self_noise | 1.3248 | 1.3435 | 1.3467 | 1.3109 | 1.2742 | 1.6993 | 1.4153 | TabPFN |
| used_fiat_500 | 735.3961 | 756.5451 | 757.3033 | 736.0660 | 732.0692 | 780.3606 | 767.5875 | TabPFN |

Notes:
- Metrics are mean test RMSE over folds `0,1,2` on commit `0fe8d26` using the isolated rerun roots under each dataset worktree.
- Lower RMSE is better.
