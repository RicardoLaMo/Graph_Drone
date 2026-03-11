# Registered Portfolio Signal vs Noise

- portfolio_id: `graphdrone_registered_signal_noise_2026_03_11`
- adaptive_prefix: `router`
- n_datasets: `9`
- datasets with positive best-pair gain vs FULL: `0.667`
- datasets with positive competition-noise gain: `1.000`

| Dataset | Best View | Verdict | CompNoise Gain vs Router | Best Pair Gain vs FULL | Capture Gap vs Fixed | Fixed Mode | Quality Mode |
|---|---|---|---:|---:|---:|---|---|
| california_housing_openml | GEO | useful_signal_obscured_by_competition | 0.0011 | 0.0110 | -0.0201 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |
| houses | GEO | useful_signal_obscured_by_competition | 0.0002 | 0.0047 | 0.0078 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |
| miami_housing | GEO | competition_noise_plus_weak_expert | 633.8337 | -12.6599 | -0.0569 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |
| diamonds | LOWRANK | competition_noise_plus_weak_expert | 8.7673 | -0.9587 | -0.0466 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |
| healthcare_insurance_expenses | GEO | useful_signal_obscured_by_competition | 1.0427 | 28.9080 | 0.1283 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |
| concrete_compressive_strength | GEO | useful_signal_obscured_by_competition | 0.1674 | 0.0051 | -0.0130 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |
| airfoil_self_noise | LOWRANK | useful_signal_obscured_by_competition | 0.0486 | 0.0242 | 0.0056 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |
| wine_quality | DOMAIN | competition_noise_plus_weak_expert | 0.0010 | -0.0005 | -0.0012 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |
| used_fiat_500 | GEO | useful_signal_obscured_by_competition | 21.1500 | 2.1890 | 0.0279 | derived_from_val_mean_weights | derived_sigma2_plus_mean_j |

## Dataset Notes

### california_housing_openml

- best view: `GEO`
- verdict: `useful_signal_obscured_by_competition`
- competition-noise gain vs full router: `0.001060 ± 0.000811`
- best pair gain vs FULL expert: `0.010995 ± 0.005288`
- best-view capture gap vs fixed: `-0.020058 ± 0.001318`
- best view counts: `{"GEO": 3}`

### houses

- best view: `GEO`
- verdict: `useful_signal_obscured_by_competition`
- competition-noise gain vs full router: `0.000168 ± 0.000097`
- best pair gain vs FULL expert: `0.004685 ± 0.001123`
- best-view capture gap vs fixed: `0.007758 ± 0.006037`
- best view counts: `{"GEO": 3}`
- stability probe capture gap vs fixed: `0.009151 ± 0.009256`

### miami_housing

- best view: `GEO`
- verdict: `competition_noise_plus_weak_expert`
- competition-noise gain vs full router: `633.833703 ± 334.636230`
- best pair gain vs FULL expert: `-12.659860 ± 165.209756`
- best-view capture gap vs fixed: `-0.056870 ± 0.039507`
- best view counts: `{"DOMAIN": 1, "GEO": 2}`

### diamonds

- best view: `LOWRANK`
- verdict: `competition_noise_plus_weak_expert`
- competition-noise gain vs full router: `8.767268 ± 1.053286`
- best pair gain vs FULL expert: `-0.958749 ± 0.938074`
- best-view capture gap vs fixed: `-0.046598 ± 0.026021`
- best view counts: `{"LOWRANK": 3}`

### healthcare_insurance_expenses

- best view: `GEO`
- verdict: `useful_signal_obscured_by_competition`
- competition-noise gain vs full router: `1.042700 ± 14.474463`
- best pair gain vs FULL expert: `28.907983 ± 14.292104`
- best-view capture gap vs fixed: `0.128282 ± 0.122708`
- best view counts: `{"DOMAIN": 1, "GEO": 2}`

### concrete_compressive_strength

- best view: `GEO`
- verdict: `useful_signal_obscured_by_competition`
- competition-noise gain vs full router: `0.167389 ± 0.098709`
- best pair gain vs FULL expert: `0.005093 ± 0.037896`
- best-view capture gap vs fixed: `-0.013033 ± 0.019753`
- best view counts: `{"DOMAIN": 1, "GEO": 1, "LOWRANK": 1}`

### airfoil_self_noise

- best view: `LOWRANK`
- verdict: `useful_signal_obscured_by_competition`
- competition-noise gain vs full router: `0.048649 ± 0.030986`
- best pair gain vs FULL expert: `0.024168 ± 0.041447`
- best-view capture gap vs fixed: `0.005560 ± 0.012240`
- best view counts: `{"LOWRANK": 3}`

### wine_quality

- best view: `DOMAIN`
- verdict: `competition_noise_plus_weak_expert`
- competition-noise gain vs full router: `0.001013 ± 0.000847`
- best pair gain vs FULL expert: `-0.000474 ± 0.001302`
- best-view capture gap vs fixed: `-0.001184 ± 0.008240`
- best view counts: `{"DOMAIN": 2, "GEO": 1}`

### used_fiat_500

- best view: `GEO`
- verdict: `useful_signal_obscured_by_competition`
- competition-noise gain vs full router: `21.149960 ± 11.417333`
- best pair gain vs FULL expert: `2.188965 ± 1.581874`
- best-view capture gap vs fixed: `0.027921 ± 0.007482`
- best view counts: `{"GEO": 2, "LOWRANK": 1}`

