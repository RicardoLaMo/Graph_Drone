# TabArena Regression Comparison

**Date:** 2026-03-19  
**Scope:** regression-only TabArena evaluation  
**Models compared:** `GraphDrone_latest_regression` vs `TA-RealTabPFN-v2.5`  
**Fold:** `0`  
**Cache:** `eval/tabarena_regression_compare_2026_03_19/`

## Data Used

The comparison used the 13 regression tasks available in the TabArena benchmark slice:

`Another-Dataset-on-used-Fiat-500`, `Food_Delivery_Time`, `QSAR-TID-11`, `QSAR_fish_toxicity`,
`airfoil_self_noise`, `concrete_compressive_strength`, `diamonds`,
`healthcare_insurance_expenses`, `houses`, `miami_housing`, `physiochemical_protein`,
`superconductivity`, `wine_quality`

## Results

| Dataset | GraphDrone RMSE | TabPFN v2.5 RMSE | Winner | Relative Delta |
|---|---:|---:|---|---:|
| Another-Dataset-on-used-Fiat-500 | 1439.443809 | 692.650527 | TabPFN v2.5 | +107.816749% |
| Food_Delivery_Time | 7.619588 | 7.595655 | TabPFN v2.5 | +0.315087% |
| QSAR-TID-11 | 0.857118 | 0.846259 | TabPFN v2.5 | +1.283245% |
| QSAR_fish_toxicity | 0.898222 | 0.925607 | GraphDrone | -2.958593% |
| airfoil_self_noise | 0.986337 | 1.113641 | GraphDrone | -11.431368% |
| concrete_compressive_strength | 3.898590 | 4.192316 | GraphDrone | -7.006292% |
| diamonds | 503.748292 | 504.463388 | GraphDrone | -0.141754% |
| healthcare_insurance_expenses | 4099.626178 | 4099.697297 | GraphDrone | -0.001735% |
| houses | 0.202392 | 0.204034 | GraphDrone | -0.804768% |
| miami_housing | 80453.572416 | 80530.109898 | GraphDrone | -0.095042% |
| physiochemical_protein | 3.049768 | 3.068804 | GraphDrone | -0.620334% |
| superconductivity | 9.243440 | 9.453658 | GraphDrone | -2.223665% |
| wine_quality | 0.617222 | 0.628944 | GraphDrone | -1.863839% |

## Aggregate

- GraphDrone won `10/13` datasets.
- TabPFN v2.5 won `3/13` datasets.
- Mean relative RMSE delta: `+6.33%` in favor of TabPFN v2.5, driven by the large miss on `Another-Dataset-on-used-Fiat-500`.

## Notes

- This is a fold-0 validation sweep, not a full multi-fold benchmark.
- The cache directory is intentionally left out of git; only the compact report is meant to be shared.
- The current benchmark script also pins baseline cache keys so future GraphDrone version bumps do not invalidate TabPFN reruns.
