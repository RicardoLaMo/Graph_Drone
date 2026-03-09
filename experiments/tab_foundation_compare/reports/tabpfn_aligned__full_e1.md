# TabPFN Aligned California Report (Full Train, 1 Estimator)

## Result

- TabPFN_on_our_split_full_e1: test RMSE `0.3891`
- val RMSE `0.4111`
- train samples used `14448`
- n_estimators `1`
- duration seconds `71.0`

## Comparison

- vs TabR_on_our_split `0.3829`: delta `+0.0062`
- vs TabM_on_our_split `0.4290`: delta `-0.0399`
- vs A6f `0.4063`: delta `-0.0172`

## Notes

- This run uses the full aligned California train split with the current TabPFN CPU override.
- On this split, `n_estimators=1` outperformed the default `n_estimators=8` full-train run.
