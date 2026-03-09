# TabPFN Aligned California Report

## Result

- TabPFN_on_our_split: test RMSE `0.4067`
- val RMSE `0.4374`
- train samples used `8000`
- n_estimators `8`

## Comparison

- vs TabR_on_our_split `0.3829`: delta `+0.0238`
- vs TabM_on_our_split `0.4290`: delta `-0.0223`
- vs A6f `0.4063`: delta `+0.0004`

## Notes

- This run uses the aligned California split and feature edits from the foundation comparison branch.
- CPU-only TabPFN runs above 1000 rows require the current package override; this run records the exact train cap used.
