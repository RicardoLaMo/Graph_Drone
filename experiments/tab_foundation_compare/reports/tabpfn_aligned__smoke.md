# TabPFN Aligned California Report

## Result

- TabPFN_on_our_split: test RMSE `0.4912`
- val RMSE `0.5415`
- train samples used `512`
- n_estimators `2`

## Comparison

- vs TabR_on_our_split `0.3829`: delta `+0.1083`
- vs TabM_on_our_split `0.4290`: delta `+0.0622`
- vs A6f `0.4063`: delta `+0.0849`

## Notes

- This run uses the aligned California split and feature edits from the foundation comparison branch.
- CPU-only TabPFN runs above 1000 rows require the current package override; this run records the exact train cap used.
