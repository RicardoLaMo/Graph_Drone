# TabPFN Aligned California Report

## Result

- TabPFN_on_our_split: test RMSE `0.3920`
- val RMSE `0.4199`
- train samples used `14448`
- n_estimators `8`

## Comparison

- vs TabR_on_our_split `0.3829`: delta `+0.0091`
- vs TabM_on_our_split `0.4290`: delta `-0.0370`
- vs A6f `0.4063`: delta `-0.0143`

## Notes

- This run uses the aligned California split and feature edits from the foundation comparison branch.
- Runtime: device `cuda`; train cap `14448`.
- This run uses the CUDA-enabled TabPFN path on the shared H200 environment.
