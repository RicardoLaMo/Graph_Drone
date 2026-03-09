# TabPFN Aligned California Report (Full Train)

## Result

- TabPFN_on_our_split_full: test RMSE `0.3927`
- val RMSE `0.4202`
- train samples used `14448`
- n_estimators `8`
- duration seconds `480.1`

## Comparison

- vs TabR_on_our_split `0.3829`: delta `+0.0098`
- vs TabM_on_our_split `0.4290`: delta `-0.0363`
- vs A6f `0.4063`: delta `-0.0136`

## Notes

- This run uses the full aligned California train split with the current TabPFN CPU override.
- It is the fairer comparison to full-train TabR/TabM than the older 8k-capped run.
