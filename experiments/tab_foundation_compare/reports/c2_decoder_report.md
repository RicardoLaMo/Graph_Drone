# C2 Decoder Challenger

Aligned California split, decoder-only comparison on top of a TabM-style ensemble backbone.

## Repo References

- `B1_HGBR`: RMSE `0.4430`
- `G2_GoRA_v1_ref`: RMSE `0.4546`
- `CA_v35b`: RMSE `0.4762`
- `HR_v4_headgated_diverse`: RMSE `0.4722`

## Results

- `C2a_TabM_mean_heads`: val RMSE `0.4870`, test RMSE `0.4578`, best epoch `72`, duration `406.5s`
- `C2b_TabM_gated_heads`: val RMSE `0.4890`, test RMSE `0.4614`, best epoch `63`, duration `290.5s`
  gate entropy `3.2041`, top-1 gate mass `0.1067`, head prediction std `1.3489`

## Interpretation

- Decoder gating delta on test RMSE: `-0.0036` (positive means the gated decoder is better).
