# C2 Decoder Challenger

Aligned California split, decoder-only comparison on top of a TabM-style ensemble backbone.

## Repo References

- `B1_HGBR`: RMSE `0.4430`
- `G2_GoRA_v1_ref`: RMSE `0.4546`
- `CA_v35b`: RMSE `0.4762`
- `HR_v4_headgated_diverse`: RMSE `0.4722`

## Results

- `C2a_TabM_mean_heads`: val RMSE `0.6385`, test RMSE `0.5820`, best epoch `7`, duration `12.2s`
- `C2b_TabM_gated_heads`: val RMSE `0.6298`, test RMSE `0.5879`, best epoch `7`, duration `12.2s`
  gate entropy `3.3961`, top-1 gate mass `0.0580`, head prediction std `0.7084`

## Interpretation

- Decoder gating delta on test RMSE: `-0.0059` (positive means the gated decoder is better).
