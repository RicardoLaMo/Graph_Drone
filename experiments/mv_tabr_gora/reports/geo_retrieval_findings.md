# Geo Retrieval Findings

## Setup

This branch tests whether train-only pseudo-community structure derived from
raw California Housing latitude/longitude can improve `A6f` retrieval without
changing the model architecture.

Variants:

- `A6f_raw`: canonical `A6f` on the standard MV-TabR-GoRA bundle
- `G1_geo_bias96`: same-segment weight bias on `FULL` and `GEO`
- `G2_geo_poolmix96`: replace half of `FULL` and `GEO` neighbors with nearest
  same-segment training points from a `KMeans(96)` pseudo-community fit on raw
  lat/lon
- `G3_random_poolmix96`: same pool-mix budget as `G2`, but random training
  neighbors instead of same-segment neighbors

## Results

### Smoke

- `A6f_raw`: val `0.8331`, test `0.9375`
- `G2_geo_poolmix96`: val `0.7864`, test `0.8846`
- `G3_random_poolmix96`: val `0.7988`, test `0.8959`

### Full paired runs

- `A6f_raw`: val `0.4471`, test `0.4139`
- `G2_geo_poolmix96`: val `0.4270`, test `0.4091`
  - delta vs paired raw: `+0.0048` RMSE improvement
- `A6f_raw` in random-control pair: val `0.4507`, test `0.4112`
- `G3_random_poolmix96`: val `0.4395`, test `0.4197`
  - delta vs paired raw: `-0.0085` RMSE regression

Reference:

- canonical parent-branch `A6f`: test `0.4063`

## Interpretation

- Geo pool-mix improves the paired `A6f` run on both validation and test.
- Random pool-mix does not; it hurts test materially.
- So the gain is not just “mixing in extra neighbors” or diversity
  regularization. The lat/lon-derived pseudo-community structure is doing real
  retrieval work.
- The gain is still small and does not yet exceed the canonical `A6f=0.4063`,
  so this is a promising retrieval prior, not a validated new champion.

## Next step

The strongest next move is to combine this geo-community candidate structure
with the best retrieval-side improvement from later branches, rather than push
it into the decoder.
