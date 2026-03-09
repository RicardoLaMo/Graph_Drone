# Geo Retrieval Multi-Seed Summary

## Setup

This report evaluates whether the `geo_poolmix96` retrieval prior is a stable
improvement over the paired `A6f_raw` baseline inside the
`codex/mv-tabr-gora-geo-retrieval` branch.

Both variants use:

- identical model architecture (`A6f`)
- identical total neighbour budget `K=24`
- train-only referenced candidate pools
- the same split seed as the base California setup

The only difference is retrieval structure:

- `A6f_raw`: standard per-view kNN
- `G2_geo_poolmix96`: half of the `FULL` and `GEO` neighbours are replaced by
  nearest same-segment training points from a `KMeans(96)` pseudo-community fit
  on raw latitude/longitude

## Results

| Seed | A6f raw val | A6f raw test | Geo poolmix val | Geo poolmix test | Test gain |
|---|---:|---:|---:|---:|---:|
| 0 | 0.4350 | 0.4315 | 0.4244 | 0.4260 | +0.0055 |
| 1 | 0.4336 | 0.4294 | 0.4199 | 0.4250 | +0.0043 |
| 2 | 0.4409 | 0.4278 | 0.4328 | 0.4211 | +0.0067 |

Aggregate:

- `A6f_raw` mean test RMSE: `0.4295 ± 0.0015`
- `G2_geo_poolmix96` mean test RMSE: `0.4241 ± 0.0021`
- mean paired test gain: `+0.0055 ± 0.0010`
- mean paired validation gain: `+0.0108 ± 0.0023`

## Controls

- Weight-bias-only geo prior hurt.
- Random pool-mix hurt test relative to paired raw.
- So the gain is not explained by simply injecting more neighbours or more
  diverse neighbours.

## Interpretation

The multi-seed result supports a branch-local claim:

- latent geo/community structure inferred from latitude/longitude is a useful
  retrieval prior for `A6f`
- same-segment pool mixing is stronger than both raw retrieval and random
  pool-mixing inside this branch

The multi-seed result does **not** support the stronger claim that this branch
is now the overall champion:

- canonical parent-branch `A6f` remains better at `0.4063`
- this branch's geo-aware result is a promising retrieval prior, not yet a new
  best architecture

## Gemini Cross-Check

Gemini independently agreed that:

- the branch-local gain looks credible
- the interpretation should stay narrow
- the main remaining gap is that this retrieval prior has not yet been rebased
  onto the canonical parent-branch `A6f=0.4063` configuration
