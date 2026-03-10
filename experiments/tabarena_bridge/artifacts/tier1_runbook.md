# Tier 1 TabArena Runbook

## Scope

- Split seeds: `42, 43, 44, 45, 46`
- Baselines: `TabR, TabPFN_full, TabM, P0_full, P0_router`

## Datasets

### diamonds
- OpenML dataset id: `46923`
- OpenML task id: `363631`
- View family: `generic_numeric`
- Integration phase: `phase_2`
- Role: `non-geo-price-transfer`
- Why: Tests whether retrieval and routing gains generalize beyond geographic structure.

### houses
- OpenML dataset id: `46934`
- OpenML task id: `363678`
- View family: `geo_housing`
- Integration phase: `phase_1`
- Role: `housing-anchor`
- Why: Closest general housing regression anchor from the TabArena suite.

### miami_housing
- OpenML dataset id: `46942`
- OpenML task id: `363686`
- View family: `geo_housing`
- Integration phase: `phase_1`
- Role: `geo-transfer`
- Why: Closest housing-like geospatial regression benchmark in the TabArena suite.

## H200 Execution Order

1. Reproduce external baselines on all Tier 1 datasets.
2. Freeze split sweep outputs.
3. Add Graph_Drone custom adapters only after the baseline protocol is stable.
4. Start custom-model transfer on geo_housing datasets before generic_numeric datasets.
