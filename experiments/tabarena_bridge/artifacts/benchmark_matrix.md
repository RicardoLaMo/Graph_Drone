# Graph_Drone TabArena Benchmark Matrix

- Source suite: `tabarena-v0.1`
- OpenML suite id: `457`
- Primary metric: `rmse`
- Split seeds: `42, 43, 44, 45, 46`

| Tier | Dataset | OpenML DID | OpenML TID | Rows | Features | View family | Phase | Role | Why |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | diamonds | 46923 | 363631 | 53940 | 10 | generic_numeric | phase_2 | non-geo-price-transfer | Tests whether retrieval and routing gains generalize beyond geographic structure. |
| 1 | houses | 46934 | 363678 | 20640 | 9 | geo_housing | phase_1 | housing-anchor | Closest general housing regression anchor from the TabArena suite. |
| 1 | miami_housing | 46942 | 363686 | 13776 | 16 | geo_housing | phase_1 | geo-transfer | Closest housing-like geospatial regression benchmark in the TabArena suite. |
| 2 | airfoil_self_noise | 46904 | 363612 | 1503 | 6 | generic_numeric | phase_2 | small-numeric-sanity | Another compact regression task useful for quick sweeps. |
| 2 | concrete_compressive_strength | 46917 | 363625 | 1030 | 9 | generic_numeric | phase_2 | small-numeric-sanity | Compact numeric regression benchmark for early pipeline validation. |
| 2 | healthcare_insurance_expenses | 46931 | 363675 | 1338 | 7 | generic_numeric | phase_2 | small-fast-regression | Small regression task for fast local ablations before GPU expansion. |
| 2 | wine_quality | 46964 | 363708 | 6497 | 13 | generic_numeric | phase_2 | midscale-regression | Mid-size regression with mixed signal quality and broad benchmark use. |
| 3 | Food_Delivery_Time | 46928 | 363672 | 45451 | 10 | generic_numeric | phase_3 | large-regression | Larger-scale regression benchmark for H200 expansion. |
| 3 | physiochemical_protein | 46949 | 363693 | 45730 | 10 | generic_numeric | phase_3 | large-regression | Large regression task to stress retrieval and foundation baselines. |
| 3 | superconductivity | 46961 | 363705 | 21263 | 82 | generic_numeric | phase_3 | high-dimensional-regression | High-dimensional regression benchmark for testing foundation-model transfer. |
