# P0 TabPFN View Router

Goal: test whether GoRA-style view routing adds value on top of a stronger tabular foundation than TabM.

Scope:
- California Housing only
- aligned split: 70/15/15, seed 42, `log1p` on cols 2 and 4
- four view experts: `FULL`, `GEO`, `SOCIO`, `LOWRANK`
- `TabPFNRegressor` per view
- compare:
  - `FULL` only
  - uniform view mean
  - inverse-`sigma2_v` routing over view predictions
  - learned router from `{sigma2_v, J_flat, mean_J}` to view weights

Rules:
- do not change TabPFN architecture
- no retrieval rewrite in P0
- keep router semantics narrow and observer-driven
- train router only on a split of validation data; evaluate on held-out test

Success:
- any routed view mixture beating `FULL` TabPFN on the aligned split
- especially if it narrows the gap to the known full TabPFN reference
