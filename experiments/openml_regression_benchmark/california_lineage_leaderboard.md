# California Lineage Leaderboard

This note separates California Housing results by experiment family and protocol.

Do not compare rows across sections as if they were one unified benchmark unless the
split protocol, preprocessing, and model family are explicitly aligned.

## Aligned OpenML California, matched split sweep

Primary apples-to-apples comparison on OpenML California `did=44024` across split
seeds `42-46`.

Source:
- `tab-foundation-baseline/experiments/tab_foundation_compare/reports/foundation_split_sweep.md`
- `p0-view-router/experiments/tabpfn_view_router/reports/p0_openml_split_sweep.md`

| Model | Mean RMSE | Std | Notes |
|---|---:|---:|---|
| P0_router | 0.3922 | 0.0108 | best GraphDrone-family result in matched split sweep |
| P0_crossfit | 0.3922 | 0.0110 | effectively tied with router |
| P0_full | 0.4013 | 0.0081 | single-view FULL expert |
| TabR | 0.4038 | 0.0101 | aligned foundation baseline |
| TabPFN_full | 0.4054 | 0.0092 | aligned foundation baseline |
| TabM | 0.4420 | 0.0098 | aligned foundation baseline |

## Aligned OpenML California, fixed split multiseed

Same OpenML California dataset, fixed `split_seed=42`, varying model seeds.

Source:
- `p0-view-router/experiments/tabpfn_view_router/reports/p0_openml_multiseed.md`
- `tab-foundation-baseline/experiments/tab_foundation_compare/reports/tabpfn_aligned__full.md`

| Model | Mean RMSE | Std | Notes |
|---|---:|---:|---|
| P0_router | 0.3786 | 0.0015 | best fixed-split GraphDrone-family result |
| P0_crossfit | 0.3787 | 0.0013 | essentially tied with router |
| TabR_on_our_split | 0.3829 | - | fixed aligned split anchor |
| TabPFN_on_our_split | 0.3920 | - | fixed aligned split anchor |
| P0_full | 0.3925 | 0.0037 | single-view FULL expert |
| TabM_on_our_split | 0.4290 | - | fixed aligned split anchor |
| MV-TabR-GoRA A6f | 0.4063 | - | older branch-local California anchor |

## Older GoRA-Tabular California line

These are earlier California runs from the GoRA lineage. They are useful historical
anchors, but not directly interchangeable with the aligned OpenML benchmark above.

Source:
- `graphdrone-openml-regression/experiments/gora_tabular/reports/california_report.md`
- `graphdrone-openml-regression/experiments/gora_tabular/reports/california_v3_report.md`

| Model | RMSE | Source |
|---|---:|---|
| G0_Standard | 0.4403 | california_report.md |
| B1_HGBR | 0.4433 | california_report.md |
| G2_GoRA_v1 | 0.4546 | california_v3_report.md |
| G2_GoRA | 0.4847 | california_report.md |
| G7_RichCtx | 0.4929 | california_v3_report.md |
| G9_Teacher | 0.5099 | california_v3_report.md |
| G8_LabelCtx | 0.5169 | california_v3_report.md |
| G10_Full | 0.5209 | california_v3_report.md |

## Practical Read

- For current GraphDrone-vs-foundation comparison, use the matched split sweep.
- For fixed-split headline numbers, use the multiseed section.
- Treat the GoRA-Tabular section as historical lineage, not as the current benchmark.
