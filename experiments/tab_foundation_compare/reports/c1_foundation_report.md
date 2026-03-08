# C1 Foundation Comparison

## Current Ranking On The Repo California Protocol

1. `TabR_on_our_split`: RMSE `0.3829`
2. `TabM_on_our_split`: RMSE `0.4290`
3. `B1_HGBR`: RMSE `0.4430`
4. `G2_GoRA_v1_ref`: RMSE `0.4546`
5. `HR_v4_headgated_diverse`: RMSE `0.4722`
6. `CA_v35b`: RMSE `0.4762`

## Main Findings

- `TabR` is the clear California foundation winner.
- `TabM` is also a serious baseline: it beats every current repo California model family reference listed above.
- The repo’s structured-routing California variants are still not cashing out enough predictive value to compete with strong generic tabular foundations.

## Implication For Challenger Work

`C2` should not be framed as “beat a weak baseline.” It should be framed as:

- can a decoder-side change recover some of the gap to strong tabular foundations?
- or did those foundations already solve the readout/credit-assignment problem more effectively than our routed models?

## Recommendation

Use `TabR_on_our_split` as the primary California champion.

Use `TabM_on_our_split` as the cleaner decoder-hypothesis testbed because it exposes ensemble heads directly and is easier to modify without rewriting retrieval internals.
