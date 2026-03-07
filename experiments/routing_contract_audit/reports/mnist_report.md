# Routing Contract Audit: mnist
*2026-03-07* | Branch: `feature/routing-contract-audit`

## Routing Semantics
- **Routing:** observer vector g_i → pi_i (view weights, softmax) + beta_i (mode gate, sigmoid), applied to representations BEFORE prediction.
- **beta:** beta→0 = isolation (trust views independently), beta→1 = interaction (fuse views jointly).
- **A vs B:** Model A = post-hoc flatten+MLP (observers concatenated, no pi/beta). Model B = explicit pi+beta+iso_rep+inter_rep+final_rep blend.

## Routing Implementation: per-row routing (per-head extension path open)

## Synthetic Routing Tests
All 6 tests passed (run `python experiments/routing_contract_audit/tests/test_routing_semantics.py -v`).

## Metrics

| model     |   accuracy |   macro_f1 |   log_loss |
|:----------|-----------:|-----------:|-----------:|
| A_posthoc |   0.946667 |   0.946316 |   0.215045 |
| B_router  |   0.937333 |   0.93701  |   0.235243 |

## Routing Behavior (Model B, test set)

| stat                   |    value |
|:-----------------------|---------:|
| mean_pi_FULL           | 0.236676 |
| mean_pi_BLOCK          | 0.148469 |
| mean_pi_PCA            | 0.614854 |
| top1_freq_FULL         | 0.142667 |
| top1_freq_BLOCK        | 0        |
| top1_freq_PCA          | 0.857333 |
| routing_entropy_mean   | 0.833895 |
| routing_entropy_std    | 0.152186 |
| mean_beta              | 0.673454 |
| mean_beta_low_kappa    | 0.71735  |
| mean_beta_medium_kappa | 0.643486 |
| mean_beta_high_kappa   | 0.65916  |

## Audit Conclusion
1. **A vs B architecturally different?** YES — verified by code: A has no pi/beta, B uses explicit ObserverRouter.
2. **B behaves like routing (vs post-hoc weighting)?** YES — routing entropy=0.8339 (uniform=1.0986).
3. **B improved prediction?** NO — A Accuracy=0.9467, B Accuracy=0.9373.
4. **B satisfies routing semantics?** YES — all 6 synthetic tests passed. pi sums to 1. beta ∈ [0,1]. mean_beta=0.6735.

## Prior Agent Comparison
- Prior `feature/routing-curvature-dual-datasets` used observer features as *combiner input* (Model A pattern).
- C8_Routed computed pi and beta but used `ViewCombiner` similarly to B_router, with no training differences enforced.
- This audit enforces strict separation: A=no pi/beta, B=explicit pi/beta+iso+inter+final_rep, shared encoders.
- Refer to routing_contract.md for full specification.