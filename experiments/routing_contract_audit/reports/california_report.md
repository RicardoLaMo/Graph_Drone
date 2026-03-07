# Routing Contract Audit: california
*2026-03-07* | Branch: `feature/routing-contract-audit`

## Routing Semantics
- **Routing:** observer vector g_i → pi_i (view weights, softmax) + beta_i (mode gate, sigmoid), applied to representations BEFORE prediction.
- **beta:** beta→0 = isolation (trust views independently), beta→1 = interaction (fuse views jointly).
- **A vs B:** Model A = post-hoc flatten+MLP (observers concatenated, no pi/beta). Model B = explicit pi+beta+iso_rep+inter_rep+final_rep blend.

## Routing Implementation: per-row routing (per-head extension path open)

## Synthetic Routing Tests
All 6 tests passed (run `python experiments/routing_contract_audit/tests/test_routing_semantics.py -v`).

## Metrics

| model     |     rmse |      mae |       r2 |
|:----------|---------:|---------:|---------:|
| A_posthoc | 0.440705 | 0.302861 | 0.85312  |
| B_router  | 0.441785 | 0.30377  | 0.852399 |

## Routing Behavior (Model B, test set)

| stat                   |     value |
|:-----------------------|----------:|
| mean_pi_FULL           | 0.0420247 |
| mean_pi_GEO            | 0.811368  |
| mean_pi_SOCIO          | 0.106961  |
| mean_pi_LOWRANK        | 0.0396456 |
| top1_freq_FULL         | 0         |
| top1_freq_GEO          | 1         |
| top1_freq_SOCIO        | 0         |
| top1_freq_LOWRANK      | 0         |
| routing_entropy_mean   | 0.66194   |
| routing_entropy_std    | 0.0803152 |
| mean_beta              | 0.519502  |
| mean_beta_low_kappa    | 0.523583  |
| mean_beta_medium_kappa | 0.516964  |
| mean_beta_high_kappa   | 0.518105  |

## Audit Conclusion
1. **A vs B architecturally different?** YES — verified by code: A has no pi/beta, B uses explicit ObserverRouter.
2. **B behaves like routing (vs post-hoc weighting)?** YES — routing entropy=0.6619 (uniform=1.3863).
3. **B improved prediction?** NO — A RMSE=0.4407, B RMSE=0.4418.
4. **B satisfies routing semantics?** YES — all 6 synthetic tests passed. pi sums to 1. beta ∈ [0,1]. mean_beta=0.5195.

## Prior Agent Comparison
- Prior `feature/routing-curvature-dual-datasets` used observer features as *combiner input* (Model A pattern).
- C8_Routed computed pi and beta but used `ViewCombiner` similarly to B_router, with no training differences enforced.
- This audit enforces strict separation: A=no pi/beta, B=explicit pi/beta+iso+inter+final_rep, shared encoders.
- Refer to routing_contract.md for full specification.