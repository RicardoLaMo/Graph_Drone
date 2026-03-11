# Phase 0f Registered Portfolio Signal vs Noise

## Scope

This phase lifts the Phase 0e signal-vs-noise diagnosis from isolated cases to the registered regression portfolio.

Datasets covered:

- `california_housing_openml`
- `houses`
- `miami_housing`
- `diamonds`
- `healthcare_insurance_expenses`
- `concrete_compressive_strength`
- `airfoil_self_noise`
- `wine_quality`
- `used_fiat_500`

Primary output:

- [registered_signal_noise_portfolio.md](/home/wliu23/projects/Graph_Drone/.worktrees/phase0f-portfolio-signal-noise/experiments/openml_regression_benchmark/reports_phase0f_registered_portfolio/registered_signal_noise_portfolio.md)
- [registered_signal_noise_portfolio.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0f-portfolio-signal-noise/experiments/openml_regression_benchmark/reports_phase0f_registered_portfolio/registered_signal_noise_portfolio.json)

## Provenance

The portfolio reuses saved benchmark artifacts from sibling worktrees instead of rerunning all datasets.

Two older-artifact gaps were handled explicitly:

- fixed hedge:
  derived from mean validation router weights when a run did not persist `router_fixed_*`
- quality tensor:
  derived from `sigma2` plus `mean_J` when a run did not persist the full `quality_*` feature matrix

This is recorded per dataset in the portfolio summary. For this phase, every registered dataset uses:

- fixed mode: `derived_from_val_mean_weights`
- quality mode: `derived_sigma2_plus_mean_j`

So the portfolio results are reproducible and comparable, but the fixed-vs-adaptive and two-expert diagnostics should be read as a reconstruction layer over the saved runs, not as a fresh benchmark protocol.

## Key Results

Portfolio rollup:

- `6/9` datasets land in `useful_signal_obscured_by_competition`
- `3/9` datasets land in `competition_noise_plus_weak_expert`
- `9/9` datasets show positive competition-noise gain vs the dense full router
- `6/9` datasets show positive best-pair gain vs `FULL`

Best-view counts by dataset:

- `GEO`: `6`
- `LOWRANK`: `2`
- `DOMAIN`: `1`

High-signal dataset reads:

- `california_housing_openml`
  - best view: `GEO`
  - best pair gain vs `FULL`: `+0.0110`
  - competition-noise gain vs full router: `+0.0011`
  - capture gap vs fixed: negative
  - interpretation:
    there is real specialist signal, but most of the gain looks like pruning dense competition rather than strong row-level harvesting beyond a conservative hedge

- `houses`
  - best view: `GEO`
  - best pair gain vs `FULL`: `+0.0047`
  - competition-noise gain vs full router: `+0.00017`
  - 10-seed stability probe remains positive
  - interpretation:
    this is the cleanest current portfolio case for a real non-`FULL` specialist, but the adaptive edge over a simpler hedge is still small

- `miami_housing`
  - best view: mostly `GEO`
  - competition-noise gain vs full router is large
  - best pair still loses to `FULL`
  - interpretation:
    pruning weak competition helps, but the non-`FULL` expert is still not good enough

- `diamonds`
  - best view: `LOWRANK`
  - same broad pattern as Miami
  - interpretation:
    there is conditional structure, but it is not yet strong enough to justify routed wins over `FULL`

## Design Read

The portfolio strengthens the earlier claim, but in a narrower way than a headline “adaptive router wins” story.

What the evidence supports:

- The dense multi-view router has a real competition-noise problem.
- Prior views are worth keeping only when at least one specialist still shows positive marginal value after that competition is stripped away.
- `GEO` is the strongest and most frequent specialist across the current portfolio.
- `LOWRANK` and `DOMAIN` remain conditional specialists, not dead views.

What the evidence does not support:

- adding more views by default
- treating all views as equally valid experts
- claiming that row-level adaptivity is already the main mechanism everywhere

The practical conclusion is:

- current GraphDrone gains are mostly about getting the expert set right
- only secondarily about sophisticated row-level integration among many experts

## Research Support

The portfolio read aligns with the literature on diversity, correlated errors, and expert specialization:

- Brown et al. 2005, *Managing Diversity in Regression Ensembles*:
  diversity improves an ensemble only through a bias-variance-covariance tradeoff, not by maximizing disagreement blindly.
  https://www.jmlr.org/papers/v6/brown05a.html

- Wood et al. 2023, *A Unified Theory of Diversity in Ensemble Learning*:
  diversity is part of model fit, tied to statistical dependency between ensemble members, not an unconditional objective.
  https://jmlr.org/beta/papers/v24/23-0041.html

- Shazeer et al. 2017, *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*:
  sparse expert systems need routing discipline because uncontrolled expert competition and imbalance are real failure modes.
  https://arxiv.org/abs/1701.06538

- Liu et al. 2023, *Diversifying the Mixture-of-Experts Representation for Language Models with Orthogonal Optimizer*:
  homogeneous experts reduce MoE value because the experts fail to specialize enough to justify routing.
  https://arxiv.org/abs/2310.09762

These sources fit the local results well:

- positive competition-noise gain across the whole portfolio means pruning expert interaction matters
- negative best-pair-vs-`FULL` on Miami, Diamonds, and Wine means diversity without sufficient expert quality is not enough

## Decision

Phase 0f supports moving the design forward with a narrower rule:

1. do not expand the expert family yet
2. promote only specialists that survive the portfolio signal-vs-noise test
3. treat `GEO` as the strongest current specialist
4. improve specialist quality or specialist separability before investing in richer dense routing

## Next Questions

The next architecture question is no longer “should GraphDrone use views?”

It is:

1. do fresh quality-feature recomputations preserve the current specialist ranking on one GEO-win dataset and one loss-case dataset?
2. if yes, should GraphDrone become a sparse `FULL + GEO` integrator on the current portfolio win cases?
3. or should the next work focus on making non-`FULL` experts stronger before any richer routing objective is introduced?
