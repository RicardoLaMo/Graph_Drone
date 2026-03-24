# v1.3 Regression AFC Revisit Lane

Branch:
- `exp/v13-reg-afc-revisit`

Base:
- `exp/v13-regression-program` at `f58b116`

Worktree:
- `/home/wliu23/projects/GraphDrone2/.worktrees/v13-reg-afc-revisit`

## Mission

Re-test AFC alignment on top of the corrected regression circuit, but only as an allocation-translation question.

## Starting evidence

Read first:
- `docs/2026-03-23-afc-phase-b-claim-first.md`
- `docs/2026-03-23-afc-phase-b-anchor-exclusion.md`
- `docs/2026-03-23-afc-phase-b-residual-usefulness.md`
- `docs/2026-03-23-v13-regression-handoff.md`

Current read:
- rotor alignment is real
- anchor contamination was partly causal
- geometry gains alone are not enough

## First implementation targets

1. Keep AFC evaluation on the corrected regression circuit only.
2. Require both:
   - alignment gain
   - realized allocation/usefulness gain
3. Reject any branch that improves geometry but leaves realized specialist value negative.

## First benchmark contract

Use:
- quick regression gate first
- then fold-0 mini-full regression

## First quick result

Contract:
- `eval/v13_reg_afc_rotor_quick_v1/comparison/promotion_decision.json`
- `eval/v13_reg_afc_rotor_quick_v1/comparison/claim_report.json`
- `eval/v13_reg_afc_rotor_quick_v1/comparison/paired_task_deltas.csv`
- `eval/v13_reg_afc_rotor_quick_v2/comparison/paired_mechanism_summary.csv`
- `eval/v13_reg_afc_rotor_quick_v2/comparison/promotion_report.md`

Setup:
- regression legitimacy gate disabled to force a learned-routing surface
- datasets: `cpu_act`, `elevators`, `kin8nm`
- challenger router: `contextual_transformer_rotor`
- `alignment_lambda=0.1`

What cleared:
- all 9 task-folds stayed `clean_routed`
- no route-state switching confounded the comparison
- rotor alignment mechanism is still supported on this cleaner regression surface
  - mean alignment cosine gain: about `+0.0155`

What did not clear:
- promotion decision stayed `hold`
- mean RMSE relative improvement was only `+0.000141`
- mean latency got worse

Most important interpretation:
- the narrow quick slice gives a tiny positive metric average
- but the allocation story is weaker than that headline suggests
- the paired mechanism summary now makes that visible directly:
  - `cpu_act` improved weighted specialist advantage slightly
  - `elevators` and `kin8nm` both got worse on weighted specialist advantage
  - positive specialist mass also fell on `elevators` and `kin8nm`
  - defer fell on `elevators` and `kin8nm`

So the first AFC revisit result is:
- geometry: yes
- tiny benchmark lift on this narrow slice: yes
- clear allocation translation: not yet

That means rotor is still not a clean routing-usefulness win for regression.
It may be helping via a small change in overall routing behavior, but it is not yet making specialist use more clearly beneficial on average.

## Second quick result: rotor plus allocation objective

Contract:
- `eval/v13_reg_afc_rotor_allocation_quick_v3/comparison/promotion_decision.json`
- `eval/v13_reg_afc_rotor_allocation_quick_v3/comparison/paired_mechanism_summary.csv`
- `eval/v13_reg_afc_rotor_allocation_quick_v3/comparison/promotion_report.md`

Setup:
- same clean-routed regression slice: `cpu_act`, `elevators`, `kin8nm`
- regression legitimacy gate disabled
- challenger router: `contextual_transformer_rotor`
- `alignment_lambda=0.1`
- `allocation_usefulness_lambda=0.2`

What cleared:
- all 9 task-folds again stayed `clean_routed`
- the combined objective improved weighted specialist advantage on all 3 datasets relative to the champion
- compared with rotor-only, the mechanism surface improved materially:
  - `cpu_act`: weighted specialist advantage delta improved from about `+0.0060` to `+0.0535`
  - `elevators`: improved from about `-0.0283` to `+0.0071`
  - `kin8nm`: improved from about `-0.0282` to `+0.0133`
- positive specialist mass also improved on all 3 datasets relative to rotor-only

What did not clear:
- promotion decision still stayed `hold`
- mean RMSE relative improvement became slightly negative: about `-0.000349`
- latency worsened sharply on `cpu_act` and `elevators`

Most important interpretation:
- this is the first AFC revisit result that clearly improves the allocation/usefulness surface across the whole clean-routed slice
- so the old story, "rotor helps geometry but not allocation," is now too narrow
- the new story is:
  - rotor plus a direct allocation objective can improve allocation diagnostics
  - but that mechanism movement still does not translate into better held-out RMSE
  - and it currently buys those mechanism gains at a substantial latency cost

So the remaining blocker is no longer pure translation from geometry into allocation.
It is translation from improved allocation diagnostics into actual prediction quality and efficient routing behavior.

## Third quick result: robust allocation target

Contract:
- `eval/v13_reg_afc_rotor_robust_quick_v5/comparison/promotion_decision.json`
- `eval/v13_reg_afc_rotor_robust_quick_v5/comparison/paired_mechanism_summary.csv`
- `eval/v13_reg_afc_rotor_robust_quick_v5/comparison/promotion_report.md`

Setup:
- same clean-routed regression slice: `cpu_act`, `elevators`, `kin8nm`
- regression legitimacy gate disabled
- challenger router: `contextual_transformer_rotor`
- `alignment_lambda=0.1`
- `robust_allocation_usefulness_lambda=0.2`
- robust score = allocation usefulness rewarded only through the weaker half of a deterministic even/odd validation split

What cleared:
- all 9 task-folds again stayed `clean_routed`
- weighted specialist advantage improved on all 3 datasets relative to the champion
- positive specialist mass also improved on all 3 datasets
- unlike the conservative penalty probe, `elevators` did not collapse into obviously worse validation-side allocation

What did not clear:
- promotion decision still stayed `hold`
- mean RMSE relative improvement got worse again: about `-0.000500`
- the consistency-aware reward did not convert the mechanism gain into better held-out RMSE

Most important interpretation:
- this is a stronger negative result than the raw rotor-only run
- even after forcing the allocation objective to survive a simple split-consistency check, the held-out regression metric still regressed
- that means the current validation-side allocation target is itself not sufficiently causal for the downstream regression objective

So the next move should not be another small variation of the same validation-allocation reward.
The lane now needs a different signal family:
- either a better causal proxy for robust specialist benefit
- or a different routing design that does not rely on this single-dataset validation target as the main teacher

## Acceptance

This lane succeeds only if AFC-style geometry improvements also improve regression routing usefulness.

## Fourth quick result: live regression task-prior / regime-conditioned router

Contracts:
- invalid first pass: `eval/v13_reg_task_prior_live_quick_v1/comparison/promotion_decision.json`
- corrected rerun: `eval/v13_reg_task_prior_live_quick_v2/comparison/promotion_decision.json`
- corrected report: `eval/v13_reg_task_prior_live_quick_v2/comparison/promotion_report.md`
- challenger raw report: `eval/v13_reg_task_prior_live_quick_v2/raw/challenger/regression/report/results_granular.csv`

Setup:
- regression task bank fit from `california`, `cpu_act`, `elevators`, `kin8nm`
- challenger router: `contextual_transformer`
- task prior bank enabled
- exact reuse blend enabled
- regression legitimacy gate disabled

What the first pass taught:
- the first live regression task-prior contract was diagnostic-grade only
- regression router fitting never called `_maybe_attach_task_prior_router(...)`
- clean-routed regression predictions also failed to merge `self._task_prior_diagnostics`
- that is why `v13_reg_task_prior_live_quick_v1` was bit-for-bit identical to the champion and surfaced no `task_prior_*` fields

What was fixed:
- regression router fitting now wraps the base router with the task-prior-conditioned router
- clean-routed regression predictions now merge `task_prior_*` diagnostics like the classification path does
- focused regression tests now cover both:
  - regression prediction diagnostics include `task_prior_*`
  - `_fit_regression_router()` actually attaches the task-prior wrapper

What cleared on the corrected rerun:
- the challenger is now a real regime-conditioned regression router, not a silent bypass
- raw regression rows now show:
  - `router_kind=contextual_transformer_router_task_prior`
  - `task_prior_enabled=1`
  - `task_prior_query_dataset` equal to the real dataset key
  - exact reuse available on all known datasets
- quick champion/challenger decision became `promote`
  - mean RMSE relative improvement: `+0.032662`
  - mean R² delta: `+0.007938`
- the win is concentrated in the unstable regime:
  - `california` mean RMSE improved from `14.7535` to `12.9852`
  - one clean-routed fold improved sharply from `13.4676` to `7.7998`
- `cpu_act`, `elevators`, and `kin8nm` stayed effectively flat

Most important interpretation:
- this is the first regression result on this lane where task-level learning is both:
  - architecturally live
  - benchmark-visible
- the gain is not broad yet; it is regime-specific and currently concentrated on `california`
- that still matters, because `california` is exactly the regime that had been exposing instability and weak single-dataset teaching signals
- so the current read is:
  - regression task priors are not just plumbing-ready
  - they can improve the live route on at least one hard regime
  - the next question is whether this survives the mini-full fold-0 gate and whether the gain remains localized to instability-heavy regimes

## Fifth result: regression task-prior mini-full fold 0

Contract:
- `eval/v13_reg_task_prior_live_mini_v1/comparison/promotion_decision.json`
- `eval/v13_reg_task_prior_live_mini_v1/comparison/promotion_report.md`
- `eval/v13_reg_task_prior_live_mini_v1/comparison/paired_mechanism_summary.csv`
- challenger raw report: `eval/v13_reg_task_prior_live_mini_v1/raw/challenger/regression/report/results_granular.csv`

Setup:
- same task-prior bank and exact reuse blend as the corrected quick run
- gate: mini-full
- tasks: regression fold `0`
- datasets: `california`, `diamonds`, `house_prices`, `elevators`, `cpu_act`, `kin8nm`

What cleared:
- the challenger remained architecturally live on the clean-routed datasets:
  - `cpu_act`, `elevators`, and `kin8nm` all surfaced
    `router_kind=contextual_transformer_router_task_prior`
  - `task_prior_enabled=1`
  - exact reuse remained available on the known bank datasets
- so the quick result was not a one-off wiring artifact; the live regression task-prior path persists on the mini-full contract

What did not clear:
- promotion decision returned `hold`
  - mean RMSE relative improvement: `+0.000006`
  - mean R² delta: `+0.000001`
- the contract was essentially flat overall

Most important interpretation:
- the mini-full failure is not “task prior stopped working”
- it is more specific:
  - the datasets where the task prior was active on this contract (`cpu_act`, `elevators`, `kin8nm`) were already near-flat regimes, and they stayed near-flat
  - the unstable regimes where a stronger gain would matter most (`california`, `diamonds`, `house_prices`) still fell into `router_training_nonfinite_anchor_only`
  - once they fall back that early, the task-prior route cannot express itself
- this is why the quick contract looked promising while mini-full went flat:
  - quick included clean-routed `california` folds where the task prior could act
  - mini-full fold `0` happened to include only the `california` fallback fold, not the clean-routed winning folds

So the current state of the claim is:
- regression task-prior routing is real
- it can help on at least one hard regime when the learned router survives
- but on the mini-full fold-0 contract, the broader bottleneck is still regression router stability on the hard datasets, not absence of a useful task prior

## Sixth result: hard-regime stability fix for live regression task priors

Contracts:
- first stabilized hard-regime pass: `eval/v13_reg_task_prior_hardregimes_quick_v1/comparison/promotion_decision.json`
- fully stabilized hard-regime pass: `eval/v13_reg_task_prior_hardregimes_quick_v2/comparison/promotion_decision.json`
- refreshed-bank hard-regime pass: `eval/v13_reg_task_prior_hardregimes_quick_v3/comparison/promotion_decision.json`
- challenger raw reports:
  - `eval/v13_reg_task_prior_hardregimes_quick_v2/raw/challenger/regression/report/results_granular.csv`
  - `eval/v13_reg_task_prior_hardregimes_quick_v3/raw/challenger/regression/report/results_granular.csv`

Setup:
- datasets: `california`, `diamonds`, `house_prices`
- regression legitimacy gate disabled
- live task prior enabled with exact reuse blend
- new stabilization changes:
  - normalize regression router training targets before learned routing optimization
  - normalize regression prediction channels before token construction

What was fixed:
- the earlier hard-regime blocker was not lack of task-bank coverage
- it was scale instability inside regression router training and token construction
- after adding both normalization stages, all hard-regime task-folds stayed `clean_routed`
  instead of falling into `train_gradients/nonfinite_gradients`

What cleared:
- `v13_reg_task_prior_hardregimes_quick_v2` kept all 9 hard-regime task-folds live:
  - `router_kind=contextual_transformer_router_task_prior`
  - `router_nonfinite_fallback=0`
  - `regression_router_fallback_stage=none`
  - `regression_router_fallback_reason=none`
- the contract became nearly flat rather than catastrophically negative:
  - mean RMSE relative improvement: `-0.000535`
  - mean R² delta: `-0.000155`

What the refreshed bank changed:
- a six-dataset regression task bank was rebuilt to include:
  - `california`
  - `cpu_act`
  - `diamonds`
  - `elevators`
  - `house_prices`
  - `kin8nm`
- exact reuse then became available on all three hard regimes in `quick_v3`

What did not clear:
- even after stability was fixed and the bank knew all three hard regimes directly,
  the refreshed-bank contract still returned `hold`
  - mean RMSE relative improvement: `-0.002261`
  - mean R² delta: `-0.000356`
- the top-neighbor structure also stayed semantically weak:
  - `california -> cpu_act`
  - `diamonds -> cpu_act`
  - `house_prices -> cpu_act`

Most important interpretation:
- this round clears the stability question much more than the task-prior question
- the hard regression regimes can now stay on a live routed surface under the task-prior router
- once that routed surface exists, the current task-prior coupling is only making small or slightly negative outcome changes
- adding exact reuse for `diamonds` and `house_prices` did not materially improve the hard-regime outcome once routing was stable

So the new state of the lane is:
- hard-regime regression router stability is largely cleared as the dominant blocker
- bank coverage is no longer the main missing piece either
- the next live problem is task-prior coupling strength and shape on the stabilized hard-regime surface

## Seventh result: task-prior coupling sweep on the stabilized hard-regime surface

Contracts:
- stronger global prior strength:
  - `eval/v13_reg_task_prior_hardregimes_coupling_strength1_v1/comparison/promotion_decision.json`
  - `eval/v13_reg_task_prior_hardregimes_coupling_strength1_v1/comparison/paired_task_deltas.csv`
- stronger exact-reuse blend:
  - `eval/v13_reg_task_prior_hardregimes_coupling_exact1_v1/comparison/promotion_decision.json`
  - `eval/v13_reg_task_prior_hardregimes_coupling_exact1_v1/comparison/paired_task_deltas.csv`

Setup:
- same stabilized hard-regime slice: `california`, `diamonds`, `house_prices`
- same refreshed six-dataset regression bank
- same live task-prior route
- only coupling knobs changed

Probe A:
- `task_prior_strength=1.0`
- `task_prior_exact_reuse_blend=0.6`

Probe B:
- `task_prior_strength=0.5`
- `task_prior_exact_reuse_blend=1.0`

What cleared:
- both probes stayed fully `clean_routed`
- no regression router fallback returned
- so the sweep isolates prior coupling rather than re-opening the stability problem

What did not clear:
- stronger global coupling was worse than the stabilized baseline:
  - mean RMSE relative improvement: `-0.001753`
  - mean R² delta: `-0.000179`
- maximal exact-reuse blending was worse again:
  - mean RMSE relative improvement: `-0.003013`
  - mean R² delta: `-0.000389`

Most important per-fold degradations:
- stronger global coupling:
  - `california fold=1`: `-0.016901` RMSE relative improvement
- maximal exact reuse:
  - `california fold=2`: `-0.012733`
  - `diamonds fold=2`: `-0.007276`

Most important interpretation:
- the current task prior is not simply under-coupled
- making the prior stronger globally makes the stabilized hard-regime result worse
- forcing exact reuse harder also makes it worse
- that means the next problem is not just parameter tuning on the current additive conditioning path

So the current read becomes:
- stability: cleared locally
- bank membership: no longer the main blocker
- simple coupling sweeps: falsified as the main fix
- next design question: how should the regression task prior shape routing behavior, rather than how much of the current prior vector should be injected

## Eighth result: routing-bias task-prior architecture

Contracts:
- first routing-bias probe:
  - `eval/v13_reg_task_prior_hardregimes_routingbias_v1/comparison/promotion_decision.json`
  - `eval/v13_reg_task_prior_hardregimes_routingbias_v1/comparison/paired_task_deltas.csv`
- stronger routing-bias probe:
  - `eval/v13_reg_task_prior_hardregimes_routingbias_strength1_v1/comparison/promotion_decision.json`
  - `eval/v13_reg_task_prior_hardregimes_routingbias_strength1_v1/comparison/paired_task_deltas.csv`

Architecture change:
- the old task prior only shifted the anchor token
- the new `routing_bias` mode uses the task prior to bias expert attention logits and defer directly
- this changes the routing shape rather than just injecting more of the same prior vector into the anchor representation

Setup:
- same stabilized hard-regime slice: `california`, `diamonds`, `house_prices`
- same refreshed six-dataset regression bank
- same exact-reuse blend baseline of `0.6`

Probe A:
- `task_prior_mode=routing_bias`
- `task_prior_strength=0.5`

Probe B:
- `task_prior_mode=routing_bias`
- `task_prior_strength=1.0`

What cleared:
- `routing_bias` at `0.5` materially improved on the old additive task-prior path
- compared with the stabilized additive baseline (`quick_v2`, mean RMSE relative improvement `-0.000535`), the first routing-bias probe reached near-flat:
  - mean RMSE relative improvement: `+0.000001`
  - mean R² delta: `-0.000071`
  - worst dataset RMSE guardrail stayed clean at `-0.000178`
- the route stayed fully `clean_routed` on all 9 task-folds
- the most visible local gain was `california fold=1`, which improved by `+0.007176` RMSE relative improvement while reducing defer sharply versus the additive route

What did not clear:
- the first routing-bias probe still stayed `hold`
- stronger routing-bias coupling at `1.0` made the contract worse again:
  - mean RMSE relative improvement: `-0.001277`
  - mean R² delta: `-0.000353`

Most important interpretation:
- this is the first regression task-prior architecture result that beats the old additive coupling design clearly enough to matter
- the remaining problem is no longer "the task prior cannot shape routing"
- it can
- but the operating range is narrow:
  - additive injection was too weak or wrongly shaped
  - routing bias at moderate strength is much better
  - routing bias at higher strength degrades again

So the next state of the lane is:
- task-prior architecture matters
- expert/defer biasing is a better direction than anchor-only prior shift
- the new question is how to make the routing-bias path selective and stable enough to turn this near-flat result into a real regression win

## Ninth result: local-gated routing bias

Contract:
- `eval/v13_reg_task_prior_hardregimes_routingbias_localgate_v1/comparison/promotion_decision.json`
- `eval/v13_reg_task_prior_hardregimes_routingbias_localgate_v1/comparison/paired_task_deltas.csv`

Architecture change:
- keep the `routing_bias` task-prior route
- add a per-row local gate so the global task prior is scaled by anchor-to-prior alignment before biasing expert logits and defer

Setup:
- `task_prior_mode=routing_bias`
- `task_prior_strength=0.5`
- `task_prior_local_gate_alpha=2.0`
- same stabilized hard-regime slice and same six-dataset bank

What cleared:
- the local-global version stayed fully `clean_routed`
- the gate was live and measurable in the challenger report
- it preserved the architectural gain over the old additive route and remained essentially flat overall:
  - mean RMSE relative improvement: `+0.000009`
  - mean R² delta: `-0.000091`
- this is slightly better on RMSE than the ungated routing-bias probe (`+0.000001`), but only marginally

What did not clear:
- promotion still stayed `hold`
- the gain over ungated routing bias is too small to count as a real architectural breakthrough
- latency got worse relative to the champion

Most important interpretation:
- the local-vs-global idea is reasonable and does not destabilize the route
- but this first local gate is not yet selective enough to create a meaningful held-out regression gain
- the global prior is still doing most of the work; local gating only trims it slightly

So the current read is:
- local-global prior shaping is more defensible than global-only injection
- but the present cosine-gated formulation is too weak to solve the regression translation problem by itself

## Tenth result: per-expert local-global routing bias

Contract:
- `eval/v13_reg_task_prior_hardregimes_routingbias_expertlocal_v1/comparison/promotion_decision.json`
- `eval/v13_reg_task_prior_hardregimes_routingbias_expertlocal_v1/comparison/paired_task_deltas.csv`

Architecture change:
- keep the global task prior
- keep the row-level local gate
- add an expert-specific local gate so each expert token is scaled by its own alignment to the task-prior basis before contributing routing bias

Setup:
- `task_prior_mode=routing_bias`
- `task_prior_strength=0.5`
- `task_prior_local_gate_alpha=2.0`
- `task_prior_expert_local_gate_alpha=2.0`
- same stabilized hard-regime slice and same six-dataset bank

What cleared:
- the route stayed fully `clean_routed`
- this is the strongest regression task-prior result on the hard-regime slice so far:
  - mean RMSE relative improvement: `+0.000200`
  - mean R² delta: `-0.000051`
- this improved over:
  - ungated routing bias: `+0.000001`
  - scalar local gate: `+0.000009`

What did not clear:
- the contract still stayed `hold`
- the gain is real but still small
- worst-dataset guardrail remained slightly negative

Most important interpretation:
- local-vs-global alignment helps more when it is expert-specific rather than one scalar gate for the whole row
- this is the first version of the local-global idea that produces a visible step up over the plain routing-bias architecture
- the effect is still too small to count as solved, but it is no longer just “safe but flat”

So the next state of the lane is:
- global regime priors should shape specialists differently, not uniformly
- per-expert local-global gating is a better direction than row-level gating alone
- the next design question is how to make the expert-local gate depend on specialist opportunity, not just token-prior cosine

## Eleventh result: opportunity-aware expert-local routing bias

Contract:
- `eval/v13_reg_task_prior_hardregimes_routingbias_expertlocalopp_v1/comparison/promotion_decision.json`
- `eval/v13_reg_task_prior_hardregimes_routingbias_expertlocalopp_v1/comparison/paired_task_deltas.csv`
- challenger raw report:
  `eval/v13_reg_task_prior_hardregimes_routingbias_expertlocalopp_v1/raw/challenger/regression/report/results_granular.csv`

Architecture change:
- keep the `routing_bias` task-prior route
- keep the row-level local gate
- keep the per-expert local gate
- multiply the expert-local gate by a validation-derived per-expert opportunity score:
  mean positive residual advantage over anchor on the router-training validation split

Setup:
- `task_prior_mode=routing_bias`
- `task_prior_strength=0.5`
- `task_prior_local_gate_alpha=2.0`
- `task_prior_expert_local_gate_alpha=2.0`
- `task_prior_exact_reuse_blend=0.6`
- same stabilized hard-regime slice: `california`, `diamonds`, `house_prices`
- same six-dataset regression bank

What cleared:
- the route again stayed fully `clean_routed` on all 9 task-folds
- this was a real code-and-execution step, not just a thought experiment:
  - the router now receives explicit expert opportunity scores during regression router fitting
  - challenger rows surface `task_prior_expert_opportunity_mean`
- the mechanism was live:
  - `task_prior_expert_local_gate_mean` varied by fold and dataset
  - `task_prior_expert_opportunity_mean` was below `1.0`, so the opportunity signal actually changed expert-local gating

What did not clear:
- promotion stayed `hold`
- mean RMSE relative improvement regressed to `-0.000147`
- mean R² delta regressed to `-0.000128`
- latency also worsened to about `-3.27%`
- this is worse than the simpler expert-local gate:
  - expert-local gate: `+0.000200`
  - expert-local + static opportunity weighting: `-0.000147`

Most important interpretation:
- this does **not** falsify local-vs-global LMA for regression
- it falsifies this specific teacher formulation:
  static dataset-level per-expert opportunity weighting is not a good enough signal for the live router
- the likely failure mode is that the mean validation opportunity score is too coarse:
  - it helps some folds (`california` fold 1, `diamonds` fold 0)
  - but it hurts others (`california` fold 2) and does not move `house_prices`
- so the expert-local gate needs a richer signal than one fixed per-expert weight for the whole dataset/fold

So the next state of the lane is:
- keep the stabilized hard-regime base
- keep `routing_bias` and expert-local gating
- stop using static mean opportunity scores as the next teacher
- next design should use row-conditional or regime-conditional specialist opportunity, not one dataset-level average per expert
