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
