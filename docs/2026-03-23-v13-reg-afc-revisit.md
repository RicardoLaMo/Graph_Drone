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
