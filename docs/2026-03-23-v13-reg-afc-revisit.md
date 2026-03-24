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
- comparing raw champion vs challenger regression rows shows:
  - mean weighted specialist advantage got slightly more negative overall
  - mean positive specialist mass also slipped slightly overall
  - defer fell on average

So the first AFC revisit result is:
- geometry: yes
- tiny benchmark lift on this narrow slice: yes
- clear allocation translation: not yet

That means rotor is still not a clean routing-usefulness win for regression.
It may be helping via a small change in overall routing behavior, but it is not yet making specialist use more clearly beneficial on average.

## Acceptance

This lane succeeds only if AFC-style geometry improvements also improve regression routing usefulness.
