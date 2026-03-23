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

## Acceptance

This lane succeeds only if AFC-style geometry improvements also improve regression routing usefulness.
