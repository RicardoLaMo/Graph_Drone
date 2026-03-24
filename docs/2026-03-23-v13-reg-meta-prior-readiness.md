# v1.3 Regression Meta-Prior Readiness Lane

Branch:
- `exp/v13-reg-meta-prior-readiness`

Base:
- `exp/v13-regression-program` at `f58b116`

Worktree:
- `/home/wliu23/projects/GraphDrone2/.worktrees/v13-reg-meta-prior-readiness`

## Mission

Determine whether regression should escalate to a task-conditioned meta-prior only after local routing is improved.

## Starting evidence

Read first:
- `docs/2026-03-23-afc-cross-dataset-lma-hypothesis.md`
- `docs/2026-03-23-v13-regression-handoff.md`
- `docs/research/current_hypotheses.md`

Current read:
- cross-dataset LMA is promising in principle
- but regression is not yet as structurally ready as classification
- local routing is still the main suspect

## First implementation targets

1. Do not start with a new regression prior model.
2. Define concrete entry criteria for this lane:
   - local routing diagnostics improve, but held-out regression still fails
   - or repeated evidence shows single-dataset router learning is too weak
3. Keep regression meta-prior work separate from the current classification-first task bank unless new evidence justifies sharing.

## First benchmark contract

No broad benchmark run should start until the entry criteria are satisfied.

## Acceptance

This lane succeeds when the team can justify, with artifacts, that regression should move beyond local routing into a task-conditioned prior.
