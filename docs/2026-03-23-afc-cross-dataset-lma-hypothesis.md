# AFC Cross-Dataset LMA / Hyper-Router Hypothesis

Date: 2026-03-23

Origin:
- surfaced during `exp/afc-b-residual-objective`
- motivated by the persistent Phase B pattern that local routing diagnostics can improve without producing a strong held-out win

## Question

Is single-dataset router fitting fundamentally too weak for GraphDrone, such that the right next-scale AFC direction is a cross-dataset latent manifold alignment prior rather than more per-dataset auxiliary losses?

## Why this is now a live hypothesis

Current Phase B evidence points in the same direction:

1. Rotor alignment is real.
- Specialist tokens become more aligned to the anchor frame.
- But the end-task win is weak or absent.

2. Some earlier negative results were caused by circuit issues.
- Regression anchor contamination was partly causal.
- California regression routing had a real non-finite training failure.

3. Even after those confounds were reduced, the main remaining gap is allocation quality.
- Useful specialists can exist on validation rows.
- The realized attention-weighted specialist value is often still negative.

4. A more direct local objective also stalls.
- The residual-usefulness penalty reduces the diagnosed gap.
- But it can do so mainly by collapsing defer rather than learning a better allocation policy.

Taken together, that suggests the bottleneck may not be only the local loss form. It may be the learning regime itself: per-dataset router fitting from a small validation split may be too data-starved to learn robust view-allocation structure.

## Proposed direction

Treat each dataset as a task-level object and each view as a token inside that task.

Then learn a shared routing prior across datasets:
- latent manifold alignment across view tokens from many datasets
- a transformer or hyper-router that learns reusable cross-task view geometry
- per-dataset routers that are conditioned or initialized from that shared prior instead of learned from scratch

This would shift AFC-AI from a local per-dataset alignment module to a meta-routing framework:
- local alignment inside a dataset
- plus cross-dataset alignment of view families and routing behavior

## Claimed advantage over the current local-only path

The expected gain is not just better per-dataset fit. The expected gain is a better inductive bias:
- specialists should stay usefully different from the anchor
- routers should learn when that difference is likely to help from prior tasks
- small per-dataset validation splits should no longer bear the full burden of learning routing policy from scratch

## What would count as supporting evidence

Any of the following would strengthen this hypothesis:
- local auxiliary losses keep improving diagnostics mainly by reducing defer rather than improving positive realized specialist value
- cross-dataset token statistics show recurring view-family geometry that is stable across tasks
- a shared prior improves routing on held-out datasets with fewer per-dataset routing updates

## What would count against it

This hypothesis weakens if a purely local routing objective can:
- improve realized specialist value without collapsing defer
- and produce a stable held-out win on the mini-full contract

## Next checks

1. Continue the local diagnostic path long enough to determine whether the residual-usefulness objective can be redesigned to reward selective positive routing rather than routing suppression.
2. If local fixes keep stalling, open a separate branch for a minimal cross-dataset routing-prior prototype.
3. In that prototype, start with offline token-bank analysis before full training:
   - dataset-level view tokens
   - cross-dataset similarity structure
   - whether view families cluster in a reusable latent manifold
