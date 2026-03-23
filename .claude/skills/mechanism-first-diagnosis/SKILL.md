---
name: mechanism-first-diagnosis
description: Use when a new method shows a local or component-level gain but the end-to-end result is flat or negative, and the real task is to explain the break between mechanism truth, integration, and final outcome.
---

# Mechanism-First Diagnosis

This is the generic parent of project-specific mechanism-diagnosis skills.

Its job is not to declare “the method failed” from the final metric alone.
Its job is to locate where the gain disappeared.

## Non-Negotiable Rules

- Never reject an idea from final benchmark scores alone.
- Never skip component-level evidence.
- If local gain exists but global gain does not, explain the failure chain.
- If the failure chain is still unclear, do not collapse the result into a simple reject.

## Workflow

1. Write the change card.
- what changed
- what bottleneck it targets
- what local signature should improve
- what end-to-end signature should improve

2. Separate four layers.
- component truth
- circuit or policy coupling
- outcome translation
- reproducibility control

3. Force a primary failure bucket.
- adapter missing
- circuit mismatch
- objective mismatch
- foundation mismatch
- numerical instability
- data regime mismatch
- mechanism illusion
- implementation bug

4. End with one discriminating next check.
- smallest analysis or code change that could falsify the current interpretation

## Adaptation Notes

Project-specific versions should bind this workflow to concrete artifacts such as claim reports, routing diagnostics, or calibration reports.

Read `references/failure-buckets.md` and `references/question-ladder.md`.
