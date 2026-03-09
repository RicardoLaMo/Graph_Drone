# MQ-GoRA v4 Split-Track Design

## Intent

MQ-GoRA v4 is a controlled continuation of v3, not an unrestricted architecture expansion. The branch keeps the validated v3 bug-fix path, begins with an integrity audit, and then evaluates split-track model changes that preserve MNIST-friendly mechanisms while removing or constraining the California mechanisms most likely to destabilize regression.

## Self-Alignment

1. The confirmed bug fixes mean the old kwargs crash and the slow centroid/label precomputes are already corrected under numerically invariant rewrites.
2. Those fixes do not explain current v3 regression differences because the recorded v3 metrics stayed unchanged after the fixes.
3. Routing in this project means observer-guided structural control over trusted view weights and isolation-vs-interaction behavior.
4. Routing is not weighted ensembling, direct curvature prediction, or blind feature augmentation.
5. California and MNIST must be split tracks because the classification-friendly label/teacher mechanisms that helped MNIST were implicated in California regression collapse.
6. The v4 objective is to verify integrity first, then test whether regression-safe routing repairs California without sacrificing the MNIST gains that were real in v3.

I confirm v4 will be evaluated under split-track logic.
I confirm geometry signals are routing priors, not appended prediction features.
I confirm known bug fixes are numerically invariant and will not be used as a false explanation for v3 model weakness.

## Shared Backbone

- Keep view-specific embedders, neighbourhood context summaries, observer vectors, teacher support, and split-track training utilities.
- Add an explicit observer-driven `beta` gate alongside `pi`, so mode routing is represented directly rather than inferred from peaked `pi`.
- Keep `tau` if useful for attention sharpness diagnostics, but do not treat it as a substitute for `beta`.
- Preserve existing v3-compatible models for integrity reference reproduction.

## California Track

- Treat California as regression-safe by default.
- Start from no label context, then add only normalised target-derived context, then LayerNorm, then teacher-lite, then healthier optimisation control.
- Keep teacher-lite restricted to `L_agree + L_label` unless later evidence justifies centroid loss.
- Report whether training dynamics improve, whether collapse recedes, and whether any variant moves back toward G2.

## MNIST Track

- Lock G10 semantics first, then make incremental extensions only.
- Preserve alpha gate unless an ablation proves it harmful.
- Focus the MNIST track on gain retention and improved routing behavior, not on sweeping changes that would blur the comparison to v3.

## Outputs

- Shared integrity outputs: interface compatibility, shape audit, precompute timing, integrity report.
- Per-dataset outputs: metrics, regime metrics, routing stats, routing figures, root-cause audit, gates report, final report.
- `experiments/mq_gora_v4/README.md` explains branch purpose, split-track rationale, and exact run commands.
