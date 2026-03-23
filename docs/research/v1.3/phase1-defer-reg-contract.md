# V1.3 Phase 1 — Defer Regularization Contract

Date: 2026-03-23
Branch: exp/v1.3/phase1-defer-reg

## Hypothesis

Task-prior-aware defer regularization from the AFC line (promoted at v10,
`eval/afc_live_task_prior_binary_deferpen_l02_v10/`) can be ported into the main
binary classification router training as an unconditional quadratic penalty:

    loss += defer_penalty_lambda * (mean_defer - defer_target)^2

This should desaturate defer from the ~0.999 regime observed on credit_g,
improving F1 by redistributing weight to specialist experts.

## Source

- AFC promoted result: `exp/afc-cross-dataset-lma` v10
  - mean_f1_delta = +0.006216, mean_log_loss_delta = -0.002678
  - Branch note: docs/2026-03-23-afc-cross-dataset-lma-kickoff.md
- AFC finding: docs/research/findings.jsonl — afc-framework-task-prior-defer-regularization

## Config

- `defer_penalty_lambda = 0.5` (env: GRAPHDRONE_DEFER_PENALTY_LAMBDA)
- `defer_target = 0.8` (env: GRAPHDRONE_DEFER_TARGET)
- Preset: v1_3_phase1
- Version string: 2026.03.23-clf-v1.3-phase1

## Benchmark Contract

- Runner: scripts/run_smart_benchmark.py
- Champion version: 2026.03.19-clf-mc-v1.20 (GRAPHDRONE_PRESET=v1_20_champion)
- Challenger version: 2026.03.23-clf-v1.3-phase1 (GRAPHDRONE_PRESET=v1_3_phase1)
- Datasets: 9 classification datasets × 3 folds (54 tasks)
- Cache policy: bump GRAPHDRONE_VERSION_OVERRIDE before each run

## Gate Criteria

MUST (all required for promotion):
- Overall clf ELO >= 1503.3 (no regression vs v1.20)
- credit_g mean F1 delta >= +0.003
- No single-dataset F1 regression > 0.005

SHOULD (not hard gates):
- diabetes F1 does not decrease by more than 0.002
- mean_defer on credit_g drops below 0.95 (desaturation evidence)

Evidence grade: evidence-grade (full benchmark, controlled comparison)

## Lambda Sweep (if MUST gates fail)

Sweep: defer_penalty_lambda in {0.1, 0.3, 0.5, 1.0} × defer_target in {0.6, 0.7, 0.8, 0.9}
Start with quick fold-0 runs on credit_g + diabetes only before full benchmark.

## Failure Protocol

If all lambda settings fail:
- Record as partially_causal or open in docs/research/findings.jsonl
- Do not proceed to Phase 2
- Investigate whether the AFC v10 result depended on task-prior confidence
  scaling (task_prior_confidence_scale()) rather than the unconditional penalty

## Files Changed

- src/graphdrone_fit/config.py: +defer_penalty_lambda, +defer_target fields
- src/graphdrone_fit/model.py: defer penalty in _fit_classification_router loop
- src/graphdrone_fit/presets.py: +v1_3_phase1 preset
- scripts/run_smart_benchmark.py: version bumped to 2026.03.23-clf-v1.3-phase1
- tests/test_defer_regularization.py: new unit tests
