# Multiclass V1.3 Phase MC-3 — Per-Class OVR Threshold Calibration

Date: 2026-03-23
Branch: exp/v1.3-mc/phase1 (continuation)
Depends on: MC-2 (bagged quality tokens promoted)

## Hypothesis

Static `argmax` is suboptimal when class prevalence is imbalanced. The OOF blend
probabilities from the trained noise_gate_router (MC-1+) can be used to compute
per-class OVR (one-vs-rest) F1-maximizing thresholds. Applied at inference via
`proba / class_thresholds` + argmax, this is equivalent to per-class decision biases.

The binary path already does this (Phase 3B, `_compute_oof_threshold`). MC-3 is the
multiclass extension: loop over classes, call `_compute_oof_threshold(y==c, proba[:, c])`.

Most likely to help: `maternal_health_risk` (3-class, highly imbalanced: 406/178/430).
Least risky: datasets where all classes are already well-calibrated will fall back to 0.5
via the existing `min_improvement` and `min_shift` guards.

## Key Reuse

- `_compute_oof_threshold` (model.py) — called per class in OVR loop
- `min_improvement=0.01, min_shift=0.08, min_pos_samples=30` guards transfer directly
- `class_thresholds_` attribute (stub added in Phase 3B prep, model.py:110)
- Benchmark script: `adjusted = proba / class_t; labels = argmax(adjusted)` (already added)

## Config

- Preset: `v1_3_mc_phase3`
- `calibrate_multiclass_thresholds=True`
- All MC-2 settings inherited
- Version string: `2026.03.23-clf-v1.3-mc-phase3`

## Benchmark Contract

- Runner: `scripts/run_smart_benchmark.py`
- Champion: `v1_3_mc_phase2` / `2026.03.23-clf-v1.3-mc-phase2`
- Challenger: `v1_3_mc_phase3` / `2026.03.23-clf-v1.3-mc-phase3`
- Datasets: 7 multiclass × 3 folds (21 tasks)
- Gate focus: F1 vs MC-2; log_loss regression < 0.005

## Gate Criteria

MUST:
- Mean F1 delta >= 0.0 vs MC-2
- No single-dataset F1 regression > 0.003 vs MC-2
- log_loss regression < 0.005 (threshold shifts improve F1 but can hurt calibration)

SHOULD:
- `maternal_health_risk` F1 improves (most imbalanced)
- Mean class_thresholds differ from 0.5 for at least one dataset (calibration active)

Evidence grade: evidence-grade (full benchmark, GD champion vs GD challenger)

## Failure Protocol

If log_loss regresses > 0.005 on any dataset:
- Reduce `min_improvement` threshold to 0.02 (stricter filter on weak signals)
- Re-run benchmark

If no thresholds differ from 0.5 (all guarded out):
- Note: the learned router already calibrates probabilities well
- MC-3 is a no-op — still correct behavior, no regression

If mean F1 delta negative:
- Record as "OVR threshold calibration hurts multiclass" in findings.jsonl
- Do not promote

## Files Changed

- `src/graphdrone_fit/config.py`: Added `calibrate_multiclass_thresholds: bool = False`
- `src/graphdrone_fit/model.py`: Added `_compute_oof_multiclass_thresholds`; wired into
  `_fit_classification_router` (multiclass branch); prints calibrated thresholds
- `src/graphdrone_fit/presets.py`: Added `v1_3_mc_phase3`
- `scripts/run_smart_benchmark.py`: `class_thresholds_` handling in label computation
- `tests/test_threshold_calibration.py`: MC-3 OVR tests
- `docs/research/v1.3/mc-phase3-multiclass-threshold-contract.md`: This file
