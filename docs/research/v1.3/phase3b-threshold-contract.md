# V1.3 Phase 3B — OOF Threshold Calibration

Date: 2026-03-23
Branch: exp/v1.3/phase3b-threshold-model
Depends on: exp/v1.3/phase2-task-prior (promoted)

## Hypothesis

For binary classification, the default argmax threshold (0.5) is suboptimal.
Computing an F1-maximizing threshold on OOF blend predictions and applying it at
inference should close the remaining credit_g gap (currently −0.0014 F1 vs TabPFN)
while preserving gains on diabetes.

The OOF blend already sees the trained router's allocation — the calibrated threshold
adjusts the decision boundary on top of the router's probability output without
re-training any weights.

## Source

- AFC finding: credit_g gap is at score ordering / threshold geometry level
- Phase 2 finding: gap narrowed 65% (−0.004 → −0.0014) via task-prior anchor shift
- Threshold sweep on AFC line: optimal threshold on credit_g OOF ≈ 0.38–0.42

## Config

- Preset: v1_3_phase3b
- `calibrate_threshold` = True (env: GRAPHDRONE_CALIBRATE_THRESHOLD)
- All Phase 2 settings inherited (task_prior_bank_dir, strength=0.5, etc.)
- Version string: 2026.03.23-clf-v1.3-phase3b

## Benchmark Contract

- Runner: scripts/run_smart_benchmark.py
- Champion version: 2026.03.23-clf-v1.3-phase2 (phase 2 promoted result)
- Challenger version: 2026.03.23-clf-v1.3-phase3b
- Datasets: 9 classification datasets × 3 folds (54 tasks)
- Gate focus: credit_g binary (primary), full suite (no regression)

## Gate Criteria

MUST (all required for promotion):
- Overall clf ELO >= Phase 2 ELO (1507.9, no regression)
- No single-dataset F1 regression > 0.005

SHOULD (not hard gates):
- credit_g gap fully closed (F1 >= TabPFN baseline 0.6937)
- diabetes F1 not degraded vs Phase 2 (>= 0.7405)
- binary_threshold_ != 0.5 for credit_g (threshold is active)

Evidence grade: evidence-grade (full benchmark, controlled comparison)

## Failure Protocol

If credit_g does not close despite threshold activation:
- Record as partially_causal in findings.jsonl
- Proceed to Phase 3A (expert-allocation ordering) as next mechanism

If ELO regresses:
1. Set calibrate_threshold=False to isolate overhead → if matches Phase 2: threshold hurts globally
2. Investigate per-dataset threshold stability across folds

## Files Changed

- src/graphdrone_fit/config.py: +calibrate_threshold: bool = False to SetRouterConfig
- src/graphdrone_fit/model.py: +_compute_oof_threshold(); call after router training loop when calibrate_threshold=True; store as self.binary_threshold_
- src/graphdrone_fit/presets.py: +v1_3_phase3b preset
- scripts/run_smart_benchmark.py: use gd.binary_threshold_ for binary label computation when present
- tests/test_threshold_calibration.py: unit tests for threshold computation
