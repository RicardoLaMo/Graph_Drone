# Multiclass V1.3 Phase MC-2 — Bagged Quality Tokens + GORA for Multiclass

Date: 2026-03-23
Branch: exp/v1.3-mc/phase1 (continuation)
Depends on: MC-1 (borderline pass, learned router wired)

## Hypothesis

With the learned router now active for multiclass (MC-1), the router can consume per-expert
uncertainty signals that static GeoPOE had no use for:

1. **Bagged quality tokens** (`foundation_classifier_bagged`): Each expert runs 4 bags of
   TabPFNClassifier(n_estimators=2). The variance across bag predictions is encoded as
   `bag_variance` in `QualityEncoding` and appended to the router token. The router can
   learn to defer more when expert variance is high.

2. **GORA tokens**: Already computed in `_build_classification_tokens` (kappa + LID per
   expert's subspace). Under static GeoPOE (MC-0), GORA had no consumer. Under the learned
   router (MC-1+), GORA is consumed. MC-2 adds bagged variance on top of this.

The binary path already uses `foundation_classifier_bagged` (model.py:174, v1.20). The
change is to apply the same model kind to multiclass when `use_learned_router_for_classification=True`.

## Change

Single line in `model.py:_build_default_specs`:

```python
# Before (MC-1):
model_kind = "foundation_classifier_bagged" if is_binary else "foundation_classifier"

# After (MC-2):
use_bagged = is_binary or self.config.use_learned_router_for_classification
model_kind = "foundation_classifier_bagged" if use_bagged else "foundation_classifier"
```

## Config

- Preset: `v1_3_mc_phase2`
- All MC-1 settings inherited (noise_gate_router, min_experts=3 guard, OOF guard)
- Version string: `2026.03.23-clf-v1.3-mc-phase2`

## Benchmark Contract

- Runner: `scripts/run_smart_benchmark.py`
- Champion: `v1_3_mc_phase1` / `2026.03.23-clf-v1.3-mc-phase1b`
- Challenger: `v1_3_mc_phase2` / `2026.03.23-clf-v1.3-mc-phase2`
- Datasets: 7 multiclass datasets × 3 folds (21 tasks)
  - segment, mfeat_factors, pendigits, optdigits, maternal_health_risk, website_phishing, SDSS17
- Gate focus: multiclass F1 vs MC-1 champion
- Binary sanity: diabetes + credit_g must be unchanged

## Gate Criteria

MUST (all required for promotion):
- Mean multiclass F1 delta >= 0.0 vs MC-1 (no net regression)
- No single-dataset F1 regression > 0.003 vs MC-1

SHOULD:
- segment F1 improves vs MC-1 (variance signal stabilizes fold0 routing instability)
- mfeat_factors or optdigits F1 improves (high-dim datasets benefit most from quality tokens)

Evidence grade: evidence-grade (full benchmark, GD champion vs GD challenger)

## Failure Protocol

If mean F1 regresses vs MC-1:
- Check whether bag_variance signal is near-zero for multiclass datasets
- Compare quality token magnitudes (binary vs multiclass) via diagnostics
- Consider reverting to MC-1 with `foundation_classifier` only

If segment fold variance increases further (already 0.9376–0.9522 in MC-1):
- This may indicate the router is more sensitive to bag variance noise
- Record as "bagged quality tokens destabilize segment routing"

## Files Changed

- `src/graphdrone_fit/model.py`: Line 174 — use `foundation_classifier_bagged` for multiclass
  when `use_learned_router_for_classification=True`
- `src/graphdrone_fit/presets.py`: Added `v1_3_mc_phase2` preset
- `docs/research/v1.3/mc-phase2-bagged-gora-contract.md`: This file
