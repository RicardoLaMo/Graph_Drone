# Multiclass V1.3 Phase MC-1 — Learned Router for Multiclass

Date: 2026-03-23
Branch: exp/v1.3-mc/phase1
Depends on: main (v1.3.0, Phase 3B promoted)

## Hypothesis

The multiclass path currently uses static `anchor_geo_poe_blend(anchor_weight=5.0)` with no
router training. A config flag `use_learned_router_for_classification: bool = True` exists in
`GraphDroneConfig` but was never wired — `_classification_router_config` hardcodes
`use_learned = is_binary`, forcing multiclass to static GeoPOE regardless of config.

Enabling the existing NLL + residual-penalty training loop for multiclass allows the router to
learn dataset-specific defer/specialist decisions that static anchor_weight=5.0 cannot. The OOF
training loop (model.py:485–629) is already C-agnostic: NLL loss, learned_geo_poe_blend_torch,
entropy tokens, and GORA all handle arbitrary C.

The "DO NOT use contextual_transformer for classification" rule was based on binary having 78–100
OOF rows (37:1 param/sample). Multiclass datasets have 1000–10992 rows yielding 100–1100 OOF
rows at 10% split (3:1 to 15:1 param/sample) — a far healthier ratio.

## Source

- Binary V1.3 Phase 3B (promoted): credit_g gap closed via OOF-trained noise_gate_router
- CLAUDE.md DO NOT rule: applies to binary small-dataset case, not multiclass
- config.py:160: `use_learned_router_for_classification` flag — existing intent, never wired

## Config

- Preset: v1_3_mc_phase1
- `use_learned_router_for_classification` = True
- Router: noise_gate_router (auto-upgraded from bootstrap_full_only, same as binary)
- Guard: if OOF rows < 150, fall back to static GeoPOE
- Guard: if len(expert_specs) <= 1 (FULL-only portfolio), fall back to static GeoPOE
- Binary settings: inherit all Phase 3B binary settings unchanged
- Version string: 2026.03.23-clf-v1.3-mc-phase1

## Benchmark Contract

- Runner: scripts/run_smart_benchmark.py
- Champion version: 2026.03.23-clf-v1.3-phase3b (v1.3.0)
- Challenger version: 2026.03.23-clf-v1.3-mc-phase1
- Datasets: 7 multiclass datasets × 3 folds (21 tasks)
  - segment, mfeat_factors, pendigits, optdigits, maternal_health_risk, website_phishing, SDSS17
- Gate focus: multiclass F1 (primary), multiclass log_loss (secondary)
- Binary sanity: diabetes + credit_g F1 must be unchanged

## Gate Criteria

MUST (all required for promotion):
- Mean multiclass F1 delta >= 0.0 (no net regression across 7 datasets)
- No single-dataset F1 regression > 0.005
- Binary datasets (diabetes, credit_g) F1 unchanged vs Phase 3B

SHOULD (not hard gates):
- website_phishing F1 improvement >= 0.001 (currently GD trails TabPFN by -0.0009)
- Mean defer_prob in [0.3, 0.9] on multiclass datasets (not saturated)
- Mean multiclass log_loss improves

Evidence grade: evidence-grade (full benchmark, controlled GD champion/challenger)

## Failure Protocol

If defer saturates (> 0.95 or < 0.05 mean):
- Inspect defer head init bias; try nn.init.constant_(defer_head[-1].bias, -1.5) instead of -3.0
- Record as "defer saturation on multiclass"

If regression on low-feature datasets (<=10 features, 1 expert):
- The guard `if len(expert_specs) <= 1: use_learned = False` should have caught this
- If not, add explicit n_features guard

If mean F1 delta is negative:
- Do NOT proceed to MC-2
- Record as "learned router hurts multiclass" in findings.jsonl
- Investigate per-dataset defer distributions before abandoning

## Files Changed

- src/graphdrone_fit/model.py: _classification_router_config() — wire use_learned_router_for_classification for multiclass; add OOF/expert guards; guard binary threshold block with is_binary check
- src/graphdrone_fit/presets.py: add v1_3_mc_phase1 preset
- tests/test_multiclass_router.py: unit tests for multiclass router training
