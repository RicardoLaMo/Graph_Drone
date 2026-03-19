# GraphDrone — Developer Guide for Claude

## Current best ELO: 1427 (2026-03-18, main, v2026.03.18h)
Benchmark: 12 datasets × 3 folds vs TabPFN v2.5 default (smart benchmark runner).
TabPFN overall ELO: 1573 | GraphDrone overall ELO: **1427**
Regression ELO: TabPFN 1560 / GraphDrone 1440
Classification ELO: TabPFN 1545 / GraphDrone 1455

---

## Architecture decisions — DO NOT change without benchmarking

### Router loss
```python
loss = main_loss + 2.0 * residual_penalty + defer_penalty
defer_penalty = out.defer_prob.mean() * 0.05
```
- `residual_penalty` guards against specialists hurting the anchor
- `defer_penalty = 0.05` is calibrated for the current specialist portfolio
- Raising this above 0.05 suppresses routing; lowering it causes over-deferral
- ONLY change if you also change the specialist portfolio

### Expert training / router validation split
```python
X_tr, X_va = train_test_split(X, test_size=0.1)
# Experts fit on X_tr (90%)
# Router validates on X_va (10%) — must be a clean holdout
```
- **Do NOT train experts on full X** — tree models (CatBoost/XGBoost) memorise X_va, inflating defer_prob artificially
- The 90/10 split keeps expert training and router validation independent

### HyperSetRouter + LayerNorm
- `expert_ln = nn.LayerNorm(hidden_dim)` after `expert_proj` is critical — prevents NaN in MultiheadAttention when regression targets have large magnitude (±hundreds)
- Do not remove this LayerNorm

### Problem type detection
```python
if self.config.problem_type == "classification" or (
    self.config.problem_type != "regression"
    and not is_float_target and len(unique_y) < 50
):
```
- Explicit `problem_type="regression"` in config is a hard override — it bypasses auto-detection
- Required to prevent CUDA NLL-loss assertion failures on integer-valued regression targets

### Anchor masking in integration
- The anchor expert (FULL) is **excluded** from the specialist weight blend
- `spec_mask[full_index] = 0` before normalisation
- This prevents the router from "deferring to itself"

---

## GORA geometric observers — NOT in main (do not re-add without reading this)

`observers.py` (kappa + LID) was removed from the active routing path.

**Why:** GORA was beneficial in v1-width (ELO 1507) where all specialists were TabPFN subspace views. In current main, specialists include CatBoost and XGBoost. GORA's geometric signal causes the router to defer to tree models in geometrically complex regions, but tree models are weaker than TabPFN on those exact regions → ELO drops to 1415.

**To re-enable GORA correctly:** Use TabPFN-only specialist views (remove CatBoost/XGBoost). Then GORA routes between TabPFN views of different feature subsets, which works well.

Investigation: PR #18 and PR #19 on GitHub.

---

## Benchmark runner

```bash
# Full run (12 datasets × 3 folds)
PYTHONPATH=src python scripts/run_smart_benchmark.py --folds 0 1 2

# Quick smoke test (3 datasets × 1 fold)
PYTHONPATH=src python scripts/run_smart_benchmark.py --quick --folds 0

# Resume after crash — already-cached tasks are skipped automatically
PYTHONPATH=src python scripts/run_smart_benchmark.py --folds 0 1 2
```

- Cache is keyed by `SHA256(dataset|fold|method|GRAPHDRONE_VERSION)[:16]`
- **Bump `GRAPHDRONE_VERSION`** in `scripts/run_smart_benchmark.py` whenever you change model code, or old results will be used
- Results in `eval/smart_benchmark/` (gitignored)

---

## ELO history

| Date | Version | ELO | Notes |
|---|---|---|---|
| 2026-03-15 | v1-width (v2026.03.15) | 1507 | GORA + TabPFN-only specialists. Binary clf only. |
| 2026-03-18 | 2026.03.18h | **1427** | ← **current main**. No GORA. CatBoost+XGBoost+TabPFN. Full multiclass. 72/72 tasks. |
| 2026-03-18 | 2026.03.18e (PR #19) | 1415 | GORA restored but mismatched with tree specialists → over-defers. Closed. |

---

## Known gaps (not yet implemented)

1. **`quality_scores` in tokens** — `portfolio_loader.py` has a `pass` stub where bagged estimator variance should be extracted. All quality tokens are currently zero. Implementing real variance should give the router uncertainty information and improve ELO.

2. **GORA with TabPFN-only specialists** — The geometric signal is sound; the specialist portfolio is the mismatch. A pure TabPFN multi-view portfolio + GORA should recover the v1-width advantage.

3. **Multiclass classification lag** — GD wins on credit_g, diabetes, mfeat_factors but trails on pendigits, optdigits, segment (saturated datasets where TabPFN is very strong). Routing signal is weak when all experts agree.
