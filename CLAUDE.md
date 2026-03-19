# GraphDrone — Developer Guide for Claude

## Current best ELO: **GD 1502.2 vs TabPFN 1497.8** — GD WINS (2026-03-19, feat/clf-multiclass-win)
Benchmark: 6 datasets × 3 folds, smart benchmark vs TabPFN v2.5 default.

### Classification summary (2026-03-19)
| Dataset | GD F1 | TPF F1 | Result |
|---|---|---|---|
| diabetes (binary) | 0.7549 | 0.7320 | **GD wins +0.023** |
| credit_g (binary) | 0.6787 | 0.6937 | Gap closed: was −0.054, now −0.015 |
| segment (7-class) | 0.9474 | 0.9474 | Tie |
| mfeat_factors (10-class) | 0.9861 | 0.9826 | **GD wins +0.004** |
| pendigits (10-class) | 0.9949 | 0.9959 | Near-saturation |
| optdigits (10-class) | 0.9930 | 0.9924 | **GD wins +0.001** |

Regression ELO (main, v1.18): GD 1523.2 vs TabPFN 1476.8 — **GD WINS**

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
| 2026-03-18 | 2026.03.18h | **1427** | No GORA. CatBoost+XGBoost+TabPFN. Full multiclass. 72/72 tasks. |
| 2026-03-18 | 2026.03.18e (PR #19) | 1415 | GORA restored but mismatched with tree specialists → over-defers. Closed. |
| 2026-03-19 | v1-geopoe-2026.03.19a | **1479.5 clf** | feat/clf-improvement. Multi-view SUB portfolio (FULL+3×SUB) + static GeoPOE (anchor_weight=3.0). 35/36 tasks (1 OOM). |
| 2026-03-19 | v1.18.0 | **Reg 1523.2 / Clf 1479.5** | main. Regression engine wins. Classification still behind TPF. |
| 2026-03-19 | **v1.19.0** | **Clf 1502.2** ← **GD WINS** | feat/clf-multiclass-win → main. Binary/multiclass split + size-aware OOF + credit_g gap closed. |

---

## Binary vs multiclass split — SHIPPED in v1.19

**Binary path** (`is_binary = n_classes == 2`):
- Learned OOF NLL router with GORA + noise_gate_router
- OOF split: 20% holdout when n≤1500, 10% otherwise; **stratified** (credit_g fix)
- Expert portfolio: FULL + 1×SUB (50% features) for n_features < 25; else FULL + 3×SUB
- Small dataset anchor-only fallback: `skip_subs` when n < 500 AND n_features < 25
- OOF experts CPU-offloaded to avoid GPU OOM (8-model contention)

**Multiclass path** (`n_classes > 2`):
- Static `anchor_geo_poe_blend(anchor_weight=5.0)`
- Portfolio: FULL + 3×SUB (fracs 0.8 / 0.85 / 0.9)
- No router training — zero NLL overhead, valid probability output guaranteed

**Credit_g fix (from research/credit-g-binary-split):**
- `oof_test_size = 0.2 if n_all <= 1500 else 0.1` — doubled holdout for small binary datasets
- `stratify=y` in OOF split — prevents class imbalance in holdout for credit_g
- Gap closed: credit_g F1 was −0.054 vs TabPFN, now −0.015

## Known gaps (future work)

1. **`quality_scores` in tokens** — `portfolio_loader.py` has a `pass` stub. All quality tokens are zero. Real variance should give the router uncertainty signal and improve binary ELO further.

2. **credit_g still lags TabPFN** (−0.015). Root cause: 20 features × 3 SUBs at 70-80% provides minimal diversity; OOF holdout still only ~160 rows after stratify fix. Tracked in research/credit-g-binary-split. Further improvement: Latin square permutations (Idea E in tabicl_inspiration.md).

3. **TabICL-inspired ideas** (research/tabicl-inspiration) — class shift + YJ view + temperature bundle tested: net −4.3 ELO. YJ 5th expert drags segment. Ablation needed: class-shift-only is promising for 10-class datasets.
