# MQ-GoRA v4: Split-Track Experiment

**Branch:** `feature/mq-gora-v4-split-track`
**Date:** 2026-03-08
**Status:** Complete — gates partially satisfied; clear root-causes documented

---

## 1. Motivation

v3 (MQ-GoRA) improved MNIST but regressed badly on California Housing:

| version | CA RMSE | MNIST acc |
|---------|---------|-----------|
| HGBR baseline | 0.4433 | 0.9580 |
| G2 (GoRA-v1) | 0.4546 | 0.9300 |
| v3 G10 (full) | **0.5208** | **0.9380** |

Root causes identified for CA regression:
- `LabelContextEncoder` receives raw unnormalized regression targets → high-variance signal → pathological early stopping at epoch ~30 (vs G2's ~90)
- Teacher fails to converge on CA (loss 0.65 vs MNIST 0.12)

v4 splits into two isolated tracks to test regression-safe fixes without disrupting MNIST.

---

## 2. Architecture Summary

### Shared backbone (`shared/src/`)
| file | role |
|------|------|
| `meta_learner_v4.py` | `LabelContextEncoderV4` — optional LayerNorm on label_ctx_vec output |
| `manifold_teacher_v4.py` | `train_teacher_v4` — `skip_centroid_loss` flag (teacher-lite), configurable epochs |
| `row_transformer_v4.py` | `MQGoraTransformerV4` — all v3 components + `use_label_ctx_layernorm`, returns 6-tuple |
| `train_v4.py` | split-track training: CA (patience=40, cosine) + MN (lam_diversity); `normalise_lbl_nei` |
| `eval_v4.py` | metrics, routing diagnostics, regime metrics, report writer |
| `integrity_check.py` | system integrity: interface compat, shape sanity, precompute timing |

### Design rules (immutable)
1. Geometry signals are routing priors, not appended prediction features.
2. "Routing" = observer-driven view trust + explicit local/global mode control (alpha-gate). It is NOT post-hoc weighted ensembling.
3. California and MNIST stay separate tracks because regression-safe and classification-friendly signals differ.
4. Known bug fixes (kwargs passthrough, vectorised precompute) are numerically invariant — they do not explain v3 weaknesses.

---

## 3. California Track Results

**Target:** recover toward G2 (0.4546) from v3-collapse region (0.52+)

| variant | description | RMSE | delta vs G2 |
|---------|-------------|------|-------------|
| B1_HGBR | baseline | 0.4433 | ref |
| G2_ref | GoRA-v1 carry-forward | 0.4515 | +0.007 |
| G10_ref | v3 G10 reproduction | 0.5023 | +0.057 |
| **CA_v4a** | no label ctx (struct routing only) | **0.4719** | +0.022 |
| CA_v4b | + normalised label ctx | 0.5696 | +0.117 |
| CA_v4c | + LayerNorm on label_ctx | 0.5328 | +0.080 |
| CA_v4d | + teacher-lite (skip L_centroid) | 0.5419 | +0.089 |
| CA_v4e | + patience=40 + cosine annealing | 0.5316 | +0.079 |

**Gate summary:**
- C1 Training Health: PASS (stop_ep=99, no premature exit)
- C2 Regression-Safe Improvement: PASS (CA_v4a=0.4719 vs v3-bad=0.5099)
- C3 Toward G2: PARTIAL (0.4719 vs G2=0.4546; gap=0.017)
- R1 Router Input Active: PASS
- R2 View-Discriminative: PARTIAL (all heads still GEO-dominant)
- R3 Mode Routing Active: PASS (beta-std > 0)
- R4 Complexity Justified: PASS

**Key finding:** Label context hurts California regardless of normalization. CA_v4a (no label ctx) is the ceiling for v4. GEO dominance persists across all variants. The primary unresolved issue is that MQGoraTransformerV4 architecture cannot beat the simpler GoraTransformer (G2) on CA.

---

## 4. MNIST Track Results

**Target:** preserve G10 accuracy (0.9380 saved), test diversity regulariser

| variant | description | accuracy | delta vs G10 |
|---------|-------------|----------|--------------|
| B1_HGBR | baseline | 0.9573 | ref |
| G2_ref | GoRA-v1 | 0.9293 | — |
| G10_ref | v3 G10 reproduce | 0.9320 | -0.006 (run var) |
| **MN_v4a** | G10 path under v4 | **0.9333** | -0.005 |
| MN_v4b | + diversity lam=0.005 | 0.9320 | -0.006 |
| MN_v4c | + diversity lam=0.010 | 0.9260 | -0.012 |
| MN_v4d | + diversity lam=0.010 + LayerNorm | 0.9320 | -0.006 |

**Gate summary:**
- S1 Integrity: PASS
- M1 Gain Retention: PARTIAL (0.9333 vs saved G10=0.9380; ~+/-0.006 run variance)
- M2 Routing Quality: PARTIAL (PCA dominant across all heads)
- R1 Router Input Active: PASS
- R3 Mode Routing Active: PASS (beta-std > 0.12)
- R4 Complexity Justified: PASS

**Key finding:** v4 framework preserves G10-level MNIST performance within run-to-run variance (~+/-0.006). Diversity regulariser is neutral at lam=0.005 and actively harmful at lam=0.01. PCA view remains dominant — head collapse not fixed by diversity pressure alone.

---

## 5. Routing Analysis Summary

### California (CA_v4a best model)
```
head  dominant  mean_pi_GEO  top1_freq_GEO  beta_mean  beta_std
  0   GEO       0.778        0.959          0.283      0.126
  1   GEO       0.674        0.876          0.172      0.120
  2   GEO       0.410        0.620          0.106      0.089
  3   GEO       0.765        0.916          0.156      0.110
```
Routing collapses to GEO view; RichMoERouter cannot differentiate views on CA.

### MNIST (MN_v4a best model)
```
head  dominant  mean_pi_PCA  top1_freq_PCA  beta_mean  beta_std
  0   PCA       0.651        0.908          0.761      0.156
  1   PCA       0.550        0.891          0.749      0.123
  2   PCA       0.623        0.928          0.752      0.152
  3   PCA       0.469        0.827          0.796      0.169
```
PCA dominant; beta values high (0.75-0.80) meaning model leans local/graph mode. Better entropy than CA.

---

## 6. Unresolved Issues Before v5

| gate | status | priority | recommended fix |
|------|--------|----------|-----------------|
| C3 CA Toward G2 | PARTIAL | HIGH | Strip teacher/label complexity from CA; try pure geometry routing with stronger positional priors |
| M1 MNIST Gain Retention | PARTIAL | HIGH | Investigate G10_ref drift (-0.006); seed-fix and multi-run averaging before claiming architecture credit |
| R2 View-Discriminative | PARTIAL | MED | Head collapse needs architectural fix; consider view-specific key projections, not just diversity loss |
| diversity regulariser | HARMFUL at lam>=0.01 | HIGH | Remove from v5 MNIST default; keep only if it provably improves specialization |

---

## 7. Commit History

| hash | step | contents |
|------|------|----------|
| `619938f` | 1 — scaffold | branch + tree + shared backbone + runners |
| `48b9fac` | 2 — integrity | experiments/__init__.py fix; integrity check PASS |
| `177e822` | 4 — CA results | California v4 full run + reports |
| `a50bfa6` | 5 — MNIST results | MNIST v4 full run + reports |

---

## 8. Directory Map

```
experiments/mq_gora_v4/
├── shared/
│   ├── src/              # backbone: meta_learner_v4, manifold_teacher_v4,
│   │                     #           row_transformer_v4, train_v4, eval_v4,
│   │                     #           integrity_check
│   ├── artifacts/        # integrity CSVs
│   ├── reports/          # system_integrity_report.md
│   └── configs/          # default.yaml
├── california/
│   ├── scripts/          # run_ca_v4.py
│   ├── artifacts/        # metrics.csv, routing_stats.csv, regime_*.csv, ...
│   ├── reports/          # final_report.md, gates_report.md, root_cause_audit.md
│   ├── figures/          # routing plots (if generated)
│   └── logs/
└── mnist/
    ├── scripts/          # run_mn_v4.py
    ├── artifacts/        # metrics.csv, routing_stats.csv, regime_*.csv, ...
    ├── reports/          # final_report.md, gates_report.md, root_cause_audit.md
    ├── figures/
    └── logs/
```

---

## 9. Honest Summary

v4 is a discipline exercise: it correctly isolates the CA label-context pathology, introduces regression-safe training improvements, and preserves MNIST performance. It does **not** beat G2 on California. The head-collapse problem is architectural and requires a more fundamental fix in v5 (not just better hyperparameters or regularizers). The split-track design successfully prevents CA regressions from contaminating MNIST results.
