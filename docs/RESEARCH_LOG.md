# GraphDrone v1-width Research Log

Each row is one experiment. "Sprint ELO" = 8-dataset × fold-0 canary.
"Full ELO" = 43-dataset × 3-fold official run.
**Sprint ELO Δ is always relative to the current integration baseline (latest merged tag), not v1.0.0-gora.**

## Baseline

| Label | Branch/Tag | Sprint ELO | Full ELO | Full Rank | Notes |
|-------|-----------|-----------|---------|-----------|-------|
| **v1.0.0-gora** | `tag:v1.0.0-gora` | — | **1420.3** | **19 / 58** | 43 datasets × 3 folds, 129 tasks |
| **v1-width.1** | `tag:v1-width.1` | **1455.7** | **1441.1** | **18.8 / 58** | P0-AB: attn_weights + BCE + NaN guard |
| **v1-width.2** | `tag:v1-width.2` | **1462.4** | **1458.9** | **17.8 / 58** | P1-C: vectorised kappa/LID; **current integration baseline** |

---

## Experiment Log

| ID | Branch | Hypothesis | Changed Files | Sprint ELO Δ | Full ELO Δ | Status | Decision |
|----|--------|-----------|--------------|-------------|-----------|--------|----------|
| P0-A | `exp/p0-attn-fix` | `attn_weights` (not `v`) → router has true inter-expert attention | `set_router.py:59` | — | — | ✅ merged into P0-AB | — |
| P0-B | `exp/p0-loss-fn` | BCE loss for binary tasks → router optimises correct metric | `model.py:111` | — | — | ✅ merged into P0-AB | — |
| P0-AB | `exp/p0-combined-v2` | Both P0-A + P0-B together (+ NaN guard in BCE path) | `set_router.py`, `model.py` | **+35.4** | **+20.8** (1441.1, rank 18.8/58, winrate 68.7%) | ✅ keep | merged → `v1-width.1` |
| P1-A | `exp/p1-descriptor-norm` | Normalise input_dim/preferred_k in descriptor token (raw int → fraction) | `token_builder.py` | **−31.1** (1424.6 abs, vs baseline 1455.7) | — | ❌ reject | Hurts on top of P0-AB; tag `exp/rejected/p1-a` |
| P1-B | `exp/p1-snr-wire` | Wire SNR via k-NN label statistics (mean_y, var_y) → router gets reliability signal | `model.py` | **−0.7** (1455.0 abs, vs baseline 1455.7) | — | ❌ reject | Neutral alone, hurts in combo (P1-BC: −5.1); SNR adds latency without gain |
| P1-BC | `exp/p1-bc-combined` | P1-B + P1-C together | `model.py`, `observers.py` | **−5.1** (1450.6 abs, vs baseline 1455.7) | — | ❌ reject | Combining SNR with vec-observers is net negative |
| P1-C | `exp/p1-kappa-vec` | Vectorise kappa SVD + LID loop → batch numpy, ~1.5–5x faster on large datasets | `observers.py` | **+6.7** (1462.4 abs, vs baseline 1455.7) | **+17.8** (1458.9, rank 17.8/58, winrate 70.5%) | ✅ keep | merged → `v1-width.2` |
| P2-A | `exp/p2-random-views` | Random 70% feature subsets (seeds 42/43) instead of fixed half-split for V1/V2 | `adapters/tabarena.py` | **−79.4** (1383.0 abs, vs baseline 1462.4) | — | ❌ reject | Random views collapse router spatial signal; tag `exp/rejected/p2-a` |
| P2-B | `exp/p2-n16` | Increase n_estimators 8→16 globally | `adapters/tabarena.py` | **+69.4** (1531.8 abs) | **−1.7** (1457.2, rank 17.9/58, winrate 70.3%) | ❌ reject | Sprint overfit to large datasets; full run flat/negative |
| P2-C | `exp/p2-pca-view` | PCA 4th expert + `_view_transforms` for correct GORA space | `adapters/tabarena.py`, `model.py` | **−26.4** (1436.0 abs) | — | ❌ reject | Extra expert adds router noise; tag `exp/rejected/p2-c` |
| P3-A | `exp/p3-router-val-split` | Router validation split 10%→20% → more stable gradient signal | `model.py:84` | **+77.6** (1540.0 abs, vs baseline 1462.4) | — | 🔁 full run pending | Strong sprint win |
| P3-B | `exp/p3-log-descriptor` | Log1p-scale `input_dim`/`preferred_k` in descriptor token | `token_builder.py` | **−78.7** (1383.7 abs) | — | ❌ reject | Descriptor encoding hurts as much as P1-A normalisation; tag `exp/rejected/p3-b` |
| P3-C | `exp/p3-loss-curriculum` | MSE warm-up (50 epochs) before BCE for binary → better router initialisation | `model.py` | **+69.8** (1532.2 abs, vs baseline 1462.4) | — | 🔁 full run pending | Strong sprint win |
| P3-D | `exp/p3-adaptive-k` | Adaptive GORA k = clip(sqrt(N)/2, 5, 30) | `model.py`, `adapters/tabarena.py` | **+2.4** (1464.8 abs) | — | ⚠️ borderline | Marginal sprint gain; skip standalone full run, consider in combo |
| P3-E | `exp/p3-adaptive-nest` | Adaptive n_estimators: 4/8/16 for N<500/500–5000/>5000 | `adapters/tabarena.py` | **+106.2** (1568.6 abs, vs baseline 1462.4) | **−0.5** (1458.4, rank 17.9/58, winrate 70.4%) | ❌ reject | Sprint canary biased toward large datasets; n=4 on small datasets offsets n=16 gains; tag `exp/rejected/p3-e` |

---

## How to Add an Entry

1. Create branch `exp/<id>-<short-name>` from `v1-width`.
2. Make the change. Commit with message `exp(<id>): <hypothesis>`.
3. Run `python scripts/run_sprint.py` → copy Sprint ELO Δ into the table.
4. If Sprint ELO Δ > 0: run full benchmark (`--retry` auto-skips cached tasks).
5. Fill Full ELO Δ and set Status = ✅ keep / ❌ reject / 🔁 iterate.
6. If ✅: `git merge --no-ff exp/<id>` into `v1-width`, tag `v1-width.<N>`.
7. If ❌: tag branch `exp/rejected/<id>`, leave for reference.

---

## Sprint Dataset Contract

Fixed 8 datasets, fold 0 only. **Never change this set between experiments.**

| Dataset | Type | Why included |
|---------|------|-------------|
| `kddcup09_appetency` | binary | Biggest GD gap vs baseline |
| `bank-marketing` | binary | Moderate gap, high traffic |
| `Diabetes130US` | binary | Large NaN dataset, GD loses badly |
| `diabetes` | binary | Small, fast, GD was competitive — catches regression |
| `concrete_compressive_strength` | regression | Small, fast regression canary |
| `airfoil_self_noise` | regression | Small, fast regression canary |
| `credit-g` | binary | Medium binary canary |
| `APSFailure` | binary | GD near-best — catches regression on large datasets |

Sprint runtime target: **< 5 min** (8 tasks × fold 0, 1 per GPU, all parallel).
