# GraphDrone — Developer Guide for Claude

## Current best ELO (2026-03-19, v1.19.0) — **BOTH ENGINES WIN**

| Engine | GD ELO | TabPFN ELO | Benchmark | Tasks |
|---|---|---|---|---|
| **Regression** | **1523.2** | 1476.8 | geopoe, v1-geopoe-2026.03.19c | 36/36 |
| **Classification** | **1502.2** | 1497.8 | smart benchmark, 2026.03.19-clf-multiclass-win-v1 | 36/36 |

Both engines are in `main`. One `GraphDrone` class dispatches via `_detect_problem_type(y)`.

### Classification per-dataset (v1.19, smart benchmark)
| Dataset | GD F1 | TPF F1 | Result |
|---|---|---|---|
| diabetes (binary) | 0.7549 | 0.7320 | **GD wins +0.023** |
| credit_g (binary) | 0.6787 | 0.6937 | Gap closed: was −0.054, now −0.015 |
| segment (7-class) | 0.9474 | 0.9474 | Tie |
| mfeat_factors (10-class) | 0.9861 | 0.9826 | **GD wins +0.004** |
| pendigits (10-class) | 0.9949 | 0.9959 | Near-saturation |
| optdigits (10-class) | 0.9930 | 0.9924 | **GD wins +0.001** |

---

## The two engines — exact current implementation

### Regression engine (`model.py`)

- **Portfolio**: FULL (anchor, all features) + SUB0 (70%, seed 0) + SUB1 (70%, seed 1) + SUB2 (80%, seed 2) — all `foundation_regressor` (TabPFN). No tree models.
- **Router**: `contextual_transformer` (`ContextualTransformerRouter`) — learned.
- **GORA**: active. `_compute_gora_obs()` computes kappa + LID per expert in each subspace view. Included in tokens fed to the router.
- **Training**: experts fit on 100% of X. Router trained on 10% OOF split. Loss = `mse + 2.0 * relu(mse - anchor_mse)`. Patience=25, max 500 steps.
- **Configured in**: `run_geopoe_benchmark.py` regression branch + `GraphDroneConfig(router=SetRouterConfig(kind="contextual_transformer"))`.

### Classification engine — binary path (`is_binary = n_classes == 2`)

- **Portfolio**: FULL + 3×SUB (fracs 0.8/0.85/0.9); 1×SUB (50%) for n_features < 25; anchor-only fallback when n < 500 AND n_features < 25
- **Router**: `noise_gate_router` — learned OOF NLL router with GORA
- **OOF split**: 20% holdout when n≤1500, 10% otherwise; **stratified** (credit_g fix)
- **OOF experts**: CPU-offloaded (`device="cpu"`, `n_jobs=1`) to avoid 8-model GPU OOM

### Classification engine — multiclass path (`n_classes > 2`)

- **Portfolio**: FULL + 3×SUB (fracs 0.8/0.85/0.9) — all `foundation_classifier` (TabPFN)
- **Router**: `bootstrap_full_only` → static `anchor_geo_poe_blend(anchor_weight=5.0)`
- **No router training** — zero NLL overhead, valid probability output guaranteed

---

## DO NOT rules — each backed by a measured failure

**DO NOT use `bootstrap_full_only` for regression.**
It returns only the FULL expert's predictions, `defer=0.0` always. That is vanilla TabPFN. You gain nothing. Prior state: every regression run before 2026-03-19 used this and scored 1440–1447 (GD loses to TabPFN).

**DO NOT add CatBoost or XGBoost to either engine.**
Both are weaker than TabPFN on these benchmark datasets. The router cannot tell when to trust them and mis-routes in geometrically complex regions. Smart benchmark (2026-03-18, v2026.03.18h) used CB+XGB → regression ELO dropped to 1440 (TabPFN 1560). Removing them entirely is correct.

**DO NOT use `contextual_transformer` router for classification.**
Overfits on the 10% OOF split. diabetes/credit_g have ~78–100 OOF rows; the router has ~2,946 parameters → 37:1 param/sample ratio → pathological defer solutions (0.0003 on one fold, 0.9998 on the next). Static GeoPOE is strictly better for small datasets.

**DO NOT omit the MSE residual penalty for regression.**
Without `2.0 * relu(mse - anchor_mse)`, the router drives `defer→1.0` whenever SUB views get lucky on the 10% split. First run (v1-geopoe-2026.03.19b, no penalty): diamonds fold 0/2 had defer=1.0, R² collapsed from 0.98 to 0.94. Penalty added in v1-geopoe-2026.03.19c: diamonds R² restored, ELO 1482→1523.

**DO NOT re-enable GORA for multiclass classification.**
GORA tokens are computed via kNN in each expert's subspace. For multiclass with static GeoPOE (no router), the signal has no consumer. GORA is valid in the binary path where the learned router can use it.

**DO NOT treat 1514.7 as a regression ELO target.**
That number is in `eval/geopoe_benchmark/run_log.txt` from v1-geopoe-2026.03.18a. It was a combined ELO (6 regression + 6 classification datasets). The regression component used `bootstrap_full_only` (= vanilla TabPFN). GD appeared to win only because it used more estimators. The true regression baseline before 2026-03-19 was ~1440–1447.

**DO NOT confuse the two benchmark scripts.**
- `scripts/run_geopoe_benchmark.py` — canonical regression benchmark. TabPFN baseline uses `n_estimators=8`. Version string `GRAPHDRONE_VERSION`. Cache in `eval/geopoe_cache/`.
- `scripts/run_smart_benchmark.py` — classification benchmark (used for the v1.19 clf ELO). Cache in `eval/smart_cache/`. ELOs from different runners are NOT directly comparable.

---

## Benchmark commands

```bash
# Regression (6 datasets × 3 folds) — geopoe benchmark
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks regression --folds 0 1 2

# Classification (6 datasets × 3 folds) — smart benchmark
PYTHONPATH=src python scripts/run_smart_benchmark.py --folds 0 1 2

# Quick smoke test (3 datasets × 1 fold)
PYTHONPATH=src python scripts/run_smart_benchmark.py --quick --folds 0
```

- **Bump `GRAPHDRONE_VERSION`** in the relevant script after any model code change, or stale cached results will be used.
- Current regression version: `v1-geopoe-2026.03.19c`
- Current classification version: `2026.03.19-clf-multiclass-win-v1`

---

## ELO history

| Date | Version | Reg ELO | Clf ELO | Notes |
|---|---|---|---|---|
| 2026-03-18 | v1-geopoe-2026.03.18a | — | — | Combined 1514.7 was NOT regression-only. See DO NOT section. |
| 2026-03-18 | v1-geopoe-2026.03.18b | — | 1455 | Learned router for clf. OOF overfitting problem. |
| 2026-03-19 | v1-geopoe-2026.03.19a | — | 1479.5 | Static GeoPOE clf. FULL+3×SUB. anchor_weight=3.0. |
| 2026-03-19 | v1-geopoe-2026.03.19b | 1482.3 | — | Multi-view reg, no residual penalty. Diamonds collapse (defer=1.0). |
| 2026-03-19 | v1-geopoe-2026.03.19c | **1523.2** | — | Residual penalty added. GD beats TabPFN on regression. v1.18.0. |
| **2026-03-19** | **v1.19.0** | **1523.2** | **1502.2** | **← current main**. Binary/multiclass split. Both engines win. |

---

## Known gaps (future work)

1. **`quality_scores` in tokens** — `portfolio_loader.py` has a `pass` stub. All quality tokens are zero. Real bagged-estimator variance would give the router better uncertainty signal for binary classification.

2. **credit_g still lags TabPFN** (−0.015). Root cause: 20 features × 3 SUBs at 70-80% provides minimal diversity; OOF holdout ~160 rows even after stratify fix. Further improvement: Latin square permutations (Idea E in `research/tabicl_inspiration.md`).

3. **TabICL-inspired ideas** (`research/tabicl-inspiration`) — class shift + YJ view + temperature bundle tested: net −4.3 ELO on smart benchmark. YJ 5th expert drags segment. Ablation roadmap in `research/tabicl_inspiration.md`. Class-shift-only is promising for 10-class datasets.
