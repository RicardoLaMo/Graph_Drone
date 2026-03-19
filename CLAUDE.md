# GraphDrone — Developer Guide for Claude

## Current best ELO (2026-03-19, main, v1.18.0)

| Engine | GD ELO | TabPFN ELO | Benchmark | Tasks |
|---|---|---|---|---|
| **Regression** | **1523.2** | 1476.8 | geopoe, v1-geopoe-2026.03.19c | 36/36 |
| **Classification** | **1479.5** | 1520.5 | geopoe, v1-geopoe-2026.03.19a | 35/36 (1 OOM credit_g fold=2) |

Both engines are in `main`. Both are independent. One `GraphDrone` class dispatches via `_detect_problem_type(y)`.

---

## The two engines — exact current implementation

### Regression engine (`model.py` lines ~300–350)

- **Portfolio**: FULL (anchor, all features) + SUB0 (70%, seed 0) + SUB1 (70%, seed 1) + SUB2 (80%, seed 2) — all `foundation_regressor` (TabPFN). No tree models.
- **Router**: `contextual_transformer` (`ContextualTransformerRouter`) — learned.
- **GORA**: active. `_compute_gora_obs()` computes kappa + LID per expert in each subspace view. Included in tokens fed to the router.
- **Training**: experts fit on 100% of X. Router trained on 10% OOF split. Loss = `mse + 2.0 * relu(mse - anchor_mse)`. Patience=25, max 500 steps.
- **Configured in**: `run_geopoe_benchmark.py` regression branch + `GraphDroneConfig(router=SetRouterConfig(kind="contextual_transformer"))`.

### Classification engine (`model.py` lines ~174–186)

- **Portfolio**: FULL (anchor, all features) + SUB0 (70%, seed 0) + SUB1 (70%, seed 1) + SUB2 (80%, seed 2) — all `foundation_classifier` (TabPFN). No tree models.
- **Router**: `bootstrap_full_only` → triggers static `anchor_geo_poe_blend()` at predict time. No router training.
- **anchor_weight**: 3.0 (default in `geo_ensemble.py`).
- **Configured in**: `run_geopoe_benchmark.py` classification branch + `GraphDroneConfig(router=SetRouterConfig(kind="bootstrap_full_only"))`.

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

**DO NOT re-enable GORA for classification.**
GORA tokens are computed via kNN in each expert's subspace. For classification, with 4 experts and small N, the kNN computation on the 10% OOF split is noisy and doesn't improve the static GeoPOE path (which has no router to consume the signal anyway).

**DO NOT treat 1514.7 as a regression ELO target.**
That number is in `eval/geopoe_benchmark/run_log.txt` from v1-geopoe-2026.03.18a. It was a combined ELO (6 regression + 6 classification datasets). The regression component used `bootstrap_full_only` (= vanilla TabPFN with n_estimators=8) vs a TabPFN baseline with unspecified n_estimators. GD appeared to win only because it used more estimators. It was never a real regression improvement. The true regression baseline before 2026-03-19 was ~1440–1447.

**DO NOT confuse the two benchmark scripts.**
- `scripts/run_geopoe_benchmark.py` — canonical benchmark. TabPFN baseline uses `TabPFNRegressor/Classifier(n_estimators=8)` implicitly via `params_fp`. Version string is `GRAPHDRONE_VERSION`. Cache in `eval/geopoe_cache/`. Results in `eval/geopoe_benchmark/`.
- `scripts/run_smart_benchmark.py` — older benchmark. Used tree models (CB+XGB). Separate cache in `eval/smart_cache/`. ELOs from this runner are NOT comparable to geopoe benchmark ELOs.

---

## Benchmark commands

```bash
cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research

# Regression only (6 datasets × 3 folds)
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks regression --folds 0 1 2

# Classification only (6 datasets × 3 folds)
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks classification --folds 0 1 2

# Both engines together (12 datasets × 3 folds)
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --folds 0 1 2
```

- Cache key: `SHA256(dataset|fold|method|GRAPHDRONE_VERSION)[:16]`
- **Bump `GRAPHDRONE_VERSION`** in `run_geopoe_benchmark.py` after any model code change, or stale cached results will be used.
- Current version string: `v1-geopoe-2026.03.19c`

---

## ELO history (geopoe benchmark only — not comparable to smart benchmark)

| Date | Version | Reg ELO | Clf ELO | Notes |
|---|---|---|---|---|
| 2026-03-18 | v1-geopoe-2026.03.18a | — | — | Combined 1514.7 was NOT regression-only. See DO NOT section. |
| 2026-03-18 | v1-geopoe-2026.03.18b | — | 1455 | Learned router for clf. OOF overfitting problem. |
| 2026-03-19 | v1-geopoe-2026.03.19a | — | **1479.5** | Static GeoPOE clf. FULL+3×SUB. anchor_weight=3.0. |
| 2026-03-19 | v1-geopoe-2026.03.19b | 1482.3 | — | Multi-view reg, no residual penalty. Diamonds collapse (defer=1.0). |
| **2026-03-19** | **v1-geopoe-2026.03.19c** | **1523.2** | **1479.5** | **← current main (v1.18.0)**. Residual penalty added. GD beats TabPFN on regression. |

---

## Known gaps

1. **`quality_scores` in tokens** — `portfolio_loader.py` has a `pass` stub. All quality tokens are zero. Implementing real bagged-estimator variance would give the router better uncertainty signal.

2. **credit_g OOM** — 4×TabPFN classifiers (FULL + 3 SUBs) on 800 training samples hit CUDA OOM on fold 2. Could stagger fitting or reduce `n_estimators` for small-N classification datasets.
