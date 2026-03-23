# GraphDrone — Developer Guide for Claude

## Project Skill Bridge

The repo now exports reusable research-operation skills into `.claude/skills/`.

- Source of truth: `skills/`
- Claude-facing copy: `.claude/skills/`
- Refresh command:
  ```bash
  python scripts/export_skills_to_claude.py
  ```

For methodology-heavy tasks, read `.claude/skills/INDEX.md` first, then the relevant skill pack:
- generic parent skills:
  - `research-platform-ops`
  - `mechanism-first-diagnosis`
  - `research-memory-ledger`
  - `benchmark-evidence-governance`
- GraphDrone specializations:
  - `graphdrone-research-ops`
  - `graphdrone-mechanism-diagnosis`
  - `graphdrone-research-memory`
  - `graphdrone-benchmark-governance`

## Current best ELO (2026-03-23, v1.3.0) — **BOTH ENGINES WIN**

| Engine | GD ELO | TabPFN ELO | Benchmark | Tasks |
|---|---|---|---|---|
| **Regression** | **1523.2** | 1476.8 | geopoe, v1-geopoe-2026.03.19c | 36/36 |
| **Classification** | **1512.4** | 1487.6 | smart benchmark, 2026.03.23-clf-v1.3-phase3b-r3 | 54/54 |

Both engines are in `main`. One `GraphDrone` class dispatches via `_detect_problem_type(y)`.

### Classification per-dataset (v1.3.0, smart benchmark — 9 datasets × 3 folds)
| Dataset | GD F1 | TPF F1 | GD log_loss | TPF log_loss | Result |
|---|---|---|---|---|---|
| diabetes (binary) | 0.7539 | 0.7320 | 0.4738 | 0.4736 | **GD wins F1 +0.022** |
| credit_g (binary) | 0.7226 | 0.6937 | 0.4805 | 0.4760 | **GD wins F1 +0.029** (gap closed) |
| segment (7-class) | 0.9474 | 0.9474 | 0.1383 | 0.1442 | Tie F1, **GD wins log_loss** |
| mfeat_factors (10-class) | 0.9843 | 0.9826 | 0.0386 | 0.0442 | **GD wins both** |
| pendigits (10-class) | 0.9949 | 0.9959 | 0.0268 | 0.0261 | Near-saturation |
| optdigits (10-class) | 0.9927 | 0.9924 | 0.0260 | 0.0262 | **GD wins F1** |
| maternal_health_risk (3-class, 7f) | 0.8644 | 0.8609 | 0.3968 | 0.3893 | **GD wins F1** |
| website_phishing (3-class, 10f) | 0.9230 | 0.9239 | 0.1865 | 0.1876 | GD wins log_loss |
| SDSS17 (3-class, 12f) | 0.9674 | 0.9672 | 0.0920 | 0.0951 | **GD wins both** |

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
- **Quality tokens**: `BaggedClassifierPredictor` (4× `TabPFNClassifier(n_estimators=2)`) provides per-expert prediction variance for router signal (v1.20)

### Classification engine — multiclass path (`n_classes > 2`)

- **Portfolio**: feature-count-dependent (v1.20):
  - ≤10 features → FULL only (SUBs at 80-90% have no diversity on 6-10 features)
  - ≤14 features → FULL + 1×SUB @ 60% (meaningful feature dropout)
  - >14 features → FULL + 3×SUB @ 0.8/0.85/0.9 (unchanged for high-dim)
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
- Current classification version: `2026.03.23-clf-v1.3-phase3b-r3`

---

## ELO history

| Date | Version | Reg ELO | Clf ELO | Notes |
|---|---|---|---|---|
| 2026-03-18 | v1-geopoe-2026.03.18a | — | — | Combined 1514.7 was NOT regression-only. See DO NOT section. |
| 2026-03-18 | v1-geopoe-2026.03.18b | — | 1455 | Learned router for clf. OOF overfitting problem. |
| 2026-03-19 | v1-geopoe-2026.03.19a | — | 1479.5 | Static GeoPOE clf. FULL+3×SUB. anchor_weight=3.0. |
| 2026-03-19 | v1-geopoe-2026.03.19b | 1482.3 | — | Multi-view reg, no residual penalty. Diamonds collapse (defer=1.0). |
| 2026-03-19 | v1-geopoe-2026.03.19c | **1523.2** | — | Residual penalty added. GD beats TabPFN on regression. v1.18.0. |
| 2026-03-19 | v1.19.0 | 1523.2 | 1502.2 | Binary/multiclass split. Both engines win. |
| 2026-03-19 | v1.20.0 | 1523.2 | 1503.3 | Feature-count portfolio + bagged quality tokens. 9 datasets. |
| 2026-03-23 | v1.3.0-rc | 1523.2 | 1507.9 | TaskConditionedPrior + confidence-gated defer penalty. credit_g gap −0.004→−0.0014. |
| **2026-03-23** | **v1.3.0** | **1523.2** | **1512.4** | **← current branch**. OOF threshold calibration (Phase 3B). credit_g gap fully closed, GD leads TabPFN +0.029. |

---

## Known gaps (future work)

1. **credit_g gap CLOSED** (v1.3.0). OOF threshold calibration moved threshold to 0.61–0.68 (credit_g has 30% positive rate). GD now leads TabPFN +0.029 F1. Remaining open: log_loss on credit_g still lags (threshold shifts improve F1 but not calibration).

2. **Multiclass log_loss on low-dim** (maternal_health_risk, SDSS17 below TabPFN). Static GeoPOE at anchor_weight=5.0 is well-calibrated for F1 but slightly over-confident. ScalarGatingAdapter (Phase 3) was designed to learn this but had a bug (`use_learned` path exclusion); fixed in `exp/clf-mc-scalar-gating` but not yet benchmarked successfully.

3. **TabICL-inspired ideas** (`research/tabicl-inspiration`) — class shift + YJ view + temperature bundle tested: net −4.3 ELO on smart benchmark. YJ 5th expert drags segment. Class-shift-only is promising for 10-class datasets.
