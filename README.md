# GraphDrone: Multi-View Mixture-of-Specialists for Tabular Data

**GraphDrone** is a high-performance meta-model for tabular classification and regression. It wraps [TabPFN](https://github.com/automl/TabPFN) with a learned cross-attention router that dynamically blends predictions from multiple specialist views of the feature space.

## Benchmark Results

Evaluated on **TabArena** — 43 binary/regression datasets × 3 folds (129 tasks), compared against 57 methods including tuned ensembles, AutoML suites, and GPU-accelerated models.

| Version | ELO | Rank (58 methods) | Win Rate | Notes |
|---------|-----|-------------------|----------|-------|
| v1.0.0-gora | 1420.3 | ~19 | — | Initial release |
| v1-width.1 | 1441.1 | 18.8 | 68.7% | P0: cross-attn fix + BCE loss |
| **v1-width.2** | **1458.9** | **#10** | **70.5%** | P1: vectorised GORA observers |

GraphDrone (default, no tuning) sits at **#10 out of 58 methods**, above all tuned/ensembled gradient boosting methods (XGBoost, CatBoost, LightGBM) and default neural nets, and below only 4-hour AutoML suites and heavily tuned + ensembled GPU models.

<details>
<summary>Top 15 leaderboard context</summary>

| Rank | Method | ELO |
|------|--------|-----|
| 1 | AutoGluon 1.5 (extreme, 4h) | 1694 |
| 2 | TABICLV2 (default) | 1622 |
| 3 | AutoGluon 1.4 (extreme, 4h) | 1613 |
| 4 | REALTABPFN-V2.5 (tuned + ensemble) | 1590 |
| 5 | REALTABPFN-V2.5 (tuned) | 1539 |
| 6 | REALTABPFN-V2.5 (default) | 1528 |
| 7 | AutoGluon 1.4 (best, 4h) | 1521 |
| 8 | REALMLP_GPU (tuned + ensemble) | 1510 |
| 9 | TABDPT_GPU (tuned + ensemble) | 1475 |
| **10** | **GraphDrone v1-width.2 (default)** | **1459** |
| 11 | TABDPT_GPU (tuned) | 1430 |
| 12 | TABM_GPU (tuned + ensemble) | 1425 |
| 13 | REALMLP_GPU (tuned) | 1417 |
| 14 | GBM (tuned + ensemble) | 1414 |
| 15 | CAT (tuned + ensemble) | 1395 |

</details>

---

## How It Works

GraphDrone builds a **portfolio of specialist experts**, each trained on a different view of the feature space (full features, first-half, second-half). At inference time, a cross-attention **noise-gate router** blends specialist predictions with the full-view anchor, guided by geometric observers (kappa dimensionality, Local Intrinsic Dimensionality) computed per query point.

```
Input X ──► Specialist 1 (FULL view)  ──► prediction_1 ─┐
         ├─► Specialist 2 (V1 half)   ──► prediction_2 ─┤
         └─► Specialist 3 (V2 half)   ──► prediction_3 ─┤
                                                          ▼
              GORA observers (kappa, LID) ──► Cross-Attention Router
                                                          │
                                             defer_prob × blend + (1-defer_prob) × anchor
                                                          │
                                                      prediction
```

Key components:
- **GORA** (Geometric Observer for Router Awareness): k-NN based kappa (intrinsic dimensionality ratio) and LID (Local Intrinsic Dimensionality) signal how "complex" the local manifold is around each query
- **Noise-Gate Router**: cross-attention over specialist tokens; `defer_prob` controls how much weight shifts from anchor to specialists
- **Task-aware loss**: BCE for binary classification, MSE for regression

---

## Installation

```bash
git clone https://github.com/RicardoLaMo/Graph_Drone.git
cd Graph_Drone
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, TabPFN ≥ 2.0, scikit-learn ≥ 1.3

---

## Quick Start

```python
import numpy as np
from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig

# Minimal usage — auto-detects binary vs regression
config = GraphDroneConfig(
    full_expert_id="FULL",
    router=SetRouterConfig(kind="noise_gate_router")
)
model = GraphDrone(config)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# With diagnostics
result = model.predict(X_test, return_diagnostics=True)
print(result.diagnostics["effective_defer_rate"])  # how much router defers to specialists
```

### TabArena adapter (for benchmarking)

```python
from graphdrone_fit.adapters.tabarena import GraphDroneTabArenaAdapter

# Drop-in for TabArena ExperimentBatchRunner
adapter = GraphDroneTabArenaAdapter(
    n_estimators=8,
    router_kind="noise_gate_router"
)
```

### Custom expert portfolio

```python
from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig, ExpertBuildSpec, ViewDescriptor, IdentitySelectorAdapter

n_features = X_train.shape[1]
mid = n_features // 2
full_idx = tuple(range(n_features))
v1_idx   = tuple(range(mid))
v2_idx   = tuple(range(mid, n_features))

specs = (
    ExpertBuildSpec(
        descriptor=ViewDescriptor(expert_id="FULL", family="FULL",
            view_name="Full dataset", is_anchor=True,
            input_dim=n_features, input_indices=full_idx),
        model_kind="foundation_classifier",   # or "foundation_regressor"
        input_adapter=IdentitySelectorAdapter(indices=full_idx),
        model_params={"n_estimators": 8, "device": "cuda"}
    ),
    ExpertBuildSpec(
        descriptor=ViewDescriptor(expert_id="V1", family="structural_subspace",
            view_name="First half features",
            input_dim=len(v1_idx), input_indices=v1_idx),
        model_kind="foundation_classifier",
        input_adapter=IdentitySelectorAdapter(indices=v1_idx),
        model_params={"n_estimators": 8, "device": "cuda"}
    ),
    ExpertBuildSpec(
        descriptor=ViewDescriptor(expert_id="V2", family="structural_subspace",
            view_name="Second half features",
            input_dim=len(v2_idx), input_indices=v2_idx),
        model_kind="foundation_classifier",
        input_adapter=IdentitySelectorAdapter(indices=v2_idx),
        model_params={"n_estimators": 8, "device": "cuda"}
    ),
)

config = GraphDroneConfig(full_expert_id="FULL", router=SetRouterConfig(kind="noise_gate_router"))
model = GraphDrone(config)
model.fit(X_train, y_train, expert_specs=specs, problem_type="binary")
```

---

## Reproducing Benchmark Results

```bash
# Sprint validator (8 datasets × fold 0, ~15 min on 6 GPUs)
conda run -n h200_tabpfn python scripts/run_sprint.py

# Full TabArena benchmark (43 datasets × 3 folds, ~2 hr on 6 GPUs)
conda run -n h200_tabpfn python scripts/run_tabarena_parallel.py
```

Results are saved to `eval/tabarena_<expname>/tabarena_leaderboard_full.csv`.

---

## Research Log

All experiments tracked in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

| Tag | Sprint ELO | Full ELO | Key Change |
|-----|-----------|---------|------------|
| v1.0.0-gora | — | 1420.3 | Baseline |
| v1-width.1 | 1455.7 | 1441.1 | Cross-attn fix + BCE loss (P0-AB) |
| **v1-width.2** | **1462.4** | **1458.9** | Vectorised kappa/LID observers (P1-C) |

---

## Hardware

Developed and benchmarked on NVIDIA H200 NVL (8× GPU node). TabPFN inference is GPU-accelerated; router training uses CUDA when available and falls back to CPU.

---

*GraphDrone Research — v2026.03.15*
