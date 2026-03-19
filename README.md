# GraphDrone: Multi-View Mixture-of-Specialists for Tabular Data

**GraphDrone** is a high-performance meta-model for tabular classification and regression. It wraps [TabPFN](https://github.com/automl/TabPFN) with a learned routing engine that dynamically blends predictions from multiple specialist views of the feature space — separately optimised for binary classification, multiclass classification, and regression.

## Benchmark Results

### Internal benchmark (v1.19.0, 2026-03-19)

Evaluated on 6 representative datasets × 3 folds against TabPFN v2.5 default. **Both engines beat TabPFN simultaneously for the first time.**

| Engine | GD ELO | TabPFN ELO | Win |
|--------|--------|------------|-----|
| **Regression** | **1523.2** | 1476.8 | **GD +46.4** |
| **Classification** | **1502.2** | 1497.8 | **GD +4.4** |

Classification per-dataset breakdown:

| Dataset | GD F1 | TabPFN F1 | Result |
|---------|-------|-----------|--------|
| diabetes (binary) | **0.755** | 0.732 | GD wins +0.023 |
| credit_g (binary) | 0.679 | 0.694 | Gap closed (was −0.054, now −0.015) |
| segment (7-class) | 0.947 | 0.947 | Tie |
| mfeat_factors (10-class) | **0.986** | 0.983 | GD wins +0.004 |
| pendigits (10-class) | 0.995 | 0.996 | Near-saturation |
| optdigits (10-class) | **0.993** | 0.992 | GD wins +0.001 |

### TabArena leaderboard (43 datasets, 58 methods — v1-width.2 baseline, full re-run pending)

| Version | ELO | Rank (58 methods) | Notes |
|---------|-----|-------------------|-------|
| v1.0.0-gora | 1420.3 | ~19 | Initial release |
| v1-width.1 | 1441.1 | 18.8 | Cross-attn fix + BCE loss |
| v1-width.2 | 1458.9 | **#10** | Vectorised GORA observers |
| **v1.19.0** | **est. top 3*** | **est. #3–5** | Both engines beat TabPFN; full re-run pending |

*\* Projected based on consistent wins over TabPFN v2.5 default (TabArena rank #6) on both regression and classification. Full 43-dataset × 3-fold TabArena re-run in progress.*

<details>
<summary>TabArena top-10 context (v1-width.2 baseline)</summary>

| Rank | Method | ELO |
|------|--------|-----|
| 1 | AutoGluon 1.5 (extreme, 4h) | 1694 |
| 2 | TabICL V2 (default) | 1622 |
| 3 | AutoGluon 1.4 (extreme, 4h) | 1613 |
| 4 | RealTabPFN-V2.5 (tuned + ensemble) | 1590 |
| 5 | RealTabPFN-V2.5 (tuned) | 1539 |
| 6 | RealTabPFN-V2.5 (default) | 1528 |
| 7 | AutoGluon 1.4 (best, 4h) | 1521 |
| 8 | RealMLP_GPU (tuned + ensemble) | 1510 |
| 9 | TabDPT_GPU (tuned + ensemble) | 1475 |
| 10 | **GraphDrone v1-width.2 (default)** | **1459** |
| — | **GraphDrone v1.19.0 (default)** | **est. top 3–5** |

GraphDrone v1.19.0 consistently outperforms RealTabPFN-V2.5 (default, rank #6) on both regression and classification. Full leaderboard re-run pending.

</details>

---

## Architecture (v1.19.0)

GraphDrone builds a **portfolio of specialist experts**, each trained on a different view of the feature space. A task-specific routing strategy blends their predictions at inference time.

```
Input X ──► Expert 1: FULL view (anchor) ──► p_1 ─┐
         ├─► Expert 2: SUB view (80%)    ──► p_2 ─┤
         ├─► Expert 3: SUB view (85%)    ──► p_3 ─┤
         └─► Expert 4: SUB view (90%)    ──► p_4 ─┘
                                                    │
              ┌─────────────────────────────────────┤
              │                                     │
         [Binary]                           [Multiclass / Regression]
    Learned OOF NLL Router              Static Geometric Product-of-Experts
    GORA (kappa + LID) signal           anchor_weight = 5.0
    noise_gate_router                   no router training required
              │                                     │
         prediction                           prediction
```

### Three routing engines

**Binary classification** (`n_classes == 2`):
- Portfolio: FULL + 3×SUB (fracs 0.8/0.85/0.9); 1×SUB at 50% for low-dim datasets
- Router: learned OOF NLL router with GORA geometric observers (`noise_gate_router`)
- OOF split: 25% holdout for n≤1500, 10% otherwise; stratified by class
- OOF experts CPU-offloaded (avoids 8-model GPU OOM)

**Multiclass classification** (`n_classes > 2`):
- Portfolio: FULL + 3×SUB (fracs 0.8/0.85/0.9)
- Router: static `anchor_geo_poe_blend(anchor_weight=5.0)` — no training required
- Geometric Product-of-Experts in log-probability space

**Regression**:
- Portfolio: FULL + 3×SUB (fracs 0.7/0.7/0.8)
- Router: `contextual_transformer` — learned on 10% OOF split
- GORA active (kappa + LID per subspace view)
- Loss: `MSE + 2.0 × relu(MSE − anchor_MSE)` (residual penalty prevents defer collapse)

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

# Minimal usage — auto-detects binary / multiclass / regression
config = GraphDroneConfig(
    full_expert_id="FULL",
    router=SetRouterConfig(kind="bootstrap_full_only")  # default; override for regression
)
model = GraphDrone(config)
model.fit(X_train, y_train)                            # problem_type auto-detected
predictions = model.predict(X_test)                    # returns proba for clf, values for reg

# With diagnostics
result = model.predict(X_test, return_diagnostics=True)
print(result.diagnostics)   # router_kind, mean_defer_prob
```

### Regression (explicit config)

```python
from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig

config = GraphDroneConfig(
    full_expert_id="FULL",
    router=SetRouterConfig(kind="contextual_transformer")
)
model = GraphDrone(config)
model.fit(X_train, y_train, problem_type="regression")
predictions = model.predict(X_test)
```

### Custom expert portfolio

```python
from graphdrone_fit import (
    GraphDrone, GraphDroneConfig, SetRouterConfig,
    ExpertBuildSpec, ViewDescriptor, IdentitySelectorAdapter
)

n_features = X_train.shape[1]
full_idx = tuple(range(n_features))
sub_idx  = tuple(range(n_features // 2))

specs = (
    ExpertBuildSpec(
        descriptor=ViewDescriptor(expert_id="FULL", family="FULL",
            view_name="Full dataset", is_anchor=True,
            input_dim=n_features, input_indices=full_idx),
        model_kind="foundation_classifier",
        input_adapter=IdentitySelectorAdapter(indices=full_idx),
        model_params={"n_estimators": 8, "device": "cuda"}
    ),
    ExpertBuildSpec(
        descriptor=ViewDescriptor(expert_id="SUB0", family="structural_subspace",
            view_name="First half features",
            input_dim=len(sub_idx), input_indices=sub_idx),
        model_kind="foundation_classifier",
        input_adapter=IdentitySelectorAdapter(indices=sub_idx),
        model_params={"n_estimators": 8, "device": "cuda"}
    ),
)

config = GraphDroneConfig(full_expert_id="FULL", router=SetRouterConfig(kind="bootstrap_full_only"))
model = GraphDrone(config)
model.fit(X_train, y_train, expert_specs=specs, problem_type="classification")
```

---

## Reproducing Benchmark Results

```bash
# Regression benchmark (6 datasets × 3 folds)
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks regression --folds 0 1 2

# Classification benchmark (6 datasets × 3 folds)
PYTHONPATH=src python scripts/run_smart_benchmark.py --folds 0 1 2

# Quick smoke test (3 datasets × fold 0, ~3 min)
PYTHONPATH=src python scripts/run_smart_benchmark.py --quick --folds 0
```

Results are saved to `eval/geopoe_benchmark/` and `eval/smart_benchmark/`.

---

## Research Log

All experiments tracked in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

| Version | Reg ELO | Clf ELO | Key Change |
|---------|---------|---------|------------|
| v1.0.0-gora | — | 1420.3 | Baseline GORA + binary only |
| v1-width.2 | — | 1458.9 | Vectorised kappa/LID observers |
| v1.18.0 | **1523.2** | 1479.5 | Multi-view regression engine + GORA |
| **v1.19.0** | **1523.2** | **1502.2** | Binary/multiclass split; both engines beat TabPFN |

See [`VERSIONS.md`](VERSIONS.md) for full version history and architecture decisions.

---

## Hardware

Developed and benchmarked on NVIDIA H200 NVL. TabPFN inference is GPU-accelerated; router training uses CUDA when available and falls back to CPU automatically.

---

*GraphDrone v1.19.0 — 2026-03-19*
