# GraphDrone Versioning & Benchmark Protocol

## 🏷 Current Versions (Stable Git Tags)

To prevent the "Looping Error" and performance fluctuations, GraphDrone versions are now strictly bound to specific Git Tags and Commits.

### **🏷 `v1.0.0-gora` (GraphDrone V1.0 - Geometric Engine)**
*   **Commit**: `8e9a707e`
*   **Status**: Legacy High-Performance (Rank #10).
*   **Engine**: Consolidated Core + GORA Geometric Observers.
*   **Reference Elo**: **1438 - 1453** (Validated on NVIDIA H200 NVL).
*   **Limitations**: Does *not* natively support 3D Multi-Class Probabilities. Designed and scored on Binary Classification and Regression tasks.

### **🏷 `v1.0.0-pure` (GraphDrone V1.0 - Pure Foundation)**
*   **Commit**: `a2ea6978` (Current `main`)
*   **Status**: Stable Production Baseline.
*   **Engine**: Pure TabPFN Specialists. No GORA. No learned router.
*   **Reference Rank**: **#30** (Full TabArena 51-dataset Portfolio).
*   **Philosophy**: Clean, simple, and 100% stable across all task types (including Multiclass).

---

### **🏷 `v1.18` — Current production (2026-03-19)**
*   **Git tag**: `v1.18` on `main`
*   **Package version**: `1.18.0` (`pyproject.toml`)
*   **Benchmark version string**: `v1-geopoe-2026.03.19c` (regression) / `v1-geopoe-2026.03.19a` (classification)
*   **Regression ELO**: **1523.2** vs TabPFN 1476.8 — GD wins (geopoe benchmark, 36/36 tasks)
*   **Classification ELO**: **1479.5** vs TabPFN 1520.5 (geopoe benchmark, 35/36 tasks)
*   **Regression engine**: FULL + 3×SUB TabPFN (seeds 0/1/2, fracs 0.7/0.7/0.8) + GORA + contextual_transformer router + MSE residual penalty
*   **Classification engine**: FULL + 3×SUB TabPFN + static anchor_geo_poe_blend (anchor_weight=3.0, no router training)
*   **Smoke test**: `python tests/test_smoke_v118.py` — 24/24 checks

---

## 🚀 The V2 Roadmap

To advance beyond Rank #10 without breaking the codebase via local patching, the following items are strictly tracked for the **GraphDrone V2.0** architecture:

1.  **Native Multiclass Support**: Refactoring `ExpertFactory` and `defer_integrator.py` to seamlessly handle `[N, C]` and `[N, E, C]` dimensional tensors for `predict_proba`.
2.  **Adaptive Hybrid Priors**: Intelligently gating between TabPFN and XGBoost.
3.  **Specialist Bagging**: Utilizing `N` randomized overlapping views instead of 3 fixed views.
