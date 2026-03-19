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

## 🚀 The V2 Roadmap

To advance beyond Rank #10 without breaking the codebase via local patching, the following items are strictly tracked for the **GraphDrone V2.0** architecture:

1.  **Native Multiclass Support**: Refactoring `ExpertFactory` and `defer_integrator.py` to seamlessly handle `[N, C]` and `[N, E, C]` dimensional tensors for `predict_proba`.
2.  **Adaptive Hybrid Priors**: Intelligently gating between TabPFN and XGBoost.
3.  **Specialist Bagging**: Utilizing `N` randomized overlapping views instead of 3 fixed views.
