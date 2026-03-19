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

### **🏷 `v1.18.0` (GraphDrone V1.18 — Dual-Engine)**
*   **Status**: Previous stable (regression wins, classification behind TabPFN).
*   **Regression ELO**: GD **1523.2** vs TabPFN 1476.8 — **GD WINS** (6 datasets × 3 folds).
*   **Classification ELO**: GD 1479.5 vs TabPFN 1520.5 — TabPFN leads.
*   **Engine**: Multi-view TabPFN (FULL + 3×SUB) + GORA + residual-protected router for regression; GeoPOE for classification.

### **🏷 `v1.19.0` (GraphDrone V1.19 — Binary/Multiclass Split)**
*   **Status**: Current stable. **Both engines win**.
*   **Regression ELO**: GD **1523.2** vs TabPFN 1476.8 — **GD WINS** (unchanged from v1.18).
*   **Classification ELO**: GD **1502.2** vs TabPFN 1497.8 — **GD WINS**.
*   **Key changes vs v1.18**:
    - Binary classification: learned OOF NLL router (GORA + noise_gate_router)
    - Multiclass: static anchor GeoPOE (anchor_weight=5.0, FULL+3×SUB fracs 0.8/0.85/0.9)
    - credit_g gap closed: OOF holdout 20% for n≤1500 + stratified split
    - OOF experts CPU-offloaded to eliminate 8-model GPU OOM
*   **Per-dataset highlights**: diabetes F1 +0.023 vs TabPFN; mfeat_factors +0.004; segment tie; credit_g gap −0.054→−0.015.

---

## 🚀 The V2 Roadmap

To advance beyond Rank #10 without breaking the codebase via local patching, the following items are strictly tracked for the **GraphDrone V2.0** architecture:

1.  **Native Multiclass Support**: Refactoring `ExpertFactory` and `defer_integrator.py` to seamlessly handle `[N, C]` and `[N, E, C]` dimensional tensors for `predict_proba`.
2.  **Adaptive Hybrid Priors**: Intelligently gating between TabPFN and XGBoost.
3.  **Specialist Bagging**: Utilizing `N` randomized overlapping views instead of 3 fixed views.
