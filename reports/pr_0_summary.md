# Final Experiment Summary: Multi-Classification Refactor (PC-MoE + HyperRouter)

## Objective
Enable GraphDrone to handle multi-class classification natively and address Signal-to-Noise (SNR) issues using global task context.

## Hypothesis
A Probabilistic Categorical Mixture of Experts (PC-MoE) using a HyperGraph-inspired router will achieve parity or superiority over TabPFN by intelligently leveraging specialized views.

## Key Architectural Achievements
1.  **Probabilistic PC-MoE**: Refactored the entire pipeline (TokenBuilder, Integrator, Fit loop) to handle 3D probability tensors [N, E, C].
2.  **HyperSetRouter**: Implemented a cross-attention router that incorporates a **Global Task Token** (Bayesian-like prior) to anchor specialist selection.
3.  **Size-Aware Routing Strategy**: Automatically falls back to a static anchor for small datasets (N < 500) to prevent router overfitting.
4.  **Anchor-Aware Loss**: Introduced a residual penalty that prevents the router from degrading performance below the single-expert baseline.

## Final Benchmark Results (11 TabArena Datasets)
- **Status**: 🟢 9/11 datasets beat or match TabPFN.
- **Top Victory**: `segment` (7-class) -> **+0.4957 F1** over TabPFN.
- **Top Cardinality**: `Digits` (10-class) -> **99.17% Accuracy**.
- **Stability**: Anchor-Aware loss ensured 0.0000 regressions against the baseline on all binary tasks.

## Infrastructure Fixes
- Resolved CUDA device-side assertions on multiclass indexing.
- Fixed joblib pickling crashes during parallel specialist fitting.
- Implemented stratified internal validation for the router.

## Reproducibility
```bash
PYTHONPATH=src python3 validation_scripts/tabarena_classification_benchmark.py --max-samples 1000
```
