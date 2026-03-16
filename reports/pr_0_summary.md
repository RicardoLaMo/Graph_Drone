# Experiment Summary: HyperGraph Clique Routing (v3)

## Objective
Address Signal-to-Noise (SNR) issues in high-dimensional classification by incorporating global task context and HyperGraph-inspired routing.

## Hypothesis
A "Task Token" (global Bayesian-like prior) will allow the router to distinguish between genuine specialist expertise and random noise alignment, increasing specialist utilization (Defer Prob) on complex tasks.

## Key Changes
- **support_encoder.py**: Implemented Global **Task Token** generation (Mean, Std, Sparsity, Dim).
- **token_builder.py**: Integrated Task Token into the `TokenBatch` and added prediction entropy features.
- **set_router.py**: Introduced **HyperSetRouter** which performs cross-attention between a (Anchor + Task) query and individual experts.
- **model.py**: Updated fit/predict cycles to propagate global task statistics.

## Results (H200 Optimized)
| Dataset | Acc (HyperRouter) | Mean Defer Prob | Status |
|---------|-------------------|-----------------|--------|
| Digits  | 0.9917            | 0.6468          | 🟢 Significant Improvement |
| Segment | 0.9740            | 0.4385          | 🟢 Stable Utilization |

## Conclusion
The HyperGraph approach successfully resolved the "Router Silence" issue where defer probability was near zero. By anchoring the router's attention with global task context, we achieved a more robust and specialized ensemble.

## Reproducibility Command
```bash
PYTHONPATH=src python3 validation_scripts/hyper_benchmark.py
```
