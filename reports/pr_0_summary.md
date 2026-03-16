# Experiment Summary: Multi-Classification Refactor (v2)

## Objective
Enable GraphDrone to handle multi-class classification natively using a probabilistic approach and validate against TabPFN on extended datasets.

## Hypothesis
Replacing the current regression-based logic with a Probabilistic Categorical Mixture of Experts (PC-MoE) will allow GraphDrone to achieve parity or superiority over TabPFN on complex multi-class tasks.

## Scope of Changes
- **config.py**: Added `problem_type` and `n_classes` to `GraphDroneConfig`.
- **portfolio_loader.py**: Updated `LoadedExpert.predict` to return class probabilities (N, C).
- **expert_factory.py**: Refactored `predict_all` to support 3D prediction tensors (N, E, C).
- **token_builder.py**: Updated `UniversalTokenBuilder` to handle 3D class probability tensors.
- **defer_integrator.py**: Refactored `integrate_predictions` to perform class-wise weighted averaging.
- **model.py**: Overhauled `fit()` for classification tasks using `NLLLoss` on log-probabilities.

## Extended Results (H200 Optimized)
| Dataset | GraphDrone Acc | TabPFN Acc | GraphDrone ROC-AUC | TabPFN ROC-AUC |
|---------|----------------|------------|-------------------|----------------|
| Wine    | 1.0000         | 1.0000     | 1.0000            | 1.0000         |
| Breast  | 0.9649         | 0.9649     | 0.9934            | 0.9951         |
| Digits  | 0.9833         | 0.9800     | 0.9997            | 0.9996         |
| Segment | 0.9850         | 0.9850     | 0.9994            | 0.9994         |

## Conclusion
GraphDrone (PC-MoE) achieves parity with TabPFN on most tasks and shows a slight edge in high-cardinality multi-class (Digits). The H200 environment provides significant acceleration for the Router optimization phase.

## Reproducibility Command
```bash
PYTHONPATH=src python3 validation_scripts/extended_benchmark.py
```
