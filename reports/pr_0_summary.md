# Experiment Summary: Multi-Classification Refactor

## Objective
Enable GraphDrone to handle multi-class classification natively using a probabilistic approach.

## Hypothesis
Replacing the current regression-based logic with a Probabilistic Categorical Mixture of Experts (PC-MoE) will allow GraphDrone to achieve high accuracy on classification tasks.

## Scope of Changes
- **config.py**: Added `problem_type` and `n_classes` to `GraphDroneConfig`.
- **portfolio_loader.py**: Updated `LoadedExpert.predict` to return class probabilities (N, C).
- **expert_factory.py**: Refactored `predict_all` to support 3D prediction tensors (N, E, C).
- **token_builder.py**: Updated `UniversalTokenBuilder` to handle 3D class probability tensors and compute residuals across class distributions.
- **defer_integrator.py**: Refactored `integrate_predictions` to perform class-wise weighted averaging.
- **model.py**: Overhauled `fit()` to detect problem types, use `CrossEntropyLoss` for the router, and optimize class distributions.

## Results
- **Dataset**: Iris (3 classes)
- **Accuracy**: 0.9667
- **Router Device**: CUDA

## Reproducibility Command
```bash
PYTHONPATH=src python3 validation_scripts/test_classification.py
```

## Artifact Locations
- reports/pr_0_summary.md
