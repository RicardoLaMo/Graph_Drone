# Theory-to-Code Mapping: GraphDrone Challenger

This document maps the implementation of the "Challenger" model to the statistical and information-theoretic principles that justify its superiority over the "Champion" (baseline).

## 1. Distributional Geometry (Beyond Variance)
**Principle:** Standard variance ($\sigma^2$) assumes a symmetric, thin-tailed Gaussian error distribution. Tabular data often exhibits heteroscedasticity, skewness, and fat tails (kurtosis).

| Code Element | Logic | Statistical Justification |
| :--- | :--- | :--- |
| `compute_real_support_encoding` | Calculates $E[(x-\mu)^3]$ and $E[(x-\mu)^4]$ | **Moment-Generating Functions**: Providing higher moments allows the router to approximate the local PDF shape, identifying systematic bias (skew) or outlier sensitivity (kurtosis). |
| `MomentSupportEncoder` | Collapses kNN labels into moment-vectors | **Sufficient Statistics**: For many distributions, the first four moments are nearly sufficient for characterizing the local regime. |

## 2. Permutation Invariance (Set-Based Routing)
**Principle:** In a multi-view system, the "best" expert is relative. The router should treat experts as a set of candidates rather than a fixed vector of features.

| Code Element | Logic | Architectural Justification |
| :--- | :--- | :--- |
| `EnhancedTokenBatch` | Packages `[pred, quality, support, descriptor]` | **Tokenization**: Decouples the expert's *identity* from its *performance*, allowing a shared routing logic to generalize across variable expert sets. |
| `LearnedSetRouter` | Shares MLP weights across all tokens | **Permutation Invariance**: Ensures that swapping the order of `V1` and `V2` does not change the routing decision. Grounded in *Deep Sets* (Zaheer et al., 2017). |

## 3. Contextual Integration (Anchor-Based Deferral)
**Principle:** The `FULL` model is the most robust generalist. Specialists should only be used when they provide a clear **Information Gain** over the `FULL` anchor.

| Code Element | Logic | Information Theory Justification |
| :--- | :--- | :--- |
| `prediction_minus_full` | Tokenizes the residual $\hat y_{expert} - \hat y_{FULL}$ | **Residual Learning**: The router focuses on the *delta* an expert provides, reducing the search space to "Is this specialist correcting a known error in the generalist?" |
| `defer_prob` (Gate) | Sigmoid head on the `FULL` token | **Trust-Region Routing**: Explicitly models the "Trust" in the generalist versus the specialist pool, preventing competition noise from degrading a stable `FULL` prediction. |

## 4. Mechanism Verification (The "Audit")
To confirm the implementation is "Correct" beyond just Accuracy, we verify:
1. **Sparsity**: Does the router successfully suppress noisy views (Entropy check)?
2. **Alignment**: Is weight $W_v$ strongly correlated with low local variance $\sigma^2_v$?
3. **Stability**: Does the router maintain consistent weights across minor feature perturbations?
