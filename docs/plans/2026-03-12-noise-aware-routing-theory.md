# Theory: Noise-Aware Contextual Routing (SNR-Gating)

## The Problem: Noise Dominance in Subspaces
In a multi-view GraphDrone system, some expert views are "noise-dominant" for specific query points. If a view's local feature subspace is uninformative, its prediction is essentially a random walk. A standard softmax router may still assign non-zero weight to these experts, diluting the "signal" from more informative views.

## Theoretical Foundation

### 1. The Information Bottleneck (Tishby et al.)
We treat each expert as a channel. The goal is to maximize the **Mutual Information** between the expert's prediction and the true label, while minimizing the **Residual Noise**.
- **Signal ($S$):** Agreement between the expert's local neighborhood centroid and its current prediction.
- **Noise ($N$):** Local label variance ($\sigma^2$) + higher-order Kurtosis (outlier sensitivity).

### 2. Signal-to-Noise Ratio (SNR) Tokens
We introduce an explicit **SNR Token** calculated as:
$$SNR_{v} = \frac{| \hat{y}_v - \bar{y}_{neighbor} |}{\sigma_v + \epsilon}$$
Where $\bar{y}_{neighbor}$ is the mean label of the k-nearest neighbors in that view's subspace.

### 3. Noise-Gate Pruning (Noisy MoE - Shazeer et al.)
Instead of a "Dense" softmax, we implement a **Noise Gate**. If an expert's $SNR_v$ falls below a learned threshold $\tau_{noise}$, its weight is set to zero *before* the Cross-Attention block. This prevents "Garbage In, Garbage Out" at the attention layer.

## Implementation Plan: "The Noise-Aware Challenger"
1.  **SNR-Token Builder**: Compute the ratio of prediction agreement to neighborhood variance.
2.  **SNR-Gate Module**: A sigmoid-based mask that suppresses experts with low local information gain.
3.  **Contextual Pruning**: Integrate this gate into the `CrossAttentionSetRouter`.

## Research Support
- **Sparse MoE (Google Brain, 2017):** Proved that "Noisy Top-K Gating" improves stability by explicitly modeling the variance of the gating signal.
- **Attention Is All You Need (Section 3.2):** Scaled Dot-Product Attention can be sensitive to magnitude; a noise gate acts as a "hard" normalization against low-signal experts.
