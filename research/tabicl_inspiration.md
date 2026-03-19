# TabICL Inspiration Research
# Branch: research/tabicl-inspiration
# Date: 2026-03-19
# Parent: feat/clf-multiclass-win

## Context

Benchmark ranking on maternal_health_risk (log loss, lower is better):
TabICL 0.3686 | TabPFN 0.4164 | GD 0.4385 | CatBoost 0.4624 | TabM 0.5559 | XGBoost 0.5768

TabICL beats TabPFN by ~11% on log loss. GD is currently 5.4% behind TabPFN on this metric.
Understanding why TabICL wins is the key to closing both gaps.

Source: TabICL is installed locally at:
  `/home/wliu23/miniconda3/lib/python3.13/site-packages/tabicl/`
ArXiv: https://arxiv.org/abs/2502.05564

---

## Why TabICL Beats TabPFN: Root Causes (Source-Verified)

### 1. Logit Averaging Before Softmax — Single Biggest Calibration Win

TabICL averages **raw logits** across 32 ensemble members, then applies softmax **once**:

```python
# tabicl/sklearn/classifier.py
if self.average_logits:
    avg = self.softmax(avg, axis=-1, temperature=self.softmax_temperature)
```

TabPFN (and current GD) averages **probabilities** (post-softmax). This is the core calibration difference:
- Logit averaging = geometric mean of distributions (log space) → conservative near decision boundaries
- Probability averaging = convex combination → tends to over-concentrate near peaks
- With 32 members where some members are confident class A and others confident class B, logit averaging
  yields a flatter/more uncertain output; probability averaging yields spurious confidence.

**Why this matters for maternal_health_risk**: "mid risk" and "high risk" classes are ambiguous on the margin.
Logit averaging correctly expresses uncertainty; probability averaging inflates the leading class.

### 2. Temperature Scaling — Free Calibration Gain

Default temperature: **0.9** (sharpening, < 1.0):
```python
def softmax(x, axis=-1, temperature=0.9):
    x = x / temperature
```
Applied post-logit-averaging. This was empirically tuned. Note: counter-intuitively, slightly sharpening
the final distribution after logit averaging (which already flattened it) improves calibration.

### 3. Three-Axis Ensemble Diversity

TabICL creates 32 members from three orthogonal axes:
1. **Feature permutation** — Latin square patterns (systematic, non-overlapping column orderings)
2. **Class label cyclic shift** — Each member shifts class labels by offset mod n_classes, then reverses at prediction
3. **Normalization method** — Split between `none` (z-score) and `power` (Yeo-Johnson transform)

Current GD: single axis (random feature subspace, same normalization, no class shift).

### 4. Tree-Based Priors (30% XGBoost SCMs)

TabICL's synthetic training data uses 70% MLP SCMs + 30% XGBoost SCMs.
XGBoost SCMs generate threshold-based class boundaries that exactly match maternal_health_risk structure
(blood pressure / glucose thresholds → risk level).
TabPFN used only MLP SCMs. GD inherits TabPFN's prior — same limitation.

**This explains the maternal_health_risk gap specifically**: threshold-structured datasets favour TabICL's prior.

### 5. Three-Stage Decoupled Architecture (Late Label Fusion)

TabICL:
- Stage 1 (TFcol): Set Transformer processes each column independently — distribution-aware embeddings
- Stage 2 (TFrow): Row interactions via RoPE transformer — NO labels yet
- Stage 3 (TFicl): Labels injected via OneHotAndLinear — ICL inference

TabPFN: Labels fused early, jointly with features. This "entangles" the representation and reduces
calibration by baking label-feature correlations into the base embedding.

### 6. More Expressive Prior Activations

TabPFN: 4 activations (Tanh, LeakyReLU, ELU, Identity).
TabICL: 18+ activations + RandomFunctionActivation (sign, heaviside, RBF exp(-x²), random sine/Fourier) —
10x oversampled. Better prior coverage → better calibration on heterogeneous real-world features.

### 7. Two-Stage Outlier Handling

```python
# preprocessing.py — 4-sigma detection + log-based clipping
x = maximum(-log1p(|x|) + lower_bound, x)
```
Preserves near-boundary values (important for medical features like blood pressure).
GD passes data through to TabPFN with only median imputation and ordinal encoding.

---

## Actionable Ideas for GraphDrone

Ranked by expected impact and implementation cost:

### HIGH IMPACT, LOW COST

#### A. Switch GeoPOE from probability-space to logit-space blending

**Current** (`geo_ensemble.py`):
```python
log_p = np.log(np.clip(predictions, 1e-9, 1.0))  # [N, E, C]
log_blend = np.einsum("ne,nec->nc", weights, log_p)  # weighted avg of log-probs
blend = np.exp(log_blend) / sum(...)  # re-normalizes
```
This IS equivalent to logit averaging IF the weights were the same for all samples. But the entropy-based
weights vary per sample — so the current approach is already geometric-mean-like. However, the blend
computes weighted avg of log probs then exponentiates, which is different from averaging raw logits then
single softmax. The difference: TabICL averages pre-softmax logits (before any normalization), which makes
the result more sensitive to the raw scale of the logit differences.

**Investigation needed**: Request TabPFN's raw logits (pre-softmax) and average those instead of averaging
probabilities. TabPFN exposes `predict_proba()` — need to check if raw logits are accessible.

#### B. Add cyclic class label shifting as an extra expert

**Cost**: ~5 lines. Per multiclass dataset with K classes, create K-1 shifted variants:
```python
# Before fitting expert i, shift labels: y_shifted = (y + i) % n_classes
# At prediction: shift output back: pred_shifted = np.roll(pred, -i, axis=-1)
```
Each expert sees a different label-space permutation → orthogonal diversity to feature subspace diversity.
This is especially powerful for classes with ordinal relationships (like risk levels: low/mid/high).

#### C. Add Yeo-Johnson normalization as a second portfolio view

**Cost**: ~10 lines. Create a second FULL expert trained on power-transformed features:
```python
from sklearn.preprocessing import PowerTransformer
X_yj = PowerTransformer(method='yeo-johnson').fit_transform(X_train)
# Add ExpertBuildSpec with a YeoJohnsonAdapter as the input_adapter
```
Two FULL experts (z-score + Yeo-Johnson) provide normalization diversity without any subspace degradation.
High value for skewed features (blood pressure, glucose, income, house prices).

#### D. Apply temperature T=0.9 to the final GeoPOE blend

**Cost**: 1 line. In `anchor_geo_poe_blend` and `learned_geo_poe_blend`, divide final logits by 0.9
before softmax. Empirically validated by TabICL. Can be tuned.

### MEDIUM IMPACT, MEDIUM COST

#### E. Latin square feature permutations instead of random subsets

Replace random 80-90% subspace sampling with systematic Latin square-derived subsets:
- Latin square ensures coverage: for 4 experts (FULL + 3 SUBs), each feature appears in exactly 3 of 4 experts
- Maximizes diversity while minimising overlap redundancy
- Implementation: `scipy.linalg.hadamard` or explicit construction for small expert counts

#### F. Two-stage outlier clipping as a preprocessing step

Apply before passing to TabPFN: detect 4-sigma outliers, set to NaN, recompute stats, log-clip:
```python
def robust_clip(X, sigma=4.0):
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    outlier_mask = np.abs(X - mu) > sigma * sd
    X_clean = np.where(outlier_mask, np.nan, X)
    mu2, sd2 = np.nanmean(X_clean, axis=0), np.nanstd(X_clean, axis=0)
    X_norm = (X - mu2) / (sd2 + 1e-9)
    return np.sign(X_norm) * np.log1p(np.abs(X_norm)) / sigma  # log-compress extremes
```
This doesn't require model changes — just preprocessing in the adapter.

### HIGH COST / ARCHITECTURAL (FUTURE WORK)

#### G. Late label fusion architecture

Retrain a foundation model with decoupled feature and label processing.
Requires a full pretraining run — not feasible short-term.

#### H. Set Transformer column embeddings (distribution-aware)

Replace per-feature linear projection with a shared Set Transformer.
Requires modifying TabPFN's architecture or building a new backbone.

---

## Experiment Priority Order

1. **B + D** (class shift + temperature): zero-risk, 1-2 hours to implement and benchmark
2. **C** (Yeo-Johnson view): low-risk, tests normalization diversity hypothesis
3. **A** (logit-space blend): requires checking TabPFN logit access API
4. **E** (Latin square permutations): replaces current random subspace sampling
5. **F** (outlier clipping): pure preprocessing, apply always
6. **G + H**: future research track

---

## maternal_health_risk Specific Notes

Dataset: OpenML 43982 (or similar), 3 classes (low/mid/high risk), features include:
- Age, SystolicBP, DiastolicBP, BS (blood sugar), BodyTemp, HeartRate

Why GD lags TabPFN here specifically:
- Blood pressure features are skewed → TabPFN's z-score normalization distorts them
- Risk thresholds are piecewise-constant → TabPFN's MLP-only prior doesn't capture this well
- Both issues hit GD equally since GD uses TabPFN as its base expert
- TabICL's tree-based SCMs + power transform normalization directly address both

Quick win: Add maternal_health_risk to the benchmark suite and test Yeo-Johnson view (idea C).

---

## References

- TabICL paper: https://arxiv.org/abs/2502.05564
- Local install: /home/wliu23/miniconda3/lib/python3.13/site-packages/tabicl/
- Key source files: sklearn/classifier.py, sklearn/preprocessing.py, prior/prior_config.py
