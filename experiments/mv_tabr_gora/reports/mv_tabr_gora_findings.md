# MV-TabR-GoRA Findings Report

**Date:** 2026-03-08
**Branch:** `feature/mv-tabr-gora`
**Run:** Full CA dataset, K=24 per view (96 total ≈ TabR context_size), 150 epochs max

---

## Results Table

| Model | Test RMSE | Δ vs A0 | Val RMSE | Best Epoch | Description |
|---|---|---|---|---|---|
| A0 | 0.5531 | 0.0000 | 0.5691 | 145 | Global FULL-view, uniform attention, label in KV |
| **A1** | **0.4289** | **+0.1242** | 0.4611 | 81 | Per-view kNN, uniform routing, label in KV |
| **A2** | **0.4221** | **+0.1310** | 0.4587 | 71 | Per-view kNN, sigma2_v routing + J-temperature |
| **A3** | **0.4075** | **+0.1456** | 0.4470 | 146 | A2 + T(z_i^v − z_j^v) direction encoding |
| A4 | 0.4276 | +0.1255 | 0.4777 | 41 | A3 + Q(q_i^v, q_j^v) quality-pair encoding |
| A5 | 0.4306 | +0.1225 | 0.4796 | 41 | A4 + learned routing MLP |
| A6 | 0.4282 | +0.1249 | 0.4800 | 12 | A5 + CrossViewMixer + β-gate |

### Reference Baselines

| Model | RMSE |
|---|---|
| TabR (champion) | **0.3829** |
| TabM | 0.4290 |
| L2_stack_hgbr | 0.4292 |
| HGBR | 0.4430 |
| GoRA v2 G2 | 0.4546 |
| GoRA v5 A0 | 0.4738 |

---

## Key Findings

### Finding 1: Per-view retrieval + label-in-KV already matches TabM (A1 ✅)

`A1` (per-view kNN, uniform routing, label in KV) achieves **test RMSE 0.4289** — essentially
tied with TabM (0.4290). This is with:
- 4 separate lightweight view encoders (Linear + LayerNorm, ~22K total params)
- Per-view kNN retrieval (K=24 per view = 96 total neighbours)
- Labels embedded in the attention value: `LabelEmbed(y_j)`
- Uniform routing across 4 views

**Implication:** The core mechanism — per-view support banks with label-in-KV attention —
is the primary driver. TabM's ensemble of 32 heads over shared features is being matched
by 4-view retrieval over view-specific feature spaces. The architectural direction is correct.

### Finding 2: sigma2_v routing adds +0.007 RMSE (A2 ✅)

`A2` (sigma2_v routing + J-temperature) beats `A1` by **+0.0068 RMSE** (0.4289→0.4221).
The non-parametric soft routing `pi = softmax(-sigma2_v / tau)` with `tau = 1/(mean_J + ε)`
adds measurable improvement, consistent with the validation result (+0.040 at retrieval oracle level).

The smaller delta here than in the validation (0.040 oracle vs 0.007 neural) is expected:
the neural model's attention already learns a form of proximity-weighted reading, so the
explicit routing adds incremental rather than additive benefit. It still helps.

### Finding 3: Direction encoding is the strongest single contribution (A3 ✅✅)

`A3` adds `T(z_i^v − z_j^v)` (anchor-relative direction encoding in the value function):
- Best model overall: **test RMSE 0.4075**
- +0.0146 vs A2 (largest single step improvement)
- Beats TabM by **0.0215 RMSE**
- Beats HGBR by **0.0355 RMSE**
- Best epoch 146 — still converging slowly at end of training window

The direction encoding captures WHERE the anchor sits relative to each neighbor in the
view's embedding space, not just who the neighbors are. This is the `T(z_i^v − z_j^v)`
value component. Validation evidence: "distance weighting adds +0.029 RMSE at oracle level."

**A3 is the current champion at 0.4075 RMSE.**

### Finding 4: Quality-pair encoding HURTS (A4 ❌)

`A4` adds `Q(q_i^v, q_j^v)` (pair of sigma2_v quality scalars mapped to d_model) but
**regresses to 0.4276** — +0.020 worse than A3. Several likely causes:

1. **Redundancy**: sigma2_v already drives the routing weights in A2/A3. Encoding it
   again in the value adds noise without orthogonal information.
2. **Signal contamination**: Q([q_i, q_j, q_i*q_j, |q_i-q_j|]) introduces all 4
   scalar interactions, but sigma2_v is a single noisy scalar at test time; the
   additional features may amplify noise rather than signal.
3. **Early stopping**: A4 stops at epoch 41 vs A3 epoch 146 — the model is overfitting
   faster with more value components, suggesting regularization is insufficient.

**Implication:** Use sigma2_v for ROUTING only, not for value augmentation.
Quality priors should inform WHICH view to read from, not WHAT to read from it.

### Finding 5: Learned routing and CrossViewMixer don't help on top of broken A4 (A5, A6)

A5 and A6 inherit A4's Q encoding regression. The learned routing MLP (A5) doesn't
recover the loss, and the CrossViewMixer (A6) early-stops at epoch 12 — optimization
is struggling. Both reflect that A4's regression propagates forward.

**Implication:** A5/A6 need to be built on top of a *fixed* A3 (without Q encoding).
These are the right mechanisms but they haven't been tested cleanly.

---

## Revised Ablation Plan (Next Experiments)

Based on findings, the revised A4/A5/A6 should remove Q from the value:

| Model | Description | Hypothesis |
|---|---|---|
| **A3** (current champion) | A2 + direction encoding T(z_i^v − z_j^v) | ✅ 0.4075 |
| **A4_fix** | A3 + learned routing MLP (no Q) | Does learned routing beat sigma2_v-only? |
| **A5_fix** | A3 + CrossViewMixer + β-gate (no Q) | Does cross-view interaction add value? |
| **A6_fix** | A4_fix + CrossViewMixer + β-gate | Full stack without Q |

Additionally, for the quality signal, consider alternatives to Q-in-value:
- Use curvature observers (kappa, LID) as quality priors instead of sigma2_v
- Use Q as a soft MASK on attention logits (not additive to value)
- Pre-train separate quality heads before fine-tuning the student

---

## Gap to TabR Analysis

Current gap: A3 (0.4075) vs TabR (0.3829) = **0.0246 RMSE**

TabR advantages not yet in A3:
1. **Deeper predictor**: TabR has `n_blocks=1` in the predictor with 303-dim features; A3's task head is 2-layer MLP. TabR's learned query/key projections may be richer.
2. **Larger d_main**: TabR uses d_main=303 vs our d_model=64. The embedding capacity is 5× larger.
3. **Gradient through retrieval**: TabR trains end-to-end with labels visible during retrieval training. Our approach fixes kNN (raw feature space) and only trains the attention/routing.
4. **Global kNN**: TabR uses FAISS global kNN (not per-view); it implicitly learns the right geometry through training.

The 0.025 gap likely comes from #2 (embedding capacity) and #3 (gradient through retrieval).
Increasing d_model from 64 to 256 and K from 24 to 48 are the most promising next steps.

---

## Architecture Implications

| Finding | Implication |
|---|---|
| F1: A1 ≈ TabM | Per-view retrieval + label-in-KV is the critical mechanism |
| F2: A2 > A1 | sigma2_v routing adds value; keep in all subsequent models |
| F3: A3 is best | T(z_i^v − z_j^v) is the strongest novel contribution |
| F4: A4 regresses | Q(q_i, q_j) = sigma2_v in value hurts; routing only, not value |
| F4: Early stop at ep 41 | Q encoding causes overfitting; tighter regularization or remove |
| F5: A5/A6 inherit A4 | Fix A4 first; then test learned routing and CrossViewMixer cleanly |

---

## Go Decision

**A3 is the confirmed new baseline at 0.4075 RMSE**, surpassing:
- TabM by +0.022
- HGBR by +0.036
- All GoRA variants

**Next step:** Fix A4/A5/A6 by removing Q encoding from value. Test learned routing and
CrossViewMixer cleanly on top of A3. Then scale (d_model=256, K=48) to challenge TabR.
