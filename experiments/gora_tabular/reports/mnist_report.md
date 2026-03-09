# GoRA-Tabular: mnist
*2026-03-07* | Branch: `feature/gora-tabular-routing`

## Architecture
- **Routing semantics:** π_{i,h,m} from MoERouter(g_i) shapes attention logits:
  `logit_{ij}^h = <q^h,k^h>/√d + log(τ_h · Ã_{ij}^{i,h} + ε)`
- **Routing is inside the softmax**, not a downstream combiner.
- **Isolation vs interaction** is structural: a peaked π confines the softmax to
  one view's neighbourhood; flat π spans all views.
- **Routing scope (this pass):** per-row, per-head (H=4 heads).

## Models
| ID | Description |
|----|-------------|
| B0 | MLP |
| B1 | HGBR |
| G0 | Standard Transformer (no graph) |
| G1 | Single-view Transformer (fixed best view) |
| **G2** | **GoRA-Tabular (full routing)** |
| G3 | Uniform-π ablation (no geometry) |
| G4 | Shuffled-g ablation (geometry destroyed) |

## Metrics

| model         |   accuracy |   macro_f1 |   log_loss |
|:--------------|-----------:|-----------:|-----------:|
| B0_MLP        |   0.938667 |   0.938303 |   0.213936 |
| B1_HGBR       |   0.958    |   0.957679 |   0.188216 |
| G0_Standard   |   0.936    |   0.935375 |   0.247903 |
| G1_SingleView |   0.94     |   0.939214 |   0.250575 |
| G2_GoRA       |   0.936    |   0.935225 |   0.258716 |
| G3_Uniform    |   0.932667 |   0.931703 |   0.244748 |
| G4_Random     |   0.932    |   0.931345 |   0.27456  |

## Head-View Affinity (GoRA Table 5 equivalent)

|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |   1.09003 | PCA             |       0.295269 |         0        |        0.309394 |          0        |      0.395337 |       1         |
|          1 |   1.08668 | BLOCK           |       0.264991 |         0        |        0.386664 |          0.947333 |      0.348344 |       0.0526667 |
|          2 |   1.0978  | BLOCK           |       0.340973 |         0.512667 |        0.344536 |          0.430667 |      0.314491 |       0.0566667 |
|          3 |   1.09439 | PCA             |       0.290602 |         0        |        0.352473 |          0.348667 |      0.356925 |       0.651333  |

## Per-head temperature τ (learned): ['1.004', '1.003', '1.003', '0.999']

## Audit Conclusion
- G2 routing beats G3 uniform? **YES** (G2=0.9360, G3=0.9327 accuracy)
- G2 beats strong tabular baseline (B1)? **NO** (G2=0.9360, B1=0.9580)

## What makes this different from prior experiments
- Prior: frozen reps → ObserverRouter → reweighting (post-hoc, Model A pattern)
- This: g_i → π_{i,h,m} → logit bias inside softmax → representation formation
- The graph neighbourhood each head sees is determined BEFORE embedding is complete.