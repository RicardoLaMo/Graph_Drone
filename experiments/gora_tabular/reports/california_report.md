# GoRA-Tabular: california
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

| model         |     rmse |      mae |       r2 |
|:--------------|---------:|---------:|---------:|
| B0_MLP        | 0.545516 | 0.388832 | 0.774949 |
| B1_HGBR       | 0.443292 | 0.300658 | 0.85139  |
| G0_Standard   | 0.440333 | 0.301465 | 0.853368 |
| G1_SingleView | 0.495559 | 0.356961 | 0.814281 |
| G2_GoRA       | 0.484665 | 0.34485  | 0.822356 |
| G3_Uniform    | 0.507591 | 0.362982 | 0.805152 |
| G4_Random     | 0.450068 | 0.310251 | 0.846813 |

## Head-View Affinity (GoRA Table 5 equivalent)

|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |   1.34323 | GEO             |       0.168579 |        0         |      0.369096 |        0.949935 |        0.210154 |       0.00452196  |          0.252171 |           0.0455426 |
|          1 |   1.34064 | GEO             |       0.180352 |        0         |      0.375864 |        0.825904 |        0.191363 |       0.000968992 |          0.252422 |           0.173127  |
|          2 |   1.34194 | GEO             |       0.257233 |        0.130491  |      0.365805 |        0.837855 |        0.160209 |       0.00129199  |          0.216754 |           0.0303618 |
|          3 |   1.31167 | GEO             |       0.190679 |        0.0394057 |      0.425918 |        0.960594 |        0.176857 |       0           |          0.206547 |           0         |

## Per-head temperature τ (learned): ['0.985', '1.008', '1.015', '1.010']

## Audit Conclusion
- G2 routing beats G3 uniform? **YES** (G2=0.4847, G3=0.5076 rmse)
- G2 beats strong tabular baseline (B1)? **NO** (G2=0.4847, B1=0.4433)

## What makes this different from prior experiments
- Prior: frozen reps → ObserverRouter → reweighting (post-hoc, Model A pattern)
- This: g_i → π_{i,h,m} → logit bias inside softmax → representation formation
- The graph neighbourhood each head sees is determined BEFORE embedding is complete.