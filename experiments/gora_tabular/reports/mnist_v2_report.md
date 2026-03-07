# GoRA-Tabular v2: mnist_v2
*2026-03-07* | Branch: `feature/gora-joint-knn`

## Architecture upgrade: joint-view kNN
Each view m independently nominates K_each=5 neighbours.
Union pool P = ∪_m kNN_m(i). Edge weight w^(m)_{ij}=0 if j∉kNN_m(i).
Routing entropy aligned with view agreement (G6 only).

## Metrics

| model             |   accuracy |   macro_f1 |   log_loss |
|:------------------|-----------:|-----------:|-----------:|
| B0_MLP            |   0.938667 |   0.938303 |   0.213936 |
| B1_HGBR           |   0.958    |   0.957679 |   0.188216 |
| B2_TabPFN         |   0.912    |   0.910944 |   0.274598 |
| G2_GoRA_v1        |   0.941333 |   0.940712 |   0.233056 |
| G5_Joint          |   0.938    |   0.937631 |   0.236303 |
| G6_Joint_Reg      |   0.930667 |   0.930212 |   0.250104 |
| G3p_Uniform_Joint |   0.936667 |   0.936132 |   0.246039 |

## Ablation results
- G5 (joint) vs G2 (single-primary): **❌ no gain** G5=0.9380 G2=0.9413 accuracy
- G6 (+ routing loss) vs G5: **❌ no gain** G6=0.9307 G5=0.9380
- G5 vs G3' (uniform+joint): G5=0.9380 G3'=0.9367 — **routing > uniform**
- G5 vs B1 (HGBR): **❌ HGBR still leads** G5=0.9380 B1=0.9580
- G5 vs B2 (TabPFN): **✅ GoRA wins** G5=0.9380 B2=0.9120

## View agreement score
Mean agree_score = 0.372 (0=all views disagree, 1=all views nominate same neighbours)

## Head-View Affinity by model

### G2_GoRA_v1
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |   1.09737 | BLOCK           |       0.315589 |      0.0173333   |        0.355673 |          0.952667 |      0.328738 |           0.03  |
|          1 |   1.09713 | BLOCK           |       0.331096 |      0.000666667 |        0.356648 |          0.999333 |      0.312256 |           0     |
|          2 |   1.09417 | PCA             |       0.291319 |      0           |        0.342756 |          0.056    |      0.365925 |           0.944 |
|          3 |   1.09342 | PCA             |       0.298939 |      0           |        0.321032 |          0.012    |      0.380029 |           0.988 |

### G5_Joint
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |  1.07301  | PCA             |       0.280965 |                0 |        0.276708 |        0.00866667 |      0.442327 |        0.991333 |
|          1 |  1.05542  | PCA             |       0.234903 |                0 |        0.295386 |        0          |      0.469711 |        1        |
|          2 |  1.06876  | PCA             |       0.331353 |                0 |        0.235188 |        0          |      0.433459 |        1        |
|          3 |  0.997757 | PCA             |       0.257078 |                0 |        0.195125 |        0          |      0.547796 |        1        |

### G6_Joint_Reg
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |   1.073   | PCA             |       0.361832 |        0.0493333 |        0.232467 |                 0 |      0.405701 |        0.950667 |
|          1 |   1.06212 | PCA             |       0.290538 |        0         |        0.248547 |                 0 |      0.460915 |        1        |
|          2 |   1.07944 | PCA             |       0.307073 |        0         |        0.268701 |                 0 |      0.424226 |        1        |
|          3 |   1.06036 | PCA             |       0.350062 |        0         |        0.215476 |                 0 |      0.434463 |        1        |


## Per-head τ (G5): ['1.001', '0.984', '1.001', '0.999']