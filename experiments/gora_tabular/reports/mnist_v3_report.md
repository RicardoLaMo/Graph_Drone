# GoRA-Tabular v3 (MQ-GoRA): mnist_v3
*2026-03-07* | Branch: `claude/funny-davinci`

## Architecture: Manifold-Query GoRA
Teacher f_T: x_i → z_i encodes manifold geometry (agree_score, label centroid, neighbour centroid).
Student stages:
  A. ViewSpecificEmbedder: M separate projections → view-discriminative values
  B. LabelContextEncoder: per-view label stats → router input + value augmentation
  C. ManifoldReader: z_i-queried cross-attention → ctx_vec (or avg-pool for G7/G8)
  D. RichMoERouter: [g_anc; z_anc; label_ctx; ctx_vec] → pi, tau
  E. SparseGeomLayers (unchanged)
  F. AlphaGate: pred_final = (1-α)·pred_base + α·pred_local (G10 only)

## Metrics

| model       |   accuracy |   macro_f1 |   log_loss |
|:------------|-----------:|-----------:|-----------:|
| B1_HGBR     |   0.958    |   0.957679 |   0.188216 |
| B2_TabPFN   | nan        | nan        | nan        |
| G2_GoRA_v1  |   0.93     |   0.929386 |   0.258996 |
| G7_RichCtx  |   0.926667 |   0.925792 |   0.275473 |
| G8_LabelCtx |   0.930667 |   0.930003 |   0.272681 |
| G9_Teacher  |   0.934667 |   0.93402  |   0.279861 |
| G10_Full    |   0.938    |   0.937367 |   0.237163 |

## Ablation ladder (G7 → G10)

| Gate | Δ component | Result |
|------|-------------|--------|
| G7 vs G2 | +ViewSpecificEmbed +ctx^(m) in router | **❌** G7=0.9267 G2=0.9300 |
| G8 vs G7 | +label_ctx (router + value aug) | **✅** G8=0.9307 G7=0.9267 |
| G9 vs G8 | +teacher z_anc as cross-attn query | **✅** G9=0.9347 G8=0.9307 |
| G10 vs G9 | +alpha-gate prediction fusion | **✅** G10=0.9380 G9=0.9347 |
| G10 vs B1 | GoRA v3 vs HGBR | **❌ HGBR still leads** G10=0.9380 B1=0.9580 |
| G10 vs B2 | GoRA v3 vs TabPFN | **❌ TabPFN leads** G10=0.9380 B2=N/A |

## View agreement score
Mean agree_score = 0.372

## Head-View Affinity by model

### G2_GoRA_v1
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |   1.07842 | PCA             |       0.264642 |                0 |        0.309815 |        0.00866667 |      0.425543 |        0.991333 |
|          1 |   1.09172 | BLOCK           |       0.290902 |                0 |        0.385714 |        1          |      0.323383 |        0        |
|          2 |   1.08949 | PCA             |       0.272441 |                0 |        0.35052  |        0          |      0.377039 |        1        |
|          3 |   1.08723 | PCA             |       0.296659 |                0 |        0.297751 |        0          |      0.40559  |        1        |

### G7_RichCtx
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |   1.05184 | PCA             |       0.234204 |         0.239333 |        0.288985 |          0.332    |      0.476812 |        0.428667 |
|          1 |   1.06514 | PCA             |       0.287233 |         0.305333 |        0.256156 |          0.293333 |      0.456611 |        0.401333 |
|          2 |   1.07493 | PCA             |       0.259834 |         0.254667 |        0.306446 |          0.362667 |      0.43372  |        0.382667 |
|          3 |   1.07117 | PCA             |       0.270006 |         0.28     |        0.284076 |          0.324    |      0.445918 |        0.396    |

### G8_LabelCtx
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |  1.06086  | PCA             |       0.301257 |      0.00866667  |        0.239121 |          0.282    |      0.459622 |        0.709333 |
|          1 |  1.02687  | PCA             |       0.238057 |      0.0126667   |        0.244357 |          0.274667 |      0.517586 |        0.712667 |
|          2 |  1.02703  | PCA             |       0.256323 |      0.0533333   |        0.227386 |          0.253333 |      0.516292 |        0.693333 |
|          3 |  0.977183 | PCA             |       0.192456 |      0.000666667 |        0.235483 |          0.265333 |      0.572061 |        0.734    |

### G9_Teacher
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |   1.03655 | PCA             |       0.207967 |      0.000666667 |        0.299632 |          0.356667 |      0.492401 |        0.642667 |
|          1 |   1.00039 | PCA             |       0.184751 |      0.00866667  |        0.276081 |          0.331333 |      0.539168 |        0.66     |
|          2 |   1.09774 | PCA             |       0.316083 |      0.214       |        0.33385  |          0.383333 |      0.350067 |        0.402667 |
|          3 |   1.03787 | PCA             |       0.213462 |      0.0193333   |        0.293046 |          0.352    |      0.493493 |        0.628667 |

### G10_Full
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_BLOCK |   top1_freq_BLOCK |   mean_pi_PCA |   top1_freq_PCA |
|-----------:|----------:|:----------------|---------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
|          0 |  1.00352  | PCA             |       0.330469 |        0.0373333 |        0.160729 |         0.192667  |      0.508802 |        0.77     |
|          1 |  1.02293  | FULL            |       0.429996 |        0.792     |        0.161337 |         0.108     |      0.408668 |        0.1      |
|          2 |  0.988242 | PCA             |       0.2928   |        0         |        0.16451  |         0.184     |      0.542691 |        0.816    |
|          3 |  0.912953 | PCA             |       0.237269 |        0         |        0.140954 |         0.0973333 |      0.621776 |        0.902667 |
