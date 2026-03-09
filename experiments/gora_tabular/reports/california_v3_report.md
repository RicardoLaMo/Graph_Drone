# GoRA-Tabular v3 (MQ-GoRA): california_v3
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

| model       |       rmse |        mae |         r2 |
|:------------|-----------:|-----------:|-----------:|
| B1_HGBR     |   0.443292 |   0.300658 |   0.85139  |
| B2_TabPFN   | nan        | nan        | nan        |
| G2_GoRA_v1  |   0.454643 |   0.314812 |   0.843683 |
| G7_RichCtx  |   0.49288  |   0.344011 |   0.816283 |
| G8_LabelCtx |   0.516909 |   0.368382 |   0.797934 |
| G9_Teacher  |   0.50986  |   0.35959  |   0.803407 |
| G10_Full    |   0.520906 |   0.36607  |   0.794797 |

## Ablation ladder (G7 → G10)

| Gate | Δ component | Result |
|------|-------------|--------|
| G7 vs G2 | +ViewSpecificEmbed +ctx^(m) in router | **❌** G7=0.4929 G2=0.4546 |
| G8 vs G7 | +label_ctx (router + value aug) | **❌** G8=0.5169 G7=0.4929 |
| G9 vs G8 | +teacher z_anc as cross-attn query | **✅** G9=0.5099 G8=0.5169 |
| G10 vs G9 | +alpha-gate prediction fusion | **❌** G10=0.5209 G9=0.5099 |
| G10 vs B1 | GoRA v3 vs HGBR | **❌ HGBR still leads** G10=0.5209 B1=0.4433 |
| G10 vs B2 | GoRA v3 vs TabPFN | **❌ TabPFN leads** G10=0.5209 B2=N/A |

## View agreement score
Mean agree_score = 0.112

## Head-View Affinity by model

### G2_GoRA_v1
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |   1.23453 | GEO             |       0.163265 |        0         |      0.506454 |        0.893734 |        0.176194 |         0.106266  |          0.154087 |                   0 |
|          1 |   1.34134 | GEO             |       0.227829 |        0.120801  |      0.383341 |        0.861434 |        0.192328 |         0.0177649 |          0.196503 |                   0 |
|          2 |   1.3406  | GEO             |       0.238123 |        0.181202  |      0.38022  |        0.818798 |        0.177436 |         0         |          0.20422  |                   0 |
|          3 |   1.26479 | GEO             |       0.190186 |        0.0604005 |      0.477751 |        0.891796 |        0.167594 |         0.0478036 |          0.164469 |                   0 |

### G7_RichCtx
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |  0.741215 | GEO             |      0.146962  |        0.107881  |      0.769155 |        0.853359 |       0.0605008 |         0.0377907 |         0.0233819 |         0.000968992 |
|          1 |  0.630507 | GEO             |      0.09304   |        0.0813953 |      0.827715 |        0.891473 |       0.0291896 |         0         |         0.0500562 |         0.0271318   |
|          2 |  0.840131 | GEO             |      0.186964  |        0.14115   |      0.716178 |        0.838824 |       0.0319661 |         0.0200258 |         0.0648918 |         0           |
|          3 |  0.717817 | GEO             |      0.0643837 |        0.0109819 |      0.79855  |        0.895672 |       0.0487058 |         0.0193798 |         0.088361  |         0.0739664   |

### G8_LabelCtx
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |  1.24369  | GEO             |       0.205229 |      0.000322997 |      0.492424 |        0.949289 |       0.165662  |       0.00807494  |          0.136685 |         0.0423127   |
|          1 |  1.02111  | GEO             |       0.124359 |      0.0332687   |      0.654662 |        0.965116 |       0.0947641 |       0.00161499  |          0.126214 |         0           |
|          2 |  0.920652 | GEO             |       0.108807 |      0.0261628   |      0.708789 |        0.971899 |       0.0796622 |       0.00193798  |          0.102742 |         0           |
|          3 |  1.11625  | GEO             |       0.130071 |      0.0258398   |      0.572832 |        0.973514 |       0.0781588 |       0.000322997 |          0.218938 |         0.000322997 |

### G9_Teacher
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |  0.94005  | GEO             |       0.092263 |        0.0151809 |      0.689961 |        0.968023 |       0.064324  |         0.0167959 |         0.153451  |          0          |
|          1 |  0.977517 | GEO             |       0.197455 |        0.112726  |      0.656994 |        0.868217 |       0.0690434 |         0.0190568 |         0.0765074 |          0          |
|          2 |  1.00009  | GEO             |       0.103071 |        0.0645995 |      0.662625 |        0.933463 |       0.0844763 |         0         |         0.149829  |          0.00193798 |
|          3 |  0.767836 | GEO             |       0.110501 |        0.0445736 |      0.77523  |        0.930233 |       0.0561807 |         0         |         0.0580885 |          0.0251938  |

### G10_Full
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |  0.890996 | GEO             |      0.0790725 |      0           |      0.718955 |        0.953488 |       0.0695636 |        0.0235788  |         0.132408  |          0.0229328  |
|          1 |  0.743098 | GEO             |      0.0810223 |      0.00322997  |      0.789742 |        0.964147 |       0.0568139 |        0.0258398  |         0.0724232 |          0.00678295 |
|          2 |  0.802061 | GEO             |      0.0817667 |      0           |      0.763843 |        0.9677   |       0.0599056 |        0.0322997  |         0.0944845 |          0          |
|          3 |  0.939604 | GEO             |      0.0920807 |      0.000645995 |      0.678379 |        0.911176 |       0.0489312 |        0.00290698 |         0.180608  |          0.0852713  |
