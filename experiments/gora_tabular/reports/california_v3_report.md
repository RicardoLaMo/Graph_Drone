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
| G9_Teacher  |   0.509864 |   0.359586 |   0.803403 |
| G10_Full    |   0.520836 |   0.366006 |   0.794852 |

## Ablation ladder (G7 → G10)

| Gate | Δ component | Result |
|------|-------------|--------|
| G7 vs G2 | +ViewSpecificEmbed +ctx^(m) in router | **❌** G7=0.4929 G2=0.4546 |
| G8 vs G7 | +label_ctx (router + value aug) | **❌** G8=0.5169 G7=0.4929 |
| G9 vs G8 | +teacher z_anc as cross-attn query | **✅** G9=0.5099 G8=0.5169 |
| G10 vs G9 | +alpha-gate prediction fusion | **❌** G10=0.5208 G9=0.5099 |
| G10 vs B1 | GoRA v3 vs HGBR | **❌ HGBR still leads** G10=0.5208 B1=0.4433 |
| G10 vs B2 | GoRA v3 vs TabPFN | **❌ TabPFN leads** G10=0.5208 B2=N/A |

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
|          0 |  0.936924 | GEO             |      0.0916025 |        0.0151809 |      0.691748 |        0.968023 |       0.0642131 |         0.0167959 |         0.152437  |          0          |
|          1 |  0.97717  | GEO             |      0.197366  |        0.111757  |      0.657195 |        0.86854  |       0.0690444 |         0.0197028 |         0.0763946 |          0          |
|          2 |  0.999301 | GEO             |      0.102766  |        0.0642765 |      0.663074 |        0.933786 |       0.0844631 |         0         |         0.149696  |          0.00193798 |
|          3 |  0.766326 | GEO             |      0.110112  |        0.0436047 |      0.775907 |        0.930556 |       0.0560041 |         0         |         0.0579781 |          0.0258398  |

### G10_Full
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |  0.888204 | GEO             |      0.0787349 |      0           |      0.720319 |        0.954457 |       0.0690879 |        0.0229328  |         0.131858  |          0.0226098  |
|          1 |  0.738305 | GEO             |      0.0806775 |      0.00355297  |      0.791699 |        0.965439 |       0.0562434 |        0.0242248  |         0.0713792 |          0.00678295 |
|          2 |  0.799151 | GEO             |      0.0813918 |      0           |      0.765096 |        0.968669 |       0.0593959 |        0.0313307  |         0.0941164 |          0          |
|          3 |  0.936264 | GEO             |      0.0918896 |      0.000645995 |      0.680341 |        0.916344 |       0.048466  |        0.00258398 |         0.179304  |          0.0804264  |
