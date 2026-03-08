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

| model       |     rmse |      mae |        r2 |
|:------------|---------:|---------:|----------:|
| B1_HGBR     | 0.628481 | 0.433651 |  0.756047 |
| G2_GoRA_v1  | 2.33588  | 1.91286  | -2.36994  |
| G7_RichCtx  | 0.926799 | 0.69894  |  0.469492 |
| G8_LabelCtx | 0.745482 | 0.556338 |  0.656762 |
| G9_Teacher  | 0.815978 | 0.603421 |  0.588776 |
| G10_Full    | 0.840499 | 0.642083 |  0.56369  |

## Ablation ladder (G7 → G10)

| Gate | Δ component | Result |
|------|-------------|--------|
| G7 vs G2 | +ViewSpecificEmbed +ctx^(m) in router | **✅** G7=0.9268 G2=2.3359 |
| G8 vs G7 | +label_ctx (router + value aug) | **✅** G8=0.7455 G7=0.9268 |
| G9 vs G8 | +teacher z_anc as cross-attn query | **❌** G9=0.8160 G8=0.7455 |
| G10 vs G9 | +alpha-gate prediction fusion | **❌** G10=0.8405 G9=0.8160 |
| G10 vs B1 | GoRA v3 vs HGBR | **❌ HGBR still leads** G10=0.8405 B1=0.6285 |
| G10 vs B2 | GoRA v3 vs TabPFN | **❌ TabPFN leads** G10=0.8405 B2=N/A |

## View agreement score
Mean agree_score = 0.112

## Head-View Affinity by model

### G2_GoRA_v1
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |   1.38554 | SOCIO           |       0.247389 |            0.065 |      0.246548 |           0.055 |        0.266113 |              0.88 |          0.239951 |               0     |
|          1 |   1.38564 | FULL            |       0.265375 |            0.875 |      0.248237 |           0.05  |        0.242412 |              0    |          0.243975 |               0.075 |
|          2 |   1.37653 | FULL            |       0.304669 |            0.965 |      0.217693 |           0.005 |        0.219967 |              0    |          0.257671 |               0.03  |
|          3 |   1.38512 | FULL            |       0.268232 |            0.81  |      0.246967 |           0.005 |        0.250498 |              0.18 |          0.234304 |               0.005 |

### G7_RichCtx
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |   1.11495 | GEO             |       0.144427 |            0.035 |      0.597575 |           0.93  |        0.137509 |             0     |         0.120489  |               0.035 |
|          1 |   1.14225 | GEO             |       0.188213 |            0     |      0.55857  |           0.94  |        0.177108 |             0.005 |         0.0761095 |               0.055 |
|          2 |   1.14691 | GEO             |       0.156135 |            0     |      0.567849 |           1     |        0.099275 |             0     |         0.176741  |               0     |
|          3 |   1.24553 | GEO             |       0.16687  |            0     |      0.492591 |           0.975 |        0.141365 |             0     |         0.199174  |               0.025 |

### G8_LabelCtx
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |  1.06036  | GEO             |      0.0876233 |                0 |      0.604779 |           1     |        0.223711 |             0     |         0.0838862 |                   0 |
|          1 |  1.18732  | GEO             |      0.122619  |                0 |      0.542544 |           0.945 |        0.161457 |             0.055 |         0.17338   |                   0 |
|          2 |  0.727151 | GEO             |      0.0509914 |                0 |      0.791846 |           1     |        0.105168 |             0     |         0.0519949 |                   0 |
|          3 |  1.28751  | GEO             |      0.193225  |                0 |      0.454624 |           1     |        0.171181 |             0     |         0.180969  |                   0 |

### G9_Teacher
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |  1.23587  | GEO             |      0.0686025 |            0.005 |      0.437529 |           0.995 |        0.237633 |             0     |          0.256236 |               0     |
|          1 |  0.936978 | GEO             |      0.13435   |            0.005 |      0.695601 |           0.99  |        0.104133 |             0.005 |          0.065916 |               0     |
|          2 |  1.27168  | GEO             |      0.149934  |            0.01  |      0.448999 |           0.99  |        0.25813  |             0     |          0.142937 |               0     |
|          3 |  0.932149 | GEO             |      0.099875  |            0     |      0.677757 |           0.995 |        0.182806 |             0     |          0.039562 |               0.005 |

### G10_Full
|   head_idx |   entropy | dominant_view   |   mean_pi_FULL |   top1_freq_FULL |   mean_pi_GEO |   top1_freq_GEO |   mean_pi_SOCIO |   top1_freq_SOCIO |   mean_pi_LOWRANK |   top1_freq_LOWRANK |
|-----------:|----------:|:----------------|---------------:|-----------------:|--------------:|----------------:|----------------:|------------------:|------------------:|--------------------:|
|          0 |   1.14224 | GEO             |      0.075113  |                0 |      0.557656 |            1    |        0.19119  |              0    |         0.176041  |                   0 |
|          1 |   1.00856 | GEO             |      0.0897538 |                0 |      0.657025 |            0.99 |        0.156442 |              0.01 |         0.0967795 |                   0 |
|          2 |   1.02383 | GEO             |      0.0886946 |                0 |      0.651128 |            0.99 |        0.141745 |              0.01 |         0.118433  |                   0 |
|          3 |   1.23479 | GEO             |      0.202439  |                0 |      0.498773 |            0.97 |        0.172429 |              0.03 |         0.12636   |                   0 |
