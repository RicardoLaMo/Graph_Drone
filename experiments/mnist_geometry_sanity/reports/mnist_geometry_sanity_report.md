# MNIST-784: Geometry Sanity Check Report
*Generated: 2026-03-07*  |  Branch: `feature/mnist-geometry-sanity-check`

> ⚠️ This report is isolated from the California Housing experiment. No prior files modified.

## 1. Mission
Test whether kNN + graph embedding + curvature-view pipeline detects and exploits hidden pixel geometry in flattened MNIST-784.

## 2. Main Results

| model            |   accuracy |   macro_f1 |   log_loss |
|:-----------------|-----------:|-----------:|-----------:|
| M2_HGBR          |   0.955333 |  0.954979  |   0.174404 |
| M8_Uniform       |   0.954    |  0.953903  | nan        |
| M9_Combiner      |   0.954    |  0.953903  | nan        |
| M11_ObsCombiner  |   0.952    |  0.951824  | nan        |
| M1_MLP           |   0.946    |  0.945478  |   0.206069 |
| M7_PCA           |   0.945333 |  0.944775  | nan        |
| M3_XGBoost       |   0.942667 |  0.942186  |   0.188344 |
| M10_KappaFeature |   0.938667 |  0.937829  | nan        |
| M6_BLOCK         |   0.938    |  0.937178  | nan        |
| M5_FULL          |   0.936667 |  0.936277  | nan        |
| M0_Majority      |   0.112667 |  0.0202516 | nan        |

## 3. Curvature Statistics
- kappa mean=0.4924, std=0.0695
- kappa has useful variance

## 4. Multi-Scale Stability

|   spearman_k10_k20 |   pval_k10_k20 |   top20pct_overlap_k10_k20 |   spearman_k10_k30 |   pval_k10_k30 |   top20pct_overlap_k10_k30 |   spearman_k20_k30 |   pval_k20_k30 |   top20pct_overlap_k20_k30 |
|-------------------:|---------------:|---------------------------:|-------------------:|---------------:|---------------------------:|-------------------:|---------------:|---------------------------:|
|             0.6628 |              0 |                     0.4885 |           0.607748 |              0 |                      0.459 |            0.84688 |              0 |                     0.6465 |

## 5. Per-Bin Accuracy

| model            | bin    |   accuracy |   n_rows |
|:-----------------|:-------|-----------:|---------:|
| M0_Majority      | low    | 0.312377   |      509 |
| M0_Majority      | medium | 0.0160966  |      497 |
| M0_Majority      | high   | 0.00404858 |      494 |
| M1_MLP           | low    | 0.950884   |      509 |
| M1_MLP           | medium | 0.939638   |      497 |
| M1_MLP           | high   | 0.947368   |      494 |
| M2_HGBR          | low    | 0.954813   |      509 |
| M2_HGBR          | medium | 0.953722   |      497 |
| M2_HGBR          | high   | 0.95749    |      494 |
| M3_XGBoost       | low    | 0.939096   |      509 |
| M3_XGBoost       | medium | 0.949698   |      497 |
| M3_XGBoost       | high   | 0.939271   |      494 |
| M5_FULL          | low    | 0.931238   |      509 |
| M5_FULL          | medium | 0.94165    |      497 |
| M5_FULL          | high   | 0.937247   |      494 |
| M6_BLOCK         | low    | 0.941061   |      509 |
| M6_BLOCK         | medium | 0.935614   |      497 |
| M6_BLOCK         | high   | 0.937247   |      494 |
| M7_PCA           | low    | 0.939096   |      509 |
| M7_PCA           | medium | 0.943662   |      497 |
| M7_PCA           | high   | 0.953441   |      494 |
| M8_Uniform       | low    | 0.948919   |      509 |
| M8_Uniform       | medium | 0.953722   |      497 |
| M8_Uniform       | high   | 0.959514   |      494 |
| M9_Combiner      | low    | 0.948919   |      509 |
| M9_Combiner      | medium | 0.953722   |      497 |
| M9_Combiner      | high   | 0.959514   |      494 |
| M10_KappaFeature | low    | 0.929273   |      509 |
| M10_KappaFeature | medium | 0.94165    |      497 |
| M10_KappaFeature | high   | 0.945344   |      494 |
| M11_ObsCombiner  | low    | 0.94499    |      509 |
| M11_ObsCombiner  | medium | 0.953722   |      497 |
| M11_ObsCombiner  | high   | 0.95749    |      494 |

## 6. Hypothesis Results
```
H1 (graph > MLP): NO — best_sage=0.9453, mlp=0.9460
H2 (multi-view > single): YES
H3 (curv adds value): NO — M11=0.9520 M9=0.9540
H5 (curv helps high-kappa): NO/N/A
H7 (best graph > XGBoost): YES
```

## 7. XGBoost & TabPFN Comparison
- TabPFN was run on a ≤1024-sample subset (documented public limit). Result not directly comparable to full-dataset models.
- XGBoost was evaluated on the full train/test split.

## 8. Conclusion
### VERDICT: **WARNING SIGN: HIDDEN GEOMETRY WAS NOT MEANINGFULLY EXPLOITED**

> If graph embeddings do not beat MLP on MNIST, that is a strong warning sign—
> MNIST has well-known hidden geometry that a proper graph pipeline should exploit.

*Outputs: `artifacts/metrics.csv`, `figures/curvature_hist.png`, `figures/view_comparison.png`*