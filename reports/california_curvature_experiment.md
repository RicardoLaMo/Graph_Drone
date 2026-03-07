# California Housing: Curvature-Aware Experiment
*Generated: 2026-03-07*

---
## 1. Objective
Primary question: does adding a curvature-based view or curvature-derived observer features improve predictive performance — especially on structurally difficult rows — compared with plain tabular baselines, single-graph kNN embeddings, or multi-view embeddings without curvature?

**Scientific stance:** Do not assume curvature helps. Prove or disprove it.

---
## 2. Experimental Setup
- **Dataset:** California Housing (sklearn), n=20,640 rows, 8 features.
- **Preprocessing:** RobustScaler. log1p on Population, AveOccup, AveRooms.
- **Splits:** 70/15/15 train/val/test (random_state=42, no leakage).
- **Graph construction:** sklearn NearestNeighbors, k=15.
- **4 Views:** FULL (all 8 features), GEO (Lat/Long), SOCIO (5 socioeconomic), CURVATURE (FULL + kappa-similarity edge weights).

---
## 3. Models Compared
| ID | Description |
|----|-------------|
| B0 | Mean predictor |
| B1 | MLP (3-layer, ReLU, early stopping) |
| B2 | HistGradientBoostingRegressor |
| M3 | GraphSAGE on FULL graph |
| M4 | GraphSAGE on GEO graph |
| M5 | GraphSAGE on SOCIO graph |
| M6 | Uniform ensemble of M3/M4/M5 |
| M7 | Learned combiner over M3/M4/M5 (no curvature) |
| M8A | kappa appended as node feature → GraphSAGE |
| M8B | kappa as 4th view in learned combiner |
| M9 | Observer-driven per-row combiner using [kappa, LID, LOF, density] |

---
## 4. Curvature Definition
- **kappa_i** = local PCA residual ratio in row i's k=15 kNN neighborhood.
- Specifically: fraction of neighborhood variance NOT explained by the top-2 principal directions.
- This is a **practical non-flatness proxy** — not exact Riemannian curvature.
- kappa: mean=0.4105, std=0.0578

---
## 5. Main Results

| model                |     rmse |      mae |           r2 |
|:---------------------|---------:|---------:|-------------:|
| B2_HGBR              | 0.443292 | 0.300658 |  0.85139     |
| M9_ObserverCombiner  | 0.521099 | 0.369965 |  0.794644    |
| M4_GEO               | 0.523334 | 0.370208 |  0.792879    |
| M7_LearnedCombiner   | 0.524729 | 0.371648 |  0.791773    |
| B1_MLP               | 0.572957 | 0.40828  |  0.751738    |
| M6_Uniform           | 0.592081 | 0.415511 |  0.734888    |
| M8B_FourViewCombiner | 0.603017 | 0.376077 |  0.725005    |
| M8A_KappaFeature     | 0.660259 | 0.46816  |  0.670318    |
| M5_SOCIO             | 0.679061 | 0.465194 |  0.651274    |
| M3_FULL              | 0.688285 | 0.484509 |  0.641737    |
| M8B_CurvView         | 0.705677 | 0.464134 |  0.623401    |
| B0_Mean              | 1.14992  | 0.911774 | -8.10623e-06 |

---
## 6. Curvature-Bin Analysis

| model                | bin    |     rmse |   n_rows |
|:---------------------|:-------|---------:|---------:|
| B0_Mean              | low    | 1.32853  |      994 |
| B0_Mean              | medium | 1.11461  |     1049 |
| B0_Mean              | high   | 0.992022 |     1053 |
| B1_MLP               | low    | 0.656966 |      994 |
| B1_MLP               | medium | 0.554658 |     1049 |
| B1_MLP               | high   | 0.501298 |     1053 |
| B2_HGBR              | low    | 0.503113 |      994 |
| B2_HGBR              | medium | 0.430393 |     1049 |
| B2_HGBR              | high   | 0.3928   |     1053 |
| M3_FULL              | low    | 0.818793 |      994 |
| M3_FULL              | medium | 0.637531 |     1049 |
| M3_FULL              | high   | 0.595905 |     1053 |
| M4_GEO               | low    | 0.606    |      994 |
| M4_GEO               | medium | 0.489581 |     1049 |
| M4_GEO               | high   | 0.468842 |     1053 |
| M5_SOCIO             | low    | 0.821601 |      994 |
| M5_SOCIO             | medium | 0.623359 |     1049 |
| M5_SOCIO             | high   | 0.57574  |     1053 |
| M6_Uniform           | low    | 0.697152 |      994 |
| M6_Uniform           | medium | 0.554213 |     1049 |
| M6_Uniform           | high   | 0.515687 |     1053 |
| M7_LearnedCombiner   | low    | 0.60705  |      994 |
| M7_LearnedCombiner   | medium | 0.493276 |     1049 |
| M7_LearnedCombiner   | high   | 0.468285 |     1053 |
| M8A_KappaFeature     | low    | 0.771661 |      994 |
| M8A_KappaFeature     | medium | 0.628346 |     1049 |
| M8A_KappaFeature     | high   | 0.57125  |     1053 |
| M8B_FourViewCombiner | low    | 0.805961 |      994 |
| M8B_FourViewCombiner | medium | 0.49024  |     1049 |
| M8B_FourViewCombiner | high   | 0.465331 |     1053 |
| M8B_CurvView         | low    | 0.89726  |      994 |
| M8B_CurvView         | medium | 0.616703 |     1049 |
| M8B_CurvView         | high   | 0.570353 |     1053 |
| M9_ObserverCombiner  | low    | 0.600112 |      994 |
| M9_ObserverCombiner  | medium | 0.489322 |     1049 |
| M9_ObserverCombiner  | high   | 0.468938 |     1053 |

---
## 7. Ablation Notes
- M7 (no curvature) vs M9 (curvature observer): direct curvature value test.
- M8A (kappa as feature) vs M3 (no kappa): isolated feature contribution.
- M8B (4-view) vs M6/M7 (3-view): curvature as graph view test.

---
## 8. Stability Results

|   spearman_k10_k20 |   pval_k10_k20 |   spearman_k10_k30 |   pval_k10_k30 |   spearman_k20_k30 |   pval_k20_k30 |   top20pct_overlap_k10_k20 |   top20pct_overlap_k10_k30 |   top20pct_overlap_k20_k30 |
|-------------------:|---------------:|-------------------:|---------------:|-------------------:|---------------:|---------------------------:|---------------------------:|---------------------------:|
|           0.448687 |              0 |           0.364749 |              0 |           0.687902 |              0 |                   0.384205 |                   0.340601 |                   0.527616 |

---
## 9. Conclusion
### VERDICT: **CURVATURE NOT JUSTIFIED ON THIS DATASET**

H1 (graph > tabular): NO

H3 (curv > no-curv combiner): gain=0.0036

H5 (curv reduces high-kappa error): NO  (M9=0.4689 vs M7=0.4683)

> **Interpretation guideline:**
> If graph embeddings themselves do not beat HGBR, the curvature result should be interpreted with caution — curvature is an additive filter on top of graph structure.
> If curvature adds nothing beyond density / LOF / LID, the recommendation is to use simpler observer features instead.

---
*Outputs: `artifacts/california_metrics.csv`, `artifacts/curvature_bins.csv`, `figures/curvature_hist.png`, `figures/error_by_curvature_bin.png`, `figures/observer_correlation_heatmap.png`*