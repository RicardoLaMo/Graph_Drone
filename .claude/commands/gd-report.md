---
name: gd-report
description: Rich benchmark analysis. Reads granular CSV results and produces ELO + raw metrics side-by-side, explains ELO volatility in terms of matchup count and margin, and identifies where GD wins/loses and why.
argument-hint: "[regression|classification|both]  default: both"
allowed-tools: Read, Bash, Grep
---

$ARGUMENTS is `regression`, `classification`, or `both` (default: `both`).

## Step 1: Read the result files

**All numbers come from files — never from memory.**

```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/results_granular.csv 2>/dev/null || echo "MISSING: results_granular.csv — run the benchmark first"`
```

```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/elo_ranking.csv 2>/dev/null || echo "MISSING: elo_ranking.csv"`
```

```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/report.txt 2>/dev/null || echo "MISSING: report.txt"`
```

If `results_granular.csv` is missing, stop: "No results to analyze. Run `/gd-bench` first."

## Step 2: Regression analysis (if $ARGUMENTS = regression or both)

From the granular CSV, for each regression dataset × fold, compute GD vs TabPFN delta:

### 2a. Per-dataset metric table

Build this table from the CSV data:

| Dataset | Metric | GD mean | TPF mean | Δ | GD wins (N/3 folds) |
|---|---|---|---|---|---|
| california | R² | X.XXXX | X.XXXX | +/-0.00XX | N/3 |
| california | RMSE | X.XXX | X.XXX | -/+X.XXX | N/3 |
| ... | | | | | |

**Key insight**: Show both R² and RMSE because they tell different stories:
- R² = proportion of variance explained (scale-free, 0–1, higher better)
- RMSE = absolute prediction error in target units (scale-dependent, lower better)
- A GD win on R² but loss on RMSE (or vice versa) signals a dataset where the margin is ambiguous

### 2b. ELO interpretation for regression

ELO is computed over N_reg = (number of regression datasets × folds with valid results).

Explain:
- "With N matchups, each matchup is worth K×(1-E(win)) ELO points to the winner."
- "At 1500 vs 1500, winning one matchup is worth K/2 = 16 ELO points."
- "With N=18 matchups (6 datasets × 3 folds), GD's current ELO of XXXX means it wins approximately W/18 matchups."
- Compute the implied win rate from the ELO difference: `W% = 1/(1+10^((ELO_tabpfn - ELO_gd)/400))`
- Compare implied win rate to actual observed win rate from the CSV

### 2c. ELO volatility estimate

With only 18 regression matchups:
- "If one fold's result flipped (GD win → GD loss), ELO would change by approximately 2×K×(1-E) ≈ 20–25 points."
- List any datasets where GD wins by margin < 0.001 R² — these are effectively ties and could flip in a different fold split
- Flag datasets where GD wins 2/3 folds (not all 3) — these are the volatility sources

## Step 3: Classification analysis (if $ARGUMENTS = classification or both)

### 3a. Per-dataset metric table

| Dataset | F1-macro | AUC-ROC | PR-AUC | GD wins F1 (N/3) |
|---|---|---|---|---|
| diabetes | GD/TPF | GD/TPF | GD/TPF | N/3 |
| credit_g | ... | | | |
| ... | | | | |

**Key insight for classification metrics:**
- **F1-macro**: primary ELO metric; sensitive to class imbalance on credit_g and diabetes
- **AUC-ROC**: probability ranking quality; less sensitive to threshold; a model can have better AUC but worse F1 if calibration differs
- **PR-AUC**: most informative on imbalanced datasets (credit_g has ~30% positive rate)
- If GD wins AUC-ROC but loses F1, the issue is likely calibration/threshold, not ranking quality

### 3b. Current classification gap analysis

Current: GD 1479.5 vs TabPFN 1520.5 — TabPFN leads.

From the granular data, identify:
- Which datasets drive TabPFN's advantage (largest negative Δ F1)?
- Which datasets GD wins (positive Δ F1)?
- For each GD loss: is AUC-ROC also worse, or just F1? (F1 loss with AUC win → threshold/calibration issue)
- Is `defer` diagnostic showing 0.0 consistently (static GeoPOE path)?

### 3c. ELO volatility for classification

Same volatility analysis as regression. With 35/36 tasks (1 OOM), identify the matchups closest to 50/50 margin.

## Step 4: Combined ELO interpretation

If both engines were run, explain why combined ELO is NOT a meaningful summary:
- "Combined ELO mixes two different primary metrics (R² for regression, F1 for classification)"
- "A model that dominates regression can mask classification losses"
- "Always report regression ELO and classification ELO separately"
- Confirm the `elo_ranking.csv` contains separate per-task-type ELOs (it should — `build_report()` computes them)

## Step 5: Improvement targeting

Based on the analysis, identify the highest-value opportunities:

**For closing the classification gap (GD 1479.5 → need > 1520.5):**
- Which 3 datasets contribute most to TabPFN's advantage?
- What's the minimum per-dataset F1 improvement needed to flip the ELO?
- Example: "If diabetes improves by +0.005 F1 (currently GD 0.725 vs TPF 0.732), GD wins 2→3/3 folds on diabetes, gaining ~16 ELO. Combined with keeping current wins, classification ELO would reach approximately 149X."

**For extending the regression lead (GD 1523.2 → defend and push):**
- Which datasets are closest to flipping (GD wins 2/3 with margin < 0.002)?
- Any datasets where GD wins all 3 folds — these are the stable anchor wins

## Step 6: Summary table

Print a final consolidated table for the current benchmark run:

```
=== GraphDrone Benchmark Analysis ===
Version: <GRAPHDRONE_VERSION>
Run date: <from report.txt>

REGRESSION (N=18 matchups)
  GD ELO: XXXX  vs  TabPFN ELO: XXXX
  GD wins: W/18 matchups (implied by ELO: W%%)
  Stable wins (3/3 folds): <datasets>
  Volatile wins (2/3 folds, margin < 0.005): <datasets>
  Losses: <datasets>

CLASSIFICATION (N=35 matchups)
  GD ELO: XXXX  vs  TabPFN ELO: XXXX
  GD wins: W/35 matchups
  Datasets driving gap: <top 2 negative-delta datasets>
  AUC-ROC GD wins despite F1 loss: <if any>

NEXT BEST IMPROVEMENT TARGET:
  <highest-value dataset+metric where GD could close the gap>
```
