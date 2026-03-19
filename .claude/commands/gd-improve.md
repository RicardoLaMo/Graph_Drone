---
name: gd-improve
description: Start a structured improvement branch. Verifies ground truth baseline first, creates a branch with naming convention feat/<engine>-improvement, and records what baseline must be beaten.
argument-hint: "<engine>  e.g.: regression  or  classification  or  both"
allowed-tools: Read, Write, Edit, Bash, Grep
---

$ARGUMENTS is the engine to improve: `regression`, `classification`, or `both`.

## Step 1: Read ground truth baseline (MANDATORY — read files, not memory)

**DO NOT proceed until you have read the actual result files.**

```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/elo_ranking.csv 2>/dev/null || echo "MISSING: elo_ranking.csv"`
```

```
!`grep -A 6 "Current best ELO" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/CLAUDE.md`
```

```
!`grep "GRAPHDRONE_VERSION" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/scripts/run_geopoe_benchmark.py | head -3`
```

```
!`cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research && git log --oneline -5`
```

Record the baseline ELOs from the result files (not CLAUDE.md, not memory). If `elo_ranking.csv` is missing, read `report.txt` and `run_log.txt`. If no result files exist at all, **STOP** and ask the user to run a baseline benchmark first.

## Step 2: Check for DO NOT violations

Read the full DO NOT section of CLAUDE.md:

```
!`grep -A 60 "DO NOT rules" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/CLAUDE.md`
```

Before proposing any changes, verify the proposed approach does NOT violate any rule listed in the CLAUDE.md DO NOT section (read above). Key rules by engine:

**If $ARGUMENTS = regression:**
- Do NOT use `bootstrap_full_only` (= vanilla TabPFN, ELO 1440)
- Do NOT add CatBoost or XGBoost (mis-routes in complex regions)
- Do NOT omit the MSE residual penalty (defer→1.0 collapse on diamonds)
- Do NOT treat 1514.7 as a regression target (it was a combined ELO, not regression-only)

**If $ARGUMENTS = classification:**
- Do NOT use `contextual_transformer` router (37:1 param/sample ratio on 78–100 OOF rows)
- Do NOT add CatBoost or XGBoost
- Do NOT re-enable GORA (kNN noisy on small N; no router to consume it anyway)
- Current classification baseline is GD 1479.5 vs TabPFN 1520.5 — improvement means closing this gap

**Either engine:**
- Do NOT mix up `run_geopoe_benchmark.py` ELOs with `run_smart_benchmark.py` ELOs — incompatible scales

If $ARGUMENTS involves any of the above, **STOP and warn the user** with the specific DO NOT rule and the measured failure that backed it.

## Step 3: Read current engine implementation

For the engine being improved, read the relevant sections:

```
!`grep -n "task_type.*regression\|task_type.*classification\|FULL\|SUB\|bootstrap_full_only\|contextual_transformer\|anchor_weight\|residual" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/scripts/run_geopoe_benchmark.py | head -40`
```

Read the relevant lines of model.py for the engine being changed.

## Step 4: Create improvement branch

```bash
cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research
git checkout -b feat/<engine>-improvement-$(date +%Y%m%d)
```

Replace `<engine>` with $ARGUMENTS (e.g. `feat/regression-improvement-20260320`).

## Step 5: Document the improvement plan

Before writing any code, produce a structured plan showing:

1. **Baseline to beat**:
   - Regression ELO: `<current>` (from result files)
   - Classification ELO: `<current>` (from result files)
   - Version string: `<GRAPHDRONE_VERSION>` (to be bumped)

2. **Hypothesis**: why will the proposed change improve ELO?

3. **Files to change**: exact file paths and line numbers

4. **Version string to use**: propose a new version string following the convention `v1-geopoe-<date><letter>` (e.g. `v1-geopoe-2026.03.20a` for the first change on that date)

5. **DO NOT check**: confirm none of the prohibited changes are being made

6. **Rollback plan**: how to restore the baseline if the change hurts ELO

**Wait for user confirmation of the plan before modifying any code.**

## Step 6: After implementation, run benchmark

Remind the user of the exact benchmark command:

```bash
cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research

# For regression:
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks regression --folds 0 1 2

# For classification:
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks classification --folds 0 1 2

# For both:
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --folds 0 1 2
```

**IMPORTANT**: GRAPHDRONE_VERSION must be bumped before running, or cached results from the old version will be used.

## Step 7: After results are in, call /gd-secure

Once the benchmark completes:
- If ELO improved: run `/gd-secure $ARGUMENTS <new_version_string>` to lock the result
- If ELO degraded: revert the code change and document what failed and why in CLAUDE.md DO NOT section

Do not merge to main until `/gd-secure` has been completed.
