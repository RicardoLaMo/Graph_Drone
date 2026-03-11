# Phase 0c View-Home Diagnosis

## Scope

Phase 0c asks a narrower question than Phase 0b:

- when adaptive routing beats the fixed hedge but still loses to `FULL`, is it because the router is failing to harvest good non-`FULL` experts, or because the non-`FULL` experts are not strong enough in the first place?

This phase stayed diagnosis-only.

New tooling:

- [analyze_view_home_quality.py](/home/wliu23/projects/Graph_Drone/.worktrees/phase0c-view-quality-diagnosis/experiments/openml_regression_benchmark/scripts/analyze_view_home_quality.py)
- [summarize_view_home_suite.py](/home/wliu23/projects/Graph_Drone/.worktrees/phase0c-view-quality-diagnosis/experiments/openml_regression_benchmark/scripts/summarize_view_home_suite.py)

## Observed Facts

### California

Source:

- [router_view_home_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/california_housing_openml__r0f0/artifacts/router_view_home_summary.json)

Key read:

- adaptive still beats `FULL`
- but adaptive barely beats fixed overall
- on `GEO` home rows, adaptive capture is `0.2462` and fixed capture is `0.2658`
- on `LOWRANK` home rows, adaptive capture is `0.0475` and fixed capture is `0.0604`
- only `SOCIO` shows a positive adaptive capture gap over fixed, and that gap is small in absolute terms

Interpretation:

- on the unified OpenML California surface, the current router is not winning mainly by better non-`FULL` harvesting
- it looks more like a conservative risk-balance gain than a strong expert-selection gain

### Houses

Sources:

- representative run: [router_view_home_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_houses_seed_sweep/seed42/houses__r0f0/artifacts/router_view_home_summary.json)
- 10-seed suite: [router_view_home_suite_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_houses_seed_sweep/router_view_home_suite_summary.json)

Key read:

- `GEO` home rows are the main real win:
  - 10-seed capture gap vs fixed mean: `+0.00915`
- `DOMAIN` is mildly positive:
  - 10-seed capture gap vs fixed mean: `+0.00448`
- `LOWRANK` is neutral to slightly negative:
  - 10-seed capture gap vs fixed mean: `-0.00069`

Interpretation:

- houses is the cleanest current evidence of a real, but small, upside-harvesting mechanism
- the gain is mostly a `FULL + GEO` integration story, not a broad win across all views

### Miami

Source:

- [router_view_home_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/miami_housing__r0f0/artifacts/router_view_home_summary.json)

Key read:

- adaptive beats fixed overall, but all non-`FULL` view-home capture gaps are negative:
  - `GEO -0.0598`
  - `DOMAIN -0.0448`
  - `LOWRANK -0.1070`
- on `FULL`-home rows, adaptive harm is `6537.6`, better than fixed harm `10829.0`

Interpretation:

- Miami is not an upside-harvesting success
- adaptive beats fixed mainly by doing less damage on rows where `FULL` was already right
- on the rows where non-`FULL` views should matter, adaptive is actually worse than the fixed hedge

### Diamonds

Source:

- [router_view_home_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/diamonds__r0f0/artifacts/router_view_home_summary.json)

Key read:

- adaptive beats fixed overall, but that is not because it is broadly better on non-`FULL` home rows
- capture gaps vs fixed:
  - `GEO -0.1095`
  - `DOMAIN +0.0468`
  - `LOWRANK -0.0061`
- `DOMAIN` helps, but it is only about `3.5%` of rows
- on `FULL`-home rows, adaptive harm is `75.43`, better than fixed harm `89.66`

Interpretation:

- diamonds has the same broad pattern as Miami
- adaptive is mostly a safer hedge than fixed
- it is not a broad non-`FULL` harvesting win

## Why

The strongest current explanation is:

- the present GraphDrone router is a **mixed mechanism**
- on the best dataset (`houses`) it does a little real non-`FULL` upside harvesting
- on the loss cases (`miami_housing`, `diamonds`) it mostly acts as a **damage-control controller**

That means `FULL` can still beat adaptive for two linked reasons:

1. the non-`FULL` experts are not strong enough, or not cleanly separable enough, on the rows where they should help
2. the router is not precise enough to recover that upside, so its practical gain comes mostly from being less reckless than the fixed hedge

So the dominant failure mode is:

- **expert quality plus integration precision**

not:

- a simple implementation mismatch
- and not a proof that row-level routing is pointless

## What This Means

There are now three distinct cases.

### Case A: real but small integration win

Dataset:

- `houses`

Meaning:

- GraphDrone does have a real row-level integration signal here
- but that signal is narrow and mostly concentrated in `GEO`

### Case B: weak hedge-style win

Dataset:

- `california_housing_openml` on the unified surface

Meaning:

- adaptive still helps overall
- but the current win is not a strong proof that the router is harvesting specialized experts better than a fixed hedge

### Case C: damage control, not true harvesting

Datasets:

- `miami_housing`
- `diamonds`

Meaning:

- adaptive beats fixed mostly by reducing `FULL`-home damage
- not by exploiting non-`FULL` experts better than fixed

This is the clearest explanation yet for why `FULL` still beats adaptive on those datasets.

## How

The next architectural question should be framed like this:

- GraphDrone should not add more views or more routing complexity until it can prove that non-`FULL` experts are both high-quality and harvestable

That leads to three priority questions:

1. **Expert quality**
   On their own oracle-home rows, are the non-`FULL` experts strong enough to justify routing away from `FULL`?

2. **Harvestability**
   When those experts are strong enough, does the router have the right signals to identify them better than a conservative fixed hedge?

3. **Integration objective**
   Should the router be optimized explicitly for asymmetric regret against leaving `FULL`, rather than only for smooth prediction blending?

## Decision

Do not move into Phase 1 portfolio claims yet.

Do not add dynamic kNN or more views yet.

The next justified work is:

1. expert-quality diagnostics on oracle-home rows, extended beyond these four datasets if needed
2. a targeted test of whether a simpler `FULL + one non-FULL expert` setup beats the current multi-view hedge on Miami and diamonds
3. only then a new architecture branch, if the evidence shows that the missing piece is integration rather than expert weakness

## External Review

Gemini review:

- [cleaned_output.txt](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T152500Z-phase0c-view-home/20260311T154551Z/cleaned_output.txt)

Useful Gemini conclusion:

- the current mechanism is primarily a damage-control engine rather than an upside harvester on the hard datasets

Claude review:

- the first long-form Phase 0c bundle returned no usable output
- the compact retry hit the local Claude rate limit before completion

Relevant prior Claude context still exists from Phase 0b:

- [analysis.claude.txt](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T150500Z-phase0b-causal-diagnosis/analysis.claude.txt)
