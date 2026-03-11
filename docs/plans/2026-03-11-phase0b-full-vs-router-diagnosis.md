# Phase 0b FULL vs Router Diagnosis

## Scope

Phase 0b answers one question:

- why can `FULL` still beat the adaptive router even though the adaptive router clearly changes row-level weights and often beats the fixed hedge?

This phase did not introduce a new model. It added diagnosis tooling only:

- [analyze_router_full_regret.py](/home/wliu23/projects/Graph_Drone/.worktrees/phase0b-router-causal-diagnosis/experiments/openml_regression_benchmark/scripts/analyze_router_full_regret.py)
- [summarize_full_regret_suite.py](/home/wliu23/projects/Graph_Drone/.worktrees/phase0b-router-causal-diagnosis/experiments/openml_regression_benchmark/scripts/summarize_full_regret_suite.py)

The diagnosis is grounded in the existing Phase 0 runs:

- California full run: [router_full_regret_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/california_housing_openml__r0f0/artifacts/router_full_regret_summary.json)
- Miami full run: [router_full_regret_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/miami_housing__r0f0/artifacts/router_full_regret_summary.json)
- Diamonds full run: [router_full_regret_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/diamonds__r0f0/artifacts/router_full_regret_summary.json)
- Houses 10-seed suite summary: [router_full_regret_suite_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_houses_seed_sweep/router_full_regret_suite_summary.json)

## Observed Facts

### California

- `GraphDrone_FULL = 0.4018`
- `GraphDrone_router = 0.3895`
- `GraphDrone_router_fixed = 0.3895`
- adaptive minus fixed is only about `+0.00002` RMSE
- `FULL` is oracle-best on about `39.9%` of rows
- on rows where `FULL` is oracle-best, adaptive movement away from `FULL` still costs about `0.0248` absolute error on average
- on rows where `FULL` is not oracle-best, the adaptive router captures only about `12.6%` of the available improvement over `FULL`

### Houses

- 10-seed mean `adaptive minus FULL = +0.00352`
- 10-seed mean `adaptive minus fixed = +0.000393`
- `FULL` is oracle-best on about `42.7%` of rows
- false diversion mean cost is about `0.0165`
- adaptive capture ratio is about `0.141`
- adaptive capture is only slightly better than fixed, but it is positive in `9/10` seeds

### Miami

- `GraphDrone_FULL = 82529.36`
- `GraphDrone_router = 83079.12`
- `GraphDrone_router_fixed = 88149.98`
- adaptive beats fixed by about `5070.87` RMSE
- but adaptive still loses to `FULL` by about `549.75` RMSE
- `FULL` is oracle-best on about `46.9%` of rows
- false diversion mean cost is about `6537.6`
- adaptive capture ratio is only about `0.0689`, below fixed at `0.1361`

### Diamonds

- `GraphDrone_FULL = 522.51`
- `GraphDrone_router = 539.34`
- `GraphDrone_router_fixed = 559.39`
- adaptive beats fixed by about `20.06` RMSE
- but adaptive still loses to `FULL` by about `16.83` RMSE
- `FULL` is oracle-best on about `54.3%` of rows
- false diversion mean cost is about `75.43`
- adaptive capture ratio is only about `0.0368`, below fixed at `0.0804`

## Why

The strongest supported explanation is:

- the dominant problem is **data dynamics plus incomplete integration quality**, not a simple implementation bug and not a proof that row-level routing is mathematically useless

More precisely:

- the router is real: it moves weights row by row and usually beats the fixed hedge
- but on datasets where `FULL` is frequently oracle-best and non-`FULL` views are weak or hard to identify, the router does not capture enough of the available non-`FULL` upside to pay for its false diversions away from `FULL`

That is why `FULL` can still win:

1. `FULL` is often the safest expert on Miami and Diamonds.
2. The router does move away from `FULL`, but not with enough precision.
3. When it is wrong, the false diversion cost is large.
4. When `FULL` is wrong, the router captures only a small fraction of the available gain.

So the failure is not “the router does nothing.”  
It is “the router does too little good and too much expensive bad on the wrong datasets.”

## What This Rejects

### Rejected: implementation mismatch as the primary cause

This is not the strongest explanation because:

- adaptive routing beats the fixed hedge on California, houses, Miami, and diamonds
- the weight tensors and diagnostics are internally consistent
- the new tests pass on the diagnosis tooling

If this were mainly a code bug, adaptive would not improve over fixed so consistently.

### Rejected: row-level routing is mathematically pointless

This is also too strong.

- California still beats `FULL`
- houses shows a stable small adaptive-over-fixed gain across 10 seeds

So there is real row-level signal. The current issue is not that row-level integration cannot work. The issue is that the current integration is not selective enough to exploit that signal reliably across the portfolio.

## What The Current Evidence Supports

There are two different regimes.

### Regime A: small but real adaptive value

Datasets:

- California
- houses

Pattern:

- adaptive beats fixed
- false diversion cost is relatively low
- adaptive capture is limited, but not collapsed

Interpretation:

- row-level adaptation exists
- but the gain is still close to a conservative `FULL`-anchored hedge, especially on California unified OpenML

### Regime B: adaptive rescues fixed but still loses to `FULL`

Datasets:

- Miami
- diamonds

Pattern:

- adaptive is much better than fixed
- but false diversion cost is still too high
- realized gain on non-`FULL` rows is too small
- capture ratio collapses below the fixed hedge

Interpretation:

- the router sees some useful row-level signal
- but the current views and routing signals do not let it identify non-`FULL` wins precisely enough

## How

The next step should still be diagnosis and architecture integration, not new attachments.

Priority questions:

1. **Why is the non-`FULL` signal weak on Miami and diamonds?**
   We need per-view quality on the rows where each non-`FULL` view is oracle-best. If those views are weak even on their own “home” rows, the problem is view quality, not router cleverness.

2. **Why does capture ratio drop below the fixed hedge on loss-case datasets?**
   That is the sharpest sign of integration failure. The router is not just conservative; it is sometimes systematically worse at harvesting the available non-`FULL` gain.

3. **What should the router optimize?**
   Current routing behaves like MSE-trained soft weighting. The Phase 0b evidence suggests the architecture question is whether the router should explicitly model the asymmetric cost of leaving `FULL`, rather than only fitting a smooth blend.

## Decision

Keep the portfolio Phase 1 gate blocked.

Do not move into dynamic kNN, more views, or new adapters yet.

The next justified work is:

1. per-view home-subset quality analysis on California, houses, Miami, and diamonds
2. capture-ratio failure analysis relative to fixed on Miami and diamonds
3. only then decide whether the next architecture branch should be:
   - stronger view integration
   - explicit defer-to-`FULL` control
   - or better expert/view construction

## External Review

Claude review:

- [analysis.claude.txt](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T150500Z-phase0b-causal-diagnosis/analysis.claude.txt)

Useful Claude contribution:

- houses and Miami/diamonds should not be treated as the same failure mode
- Miami and diamonds look like severe false-diversion plus weak non-`FULL` signal
- houses looks more like a small-signal regime where the router helps only a little

Gemini review:

- [cleaned_output.txt](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T151200Z-phase0b-causal-diagnosis-compact/20260311T150952Z/cleaned_output.txt)

Useful Gemini contribution:

- the best current short description is that the router is functional, but its false diversion cost often outweighs the gain it captures from specialized views
- that again points to data dynamics and integration quality, not a simple broken implementation
