# Phase 0e Signal-Noise Tradeoff

## Scope

Phase 0e answers the tradeoff question directly:

- when should GraphDrone leverage prior views as experts, and when do those same views mostly add competition noise?

This phase stayed diagnosis-only. It did not change the predictive model.

New tooling:

- [analyze_signal_noise_tradeoff.py](/home/wliu23/projects/Graph_Drone/.worktrees/phase0e-signal-noise-tradeoff/experiments/openml_regression_benchmark/scripts/analyze_signal_noise_tradeoff.py)
- [summarize_signal_noise_suite.py](/home/wliu23/projects/Graph_Drone/.worktrees/phase0e-signal-noise-tradeoff/experiments/openml_regression_benchmark/scripts/summarize_signal_noise_suite.py)

## Core Tradeoff

The current tradeoff can be stated simply:

- **views are useful when they add a specialist signal that survives routing competition**
- **views are harmful when they add more competition noise than recoverable specialist value**

So the decision is not:

- “more views or fewer views”

It is:

- “does one non-`FULL` view contribute enough clean marginal value to survive both expert weakness and routing competition?”

## Local Results

### California

Source:

- [router_signal_noise_tradeoff.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/california_housing_openml__r0f0/artifacts/router_signal_noise_tradeoff.json)

Classification:

- `useful_signal_obscured_by_competition`

Best view:

- `GEO`

Read:

- best pair `FULL + GEO` beats the full router by `+0.00089`
- best pair also beats `FULL` by `+0.01315`
- this means California still has real specialist signal
- but the full multi-view router is diluting some of that value

Important nuance:

- the best-view capture gap vs fixed is negative on `GEO`
- so the specialist signal is real, but the current dense multi-view router is not harvesting it cleanly

### Houses

Sources:

- representative run: [router_signal_noise_tradeoff.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_houses_seed_sweep/seed42/houses__r0f0/artifacts/router_signal_noise_tradeoff.json)
- suite: [router_signal_noise_suite_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_houses_seed_sweep/router_signal_noise_suite_summary.json)

Classification:

- `useful_signal_obscured_by_competition` in `6/10` seeds
- `weak_competition_effect` in `4/10` seeds

Best view:

- `GEO` in `10/10` seeds

Read:

- competition-noise gain vs full router is positive but small
- best pair gain vs `FULL` is stably positive
- best-view capture gap vs fixed is positive on average

So houses is the cleanest current example of the desired tradeoff:

- one prior view adds real value
- extra competing views are mostly unnecessary

### Miami

Source:

- [router_signal_noise_tradeoff.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/miami_housing__r0f0/artifacts/router_signal_noise_tradeoff.json)

Classification:

- `competition_noise_plus_weak_expert`

Best view:

- `GEO`

Read:

- best pair beats the full router by `+424.57`
- but still loses to `FULL` by `125.19`
- best-view capture gap vs fixed is negative

Interpretation:

- pruning helps, so competition noise is real
- but the remaining specialist is still not strong enough or separable enough to justify leaving `FULL`

### Diamonds

Source:

- [router_signal_noise_tradeoff.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/diamonds__r0f0/artifacts/router_signal_noise_tradeoff.json)

Classification:

- `competition_noise_plus_weak_expert`

Best view:

- `LOWRANK`

Read:

- best pair beats the full router by `+14.81`
- but still loses to `FULL` by `2.02`
- best-view capture gap vs fixed is slightly negative

Interpretation:

- same pattern as Miami
- noise reduction helps
- but expert quality or separability still limits the ceiling

## Why

The strongest current interpretation is:

- GraphDrone should leverage prior views only when they produce a **positive marginal expert** after noise is stripped away

That implies a sequence:

1. identify the one non-`FULL` view with real marginal value
2. test whether it still helps when extra competitors are removed
3. only then consider a richer routing family

This is the practical signal-to-noise lesson:

- **signal** = a view that still helps after being paired directly with `FULL`
- **noise** = the extra degradation introduced when several weak or overlapping views compete in the same router

## What This Rejects

### Rejected: all prior views should be kept because they may help somewhere

The current evidence does not support that.

Extra views often hurt by introducing competition noise.

### Rejected: all prior views are bad and should be removed

This is also too strong.

- California still has a real `FULL + GEO` gain
- houses has a stable `FULL + GEO` pattern

So the right policy is selective leverage, not blanket removal.

## Research Connection

The tradeoff matches the MoE literature.

### Shazeer et al. 2017

Source:

- https://arxiv.org/abs/1701.06538

Relevant point:

- more experts increase routing difficulty and require mechanisms to control expert competition

Local match:

- the full router is worse than the best two-expert pair on every dataset we tested

### Chen et al. 2022

Source:

- https://arxiv.org/abs/2208.02813

Relevant point:

- MoE gains depend on intrinsic cluster structure and on experts specializing cleanly enough to justify routing

Local match:

- California and houses look like at least partial cluster/separability wins
- Miami and diamonds do not yet

### Lee et al. 2023

Source:

- https://aclanthology.org/2023.emnlp-main.217/

Relevant point:

- learned routing helps when routing signal is structured and learnable; it should not be assumed beneficial by default

Local match:

- our prior views only help when one specialist is clearly preferred
- otherwise the router mainly spends effort resolving weak competition

## Design Implication

At this stage, the best design direction is:

- do **not** build around a broad dense router over many prior views
- do **not** throw away prior views entirely
- instead, treat prior views as a **candidate expert bank** and promote only the views that survive the signal-vs-noise test

Concretely, the current evidence suggests:

- `GEO` is the only consistently promotable specialist on California and houses
- `LOWRANK` is conditionally useful on diamonds, but not yet enough to beat `FULL`
- Miami still does not have a non-`FULL` expert strong enough to justify a routed win over `FULL`

## Next Step

The next justified move is:

1. build explicit `FULL + one expert` experimental lines for the datasets where a candidate specialist survived this diagnosis
2. stop treating all views as equally eligible experts
3. only revisit richer routing after one non-`FULL` expert can reliably beat the current dense router and justify itself against `FULL`

## External Review

Gemini review:

- [cleaned_output.txt](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T164500Z-phase0e-signal-noise/20260311T163049Z/cleaned_output.txt)

Useful Gemini conclusion:

- the current router is over-inclusive
- prior views help only when the best specialist survives pruning and still justifies itself against `FULL`

Claude review:

- not rerun here because the local Claude CLI remained rate-limited during this phase

The most recent usable Claude diagnosis remains:

- [analysis.claude.txt](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T150500Z-phase0b-causal-diagnosis/analysis.claude.txt)
