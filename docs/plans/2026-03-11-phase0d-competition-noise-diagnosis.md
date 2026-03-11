# Phase 0d Competition-Noise Diagnosis

## Scope

Phase 0d asks:

- are extra views hurting because they are weak experts, because they create routing competition noise, or both?

This phase stayed diagnosis-only. It did not change the predictive architecture.

New tooling:

- [analyze_two_expert_competition.py](/home/wliu23/projects/Graph_Drone/.worktrees/phase0d-competition-noise-diagnosis/experiments/openml_regression_benchmark/scripts/analyze_two_expert_competition.py)
- [summarize_two_expert_suite.py](/home/wliu23/projects/Graph_Drone/.worktrees/phase0d-competition-noise-diagnosis/experiments/openml_regression_benchmark/scripts/summarize_two_expert_suite.py)

## Local Results

### California

Source:

- [router_two_expert_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/california_housing_openml__r0f0/artifacts/router_two_expert_summary.json)

Best pair:

- `FULL + GEO`

Metrics:

- full router RMSE: `0.38950`
- best two-expert RMSE: `0.38860`
- gain vs full router: `+0.00089`
- gain vs FULL expert: `+0.01315`

Read:

- competition noise is present
- removing the extra views improves over the full multi-view router
- California still supports a real `FULL + GEO` integration effect

### Houses

Sources:

- representative run: [router_two_expert_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_houses_seed_sweep/seed42/houses__r0f0/artifacts/router_two_expert_summary.json)
- 10-seed suite: [router_two_expert_suite_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_houses_seed_sweep/router_two_expert_suite_summary.json)

Best pair:

- `FULL + GEO` on all 10 seeds

Suite metrics:

- mean gain vs full router: `+0.0000715`
- positive fraction vs full router: `0.6`
- mean gain vs FULL expert: `+0.003589`

Read:

- the gain is small, but the preferred pair is stable
- houses continues to look like a narrow `FULL + GEO` success, not a broad multi-view success

### Miami

Source:

- [router_two_expert_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/miami_housing__r0f0/artifacts/router_two_expert_summary.json)

Best pair:

- `FULL + GEO`

Metrics:

- full router RMSE: `83079.12`
- best two-expert RMSE: `82654.55`
- gain vs full router: `+424.57`
- gain vs FULL expert: `-125.19`

Read:

- competition noise is clearly present
- but even the best simplified pair still loses to `FULL`
- so removing competition helps, but weak expert quality or weak separability remains

### Diamonds

Source:

- [router_two_expert_summary.json](/home/wliu23/projects/Graph_Drone/.worktrees/phase0-router-mechanism-proof/experiments/openml_regression_benchmark/reports_phase0_mechanism_full/diamonds__r0f0/artifacts/router_two_expert_summary.json)

Best pair:

- `FULL + LOWRANK`

Metrics:

- full router RMSE: `539.34`
- best two-expert RMSE: `524.52`
- gain vs full router: `+14.81`
- gain vs FULL expert: `-2.02`

Read:

- competition noise is again real
- but the best pair still cannot surpass `FULL`

## Why

The strongest supported explanation is:

- **both competition noise and expert weakness matter**

But the order now looks clear:

1. **competition noise is the first-order problem inside the current multi-view router**
   - best two-expert pairs beat the full router on every dataset we tested
2. **expert weakness or weak separability is the limiting problem after that**
   - on Miami and diamonds, even the best simplified pair still loses to `FULL`

So the current GraphDrone failure is not “the router cannot route.”
It is:

- the full router is asked to arbitrate across too many weakly helpful experts
- once that competition is reduced, performance improves
- but on some datasets the remaining non-`FULL` expert still does not justify leaving `FULL`

## What This Means

### Strongest supported mechanism

- competition noise inside the multi-view router is real and measurable

Local proof:

- California: two-expert beats full router
- houses: best pair is stable and slightly better than full router
- Miami: two-expert beats full router by `424.57`
- diamonds: two-expert beats full router by `14.81`

### Strongest rejected explanation

- “more experts are inherently better”

The data rejects that directly.

Extra experts are not helping by default. In the current system they often dilute the useful signal from the best non-`FULL` expert.

### Residual explanation after pruning

- weak expert quality or weak routing separability still matters

Evidence:

- Miami best pair still loses to `FULL` by `125.19`
- Diamonds best pair still loses to `FULL` by `2.02`

So the right conclusion is not:

- “prune to two experts and we are done”

It is:

- “the current full router is too noisy, and the remaining expert gap must still be solved”

## Research Synthesis

This local result lines up with the MoE literature.

### Shazeer et al. 2017

Source:

- https://arxiv.org/abs/1701.06538

Relevant idea:

- sparse MoE routing has real algorithmic difficulty and needs explicit machinery like noisy gating and load balancing to keep expert competition under control

Why it matters here:

- our local result is a small-scale analogue
- adding more experts without strong routing control increases competition pressure and can reduce quality instead of increasing it

### Chen et al. 2022

Source:

- https://arxiv.org/abs/2208.02813

Relevant idea:

- MoE gains depend on intrinsic cluster structure and conditions that let routing and specialization separate the data cleanly

Why it matters here:

- Miami and diamonds look like weak-separability cases for the current view family
- if the underlying regimes are not cleanly isolated by the available routing signal, the generalist `FULL` expert remains hard to beat

### Lee et al. 2023

Source:

- https://aclanthology.org/2023.emnlp-main.217/

Relevant idea:

- learned routing helps when there is structured overlap and a learnable routing signal, and should be compared against weaker routing baselines rather than assumed beneficial by default

Why it matters here:

- our local evidence matches that pattern
- `houses` and California show some learnable routing structure
- Miami and diamonds do not yet show enough separable signal for the current views

## Signal-To-Noise Interpretation

The current GraphDrone router can be read through a simple signal-to-noise lens:

- **signal** = the gain available from the best non-`FULL` expert on the rows where it should help
- **noise** = the cost of forcing the router to compare multiple weakly useful or correlated experts

Phase 0d says:

- the noise term is definitely nontrivial
- pruning experts improves the decision quality
- but on Miami and diamonds the remaining signal is still too weak to beat `FULL`

So the next architecture question is not:

- how do we add more experts?

It is:

- how do we reduce routing competition and raise the quality or separability of the one non-`FULL` expert that actually matters?

## Decision

Do not move to broader Phase 1 portfolio claims yet.

The next justified move is:

1. test explicit `FULL + one expert` branches on the loss cases
2. measure whether stronger view construction or stronger routing signal is the real bottleneck
3. only then decide whether the next architecture change should target:
   - expert construction
   - router objective
   - or both

## External Review

Gemini review:

- [cleaned_output.txt](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T160500Z-phase0d-competition-noise/20260311T161705Z/cleaned_output.txt)

Useful Gemini contribution:

- the strongest local read is the same as ours:
  - competition noise explains why two-expert beats full router
  - expert weakness explains why two-expert still loses to `FULL` on Miami and diamonds

Claude review:

- attempted in [20260311T160500Z-phase0d-competition-noise](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T160500Z-phase0d-competition-noise)
- blocked by the local Claude rate limit during this phase

The last usable Claude result remains the earlier causal diagnosis:

- [analysis.claude.txt](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T150500Z-phase0b-causal-diagnosis/analysis.claude.txt)
