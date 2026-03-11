# GraphDrone.fit() Implementation Program

## Scope

This document freezes the implementation program for a portfolio-general `GraphDrone.fit()` line.

It is intentionally not:

- a California-only continuation
- a `FULL` vs `GEO` tuning loop
- a scalar-weight retrofit
- a benchmark-only note without architecture consequences

The implementation target is:

1. a portfolio-level expert factory
2. a per-row, per-view token builder
3. a contextual set-router over variable expert sets
4. a sparse defer-to-`FULL` integrator

## Why This Program Exists

Three facts now anchor the design.

1. The registered portfolio shows real specialist signal, but dense multi-view routing adds competition noise.
2. The current routed wins are mostly conservative `FULL`-anchored integration, not hard expert switching.
3. External review from Claude and Gemini both warned that the present codebase is still experiment-fragmented, and that a contextual router must be defined by a real token protocol rather than by another scalar hedge.

Relevant evidence:

- [2026-03-11-contextual-portfolio-router-spec.md](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/docs/plans/2026-03-11-contextual-portfolio-router-spec.md)
- [2026-03-11-phase0f-registered-portfolio-signal-noise.md](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/docs/plans/2026-03-11-phase0f-registered-portfolio-signal-noise.md)
- [registered_signal_noise_portfolio.md](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/experiments/openml_regression_benchmark/reports_phase0f_registered_portfolio/registered_signal_noise_portfolio.md)
- Claude review: [/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T191334Z-graphdrone-fit-plan/analysis.claude.txt](/home/wliu23/projects/Graph_Drone/.claude-analysis/20260311T191334Z-graphdrone-fit-plan/analysis.claude.txt)
- Gemini review: [/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T191334Z-graphdrone-fit-plan/20260311T193020Z/cleaned_output.txt](/home/wliu23/projects/Graph_Drone/.gemini-cross-checks/20260311T191334Z-graphdrone-fit-plan/20260311T193020Z/cleaned_output.txt)

## Design Question

The current question is not:

- how to make `FULL + GEO` look good again

It is:

- how to build a row-level integration system that can instantiate dataset-specific specialists from a shared ontology, contextualize them per row, and suppress competition noise without collapsing into fixed hedging

## Current Code Reality

The current routed baseline still lives in experiment-specific surfaces:

- [router.py](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/experiments/tabpfn_view_router/src/router.py)
- [data.py](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/experiments/tabpfn_view_router/src/data.py)
- [run_graphdrone_openml.py](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/experiments/openml_regression_benchmark/scripts/run_graphdrone_openml.py)
- [openml_tasks.py](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/experiments/openml_regression_benchmark/src/openml_tasks.py)

That code proves:

- row-level routing exists
- kNN-derived priors are useful
- competition noise is real

That code does not yet provide:

- a unified `GraphDrone.fit()` surface
- a portfolio expert factory with typed view descriptors
- a per-row token builder with embeddings and support summaries
- a contextual set-router over variable expert sets
- a sparse defer-to-`FULL` path that is part of the same model object

## Architecture Contract

### 1. Expert Factory

For dataset `d`, GraphDrone should construct an expert set

\[
\mathcal{E}_d = \{e_{d,1}, \dots, e_{d,m_d}\}
\]

from a shared ontology, not from one-off branch logic.

Required expert families:

- `FULL`
- structural subspace
- local-support
- learned regime
- optional domain-semantic specialist when the dataset schema justifies it

Important constraint:

- expert identity is dataset-specific
- expert family is portfolio-general

### 2. Per-Row Per-View Token Builder

For row `i` and expert `v`, build a token

\[
t_{i,v} = [p_{i,v}, q_{i,v}, s_{i,v}, d_v]
\]

where:

- `p_{i,v}`: predictive fields
  - expert prediction
  - residual to `FULL`
  - disagreement to peer experts
- `q_{i,v}`: prior-quality fields
  - uncertainty
  - density
  - overlap and instability
  - competition-noise indicators
- `s_{i,v}`: support summary
  - pooled support embedding or moments
  - local covariance sketch
  - neighbor label statistics when legal
- `d_v`: learned or embedded view descriptor
  - family id
  - input dimension
  - categorical/numeric composition
  - expert-construction metadata

Constraint:

- no scalar-only shortcut may stand in for the token builder

### 3. Contextual Set-Router

Given the token set and the `FULL` anchor token, learn

\[
h_i = \mathrm{SetRouter}(\{t_{i,v}\}_{v \in \mathcal{E}_d}, t_{i,\mathrm{FULL}})
\]

then produce:

\[
\alpha_{i,v} = \mathrm{sparsemax}(W_\alpha h_i)
\]

\[
\delta_i = \sigma(W_\delta h_i)
\]

Interpretation:

- `\alpha_{i,v}` governs specialist competition
- `\delta_i` governs deferral away from `FULL`

### 4. Sparse Defer-to-FULL Integrator

\[
\hat y_i = (1 - \delta_i)\hat y_{i,\mathrm{FULL}} + \delta_i \sum_v \alpha_{i,v}\hat y_{i,v}
\]

Constraints:

- defer is part of the same model, not a post-hoc attachment
- routing sparsity must be explicit
- fixed-weight averaging is never an acceptable surrogate

## Implementation Package Boundary

The next implementation should stop adding core logic to benchmark runner scripts.

Target package:

- `src/graphdrone_fit/`

Required modules:

- `portfolio_loader.py`
- `expert_factory.py`
- `view_descriptor.py`
- `token_builder.py`
- `support_encoder.py`
- `set_router.py`
- `defer_integrator.py`
- `model.py`
- `config.py`

Bridging modules for the existing benchmark surface:

- `experiments/openml_regression_benchmark/src/graphdrone_fit_adapter.py`
- `experiments/openml_regression_benchmark/scripts/run_graphdrone_fit_openml.py`

## File Responsibilities

### Core package

- `portfolio_loader.py`
  - loads frozen experts, descriptors, and artifact metadata
  - removes hard dependency on benchmark-local paths
  - is the first guard against experiment-debt

- `expert_factory.py`
  - builds dataset-specific expert sets from the portfolio ontology
  - owns no routing logic

- `view_descriptor.py`
  - builds and validates typed expert descriptors
  - owns no prediction logic

- `token_builder.py`
  - converts per-row expert outputs and prior signals into tokens
  - owns no set-routing objective

- `support_encoder.py`
  - turns current kNN/support information into tensor summaries
  - replaces the current reliance on scalar-only quality vectors

- `set_router.py`
  - contextual attention or set-transformer style routing over tokens
  - outputs sparse specialist scores and defer logits

- `defer_integrator.py`
  - applies sparse routing and defer-to-`FULL`
  - owns no expert construction

- `model.py`
  - public `GraphDrone` object with `fit()` and `predict()`
  - orchestrates all submodules

### Benchmark layer

- `graphdrone_fit_adapter.py`
  - maps OpenML split objects and portfolio manifests into the new model API

- `run_graphdrone_fit_openml.py`
  - benchmark runner only
  - no model internals

## Anti-Drift Rules

1. No `FULL` vs `GEO` architecture branch.
   Dataset-level best-pair results may inform expert-family ranking, but may not become the architecture itself.

2. No scalar-only postponement.
   Current `sigma2`, `J`, and `mean_J` style features can be used as token fields, but not as the final routing representation.

3. No fixed-weight surrogate.
   Any improvement claim must beat both `FULL` and a fixed-weight hedge on the registered portfolio.

4. No benchmark-script core logic.
   Core training and inference must live in `src/graphdrone_fit/`, not in `run_*` scripts.

5. No hardcoded dataset assumptions.
   The model must accept variable expert sets and variable descriptors without assuming `GEO` exists.

6. No retrospective leakage.
   Router or defer logic may not rely on test-derived mean weights or post-hoc oracles.

7. No silent fallback to `FULL`.
   Deferral must be explicit, logged, and measurable.

8. No benchmark-path coupling.
   The model package must run from explicit config and loaded descriptors, not from benchmark-script assumptions about manifest locations.

## Phase Program

### Phase I-A: Package Skeleton, Portfolio Loader, And Contracts

Goal:

- create the new package surface without changing algorithmic behavior yet

Deliverables:

- `src/graphdrone_fit/` package skeleton
- typed config objects
- `GraphDrone.fit()` and `GraphDrone.predict()` API
- `portfolio_loader.py`
- expert/view descriptor validation
- standalone smoke path that does not require the registered benchmark matrix

Gate:

- the new package can reproduce the current experiment runner interfaces without embedding benchmark logic in the model
- a non-registry smoke run completes without falling back to benchmark-local hardcoded paths

### Phase I-B: Expert Factory

Goal:

- move expert construction into a reusable portfolio-level factory

Deliverables:

- expert ontology
- dataset-to-expert instantiation rules
- registry-backed specialist configuration

Gate:

- registered datasets can instantiate variable expert sets without dataset-specific if/else logic inside the router

### Phase I-C: Token Builder And Support Encoder

Goal:

- replace scalar routing inputs with tensor/token surfaces

Deliverables:

- token schema implementation
- support encoder output tensors
- descriptor embeddings
- compatibility bridge for legacy scalar priors as token subfields

Gate:

- every routing decision is traceable to a row-level token set

### Phase I-D: Contextual Set-Router

Goal:

- implement the first lightweight contextual router over variable expert sets

Deliverables:

- sparse router
- defer gate
- logged routing diagnostics

Gate:

- must beat a fixed hedge on the portfolio evaluation surface
- must not degrade into dense weight averaging

### Phase I-E: Portfolio Runner And Logging

Goal:

- benchmark the new model against the registered portfolio and current runners

Deliverables:

- new runner script
- leaderboard integration
- diagnostics per dataset
- milestone artifacts in `results/`

Gate:

- full registered-portfolio comparison against existing runners

## Evaluation Surface

Phase I stays portfolio-first.

Regression metrics:

- RMSE
- MAE
- \(R^2\)

Mechanism metrics:

- adaptive vs fixed delta
- defer rate
- routing sparsity
- per-expert token entropy
- competition-noise sensitivity
- view-home capture by expert family

The next classification lane should add:

- ROC-AUC
- PR-AUC
- F1
- calibration

but that is a separate evaluation contract.

## Agent Review Contract

Every implementation phase must pass:

1. local tests
2. file-by-file Gemini review
3. Claude architecture review on the resulting diff and artifacts

Review prompts must explicitly ask:

- whether the implementation slipped back to scalar-only routing
- whether any module boundary is violated
- whether the model silently hardcodes portfolio assumptions
- whether defer-to-`FULL` is a real integrator or a shortcut

## Git And Worktree Discipline

- keep one worktree per implementation phase
- keep one docs/milestone artifact per phase under `results/`
- commit package skeleton before algorithmic internals
- do not mix benchmark artifact noise into code commits
- refresh tree and repo snapshots before each phase closeout

Current planning snapshots:

- [git_tree_snapshot.md](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/output/repo_tree/git_tree_snapshot.md)
- [git_snapshot.json](/home/wliu23/projects/Graph_Drone/.worktrees/graphdrone-fit-impl-plan/output/repo/git_snapshot.json)

## Immediate Next Move

Start Phase I-A only:

- create the package skeleton
- define `GraphDrone.fit()` config and object contracts
- move zero benchmark logic into the model package

Do not yet:

- retune California
- add dynamic kNN branches
- optimize a dataset-specific specialist pair
- insert more scalar routing heuristics
