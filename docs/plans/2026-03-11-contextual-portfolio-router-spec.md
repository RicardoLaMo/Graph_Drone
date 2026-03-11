# Contextual Portfolio Router Spec

## Goal

Generalize GraphDrone from:

- dataset-specific expert sets
- scalar-quality row routing
- post-hoc dense mixing

to:

- a portfolio-level expert factory
- row-level contextual routing over view tokens
- sparse integration around `FULL`, not hardcoded `FULL + GEO`

This is a design memo only. No model code is changed in this branch.

## Why California Was Only A Probe

California was the cleanest early mechanism case, not the target architecture.

The portfolio result in [registered_signal_noise_portfolio.md](/home/wliu23/projects/Graph_Drone/.worktrees/portfolio-contextual-router-spec/experiments/openml_regression_benchmark/reports_phase0f_registered_portfolio/registered_signal_noise_portfolio.md) is the real design anchor:

- `9` registered datasets
- `6/9` useful signal obscured by competition
- `3/9` competition noise plus weak expert
- best-view counts by dataset:
  - `GEO = 6`
  - `LOWRANK = 2`
  - `DOMAIN = 1`

So the design question is no longer “should we hardcode GEO?”

It is:

- how do we let each dataset instantiate its own candidate experts
- while keeping a single row-level integration mechanism

## Current Code Constraints

Current routing surface:

- [router.py](/home/wliu23/projects/Graph_Drone/.worktrees/portfolio-contextual-router-spec/experiments/tabpfn_view_router/src/router.py)
  - `alpha_i = softmax(MLP(q_i))`
  - learned row routing is real, but `q_i` is only a scalar feature vector

Current prior surface:

- [data.py](/home/wliu23/projects/Graph_Drone/.worktrees/portfolio-contextual-router-spec/experiments/tabpfn_view_router/src/data.py)
  - per-view `sigma2`
  - pairwise kNN-overlap `J_flat`
  - `mean_J`

Current portfolio view surface:

- [openml_tasks.py](/home/wliu23/projects/Graph_Drone/.worktrees/portfolio-contextual-router-spec/experiments/openml_regression_benchmark/src/openml_tasks.py)
  - dataset-specific view declarations
  - California: `FULL/GEO/SOCIO/LOWRANK`
  - others: `FULL/GEO/DOMAIN/LOWRANK`

This means the present system is row-conditional but not yet contextual over a set of view tokens or view embeddings.

## Core Design Principle

GraphDrone should be a **row-level contextual integrator over a variable expert set**, not a family of dataset-specific hardcoded blends.

The generalized design should separate three concerns:

1. `Expert factory`
2. `Prior/token builder`
3. `Contextual router`

## Proposed Generalized Architecture

### 1. Expert Factory

Each dataset should instantiate experts from a shared ontology instead of hand-built one-off views.

Proposed expert types:

- `FULL`
- structural subspace expert
  - PCA / low-rank / residual subspace
- local-support expert
  - neighborhood-conditioned support statistics
- domain or semantic subspace expert
  - only when a stable domain slice exists
- learned regime expert
  - cluster- or embedding-defined regime

Important rule:

- `GEO` is a successful current specialist, but only one member of a broader specialist class
- no architecture should assume every dataset has a meaningful `GEO`

### 2. Per-Row Per-View Token

For row `i` and expert/view `v`, construct:

\[
t_{i,v} = \left[
\hat y_{i,v},
u_{i,v},
s_{i,v},
d_v
\right]
\]

where:

- `\hat y_{i,v}`:
  view prediction statistics
  - raw prediction
  - disagreement to `FULL`
  - disagreement to portfolio mean
- `u_{i,v}`:
  prior-quality features
  - `sigma2_v`
  - density / uncertainty
  - overlap with other views
  - support instability
- `s_{i,v}`:
  support summary
  - pooled kNN embedding
  - local label moments
  - neighborhood covariance sketch
- `d_v`:
  view descriptor
  - type id
  - input dimension
  - construction metadata
  - expert family flags

This is the critical generalization step. It turns “router over a fixed scalar vector” into “router over a set of typed view tokens.”

### 3. Contextual Router

Use a set-based contextual model over the expert tokens plus a `FULL` anchor token.

Minimal formulation:

\[
h_i^{\text{ctx}} = \mathrm{SetRouter}\left(\{t_{i,v}\}_{v \in \mathcal{V}_d},\ t_{i,\text{FULL}}\right)
\]

\[
\alpha_{i,v} = \mathrm{sparsemax}(W_\alpha h_i^{\text{ctx}})
\]

\[
\delta_i = \sigma(W_\delta h_i^{\text{ctx}})
\]

\[
\hat y_i = (1-\delta_i)\hat y_{i,\text{FULL}} + \delta_i \sum_{v \in \mathcal{V}_d} \alpha_{i,v} \hat y_{i,v}
\]

Interpretation:

- `\alpha_{i,v}` chooses among non-`FULL` specialists
- `\delta_i` decides how much to defer away from `FULL`

This is portfolio-general:

- variable number of experts
- dataset-specific expert identities
- row-specific routing
- no fixed weights per dataset

## Why Not Jump Straight To A Large Transformer

Two constraints from the portfolio still matter:

1. some datasets have weak specialists
2. dense competition is itself a failure mode

So the first contextual step should be a **small set-router** or lightweight attention block, not a large transformer stack.

The real bottleneck may still be expert quality on loss cases like:

- `miami_housing`
- `diamonds`
- `wine_quality`

That means a bigger router alone is not enough.

## Portfolio-Level Research Questions

Before implementation, the portfolio-level questions are:

1. Which expert types survive the signal-vs-noise test across datasets?
2. Which per-row priors actually change routing usefully?
3. Are support-set embeddings better than scalar kNN summaries?
4. Does contextual routing improve over a sparse `FULL + specialist` baseline?

## Implementation Phases

### Phase A: Shared Expert Ontology

- define expert families independent of dataset names
- attach view descriptors to every expert
- keep the benchmark surface portfolio-wide

### Phase B: Token Builder

- convert current scalar priors into token fields
- add support-summary embeddings
- preserve compatibility with current saved diagnostics

### Phase C: Lightweight Contextual Router

- set-attention or small transformer over expert tokens
- explicit defer-to-`FULL` gate
- sparse top-k or sparsemax routing

### Phase D: Portfolio Evaluation

Regression portfolio:

- RMSE
- MAE
- \(R^2\)

Future classification portfolio:

- ROC-AUC
- PR-AUC
- F1
- calibration

## Decision

The next GraphDrone direction should be:

- portfolio-general
- row-level
- contextual over view tokens
- sparse around `FULL`

but not:

- California-specific
- GEO-specific
- a fixed hedge per dataset
- a larger dense router over the same weak expert family
