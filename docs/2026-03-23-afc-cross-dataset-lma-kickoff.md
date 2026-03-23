# AFC Cross-Dataset LMA Kickoff

Date: 2026-03-23

Branch lineage:
- branch: `exp/afc-cross-dataset-lma`
- base commit: `8fa32a6`

## Why this branch exists

The current Phase B research line established three things:

1. Rotor alignment is a real local mechanism.
2. Several early negative results were caused by circuit issues and instability, not by obviously bad AFC math.
3. Even after those confounds were reduced, local routing objectives still fail to convert specialist signal into a stable held-out win.

The strongest recent result is that the residual-usefulness-gap penalty was active but falsified as a local fix:
- it stayed slightly worse than the champion for all tested `lambda`
- it mainly changed routing by suppressing defer
- it did not learn a clearly better specialist-allocation policy

That makes the next-scale question legitimate:

> Is single-dataset router fitting fundamentally too weak, such that the next AFC line should learn a cross-dataset routing prior from view-token geometry?

## First research contract

This branch does **not** start by training a new hyper-router.

It starts with offline geometry analysis:
- extract GraphDrone view tokens from multiple datasets
- preserve descriptor metadata for each expert/view
- compare token geometry across datasets
- test whether view families and anchor/specialist relations form reusable latent structure

If that structure is not present, a cross-dataset LMA prior is probably premature.
If that structure is present, then a hyper-router or shared routing prior becomes a grounded next step rather than a speculative rewrite.

## Initial artifact

New script:
- `scripts/analyze_cross_dataset_view_tokens.py`

Purpose:
- fit the current GraphDrone on selected datasets
- extract per-view token summaries using the existing token builder
- export descriptor inventory, per-view token statistics, and pairwise cross-dataset similarities

## Initial smoke result

Diagnostic smoke:

```bash
PYTHONPATH=src python scripts/analyze_cross_dataset_view_tokens.py \
  --datasets california cpu_act \
  --max-samples 256 \
  --sample-rows 32 \
  --output-dir eval/afc_cross_dataset_lma_smoke
```

Artifacts:
- `eval/afc_cross_dataset_lma_smoke/descriptor_inventory.csv`
- `eval/afc_cross_dataset_lma_smoke/token_summary.csv`
- `eval/afc_cross_dataset_lma_smoke/pairwise_similarity.csv`
- `eval/afc_cross_dataset_lma_smoke/summary.json`

Observed summary on this tiny regression-only slice:
- `mean_cosine_same_family = 0.6773`
- `mean_cosine_cross_family = 0.7186`
- `mean_cosine_anchor_anchor = 0.4648`
- `mean_cosine_subspace_subspace = 0.6914`

Read:
- the extractor works
- but this tiny smoke does **not** yet show clean family-level separation
- that is useful, because it means the next step should control for trivial descriptor and dataset effects before assuming a shared manifold is already obvious

## Mixed-slice controlled result

Controlled mixed slice:

```bash
PYTHONPATH=src python scripts/analyze_cross_dataset_view_tokens.py \
  --datasets california cpu_act pendigits diabetes \
  --max-samples 384 \
  --sample-rows 48 \
  --output-dir eval/afc_cross_dataset_lma_mixed_slice_v1
```

Artifacts:
- `eval/afc_cross_dataset_lma_mixed_slice_v1/summary.json`
- `eval/afc_cross_dataset_lma_mixed_slice_v1/grouped_similarity_summary.csv`
- `eval/afc_cross_dataset_lma_mixed_slice_v1/pairwise_similarity.csv`

Headline summary:
- `mean_cosine_same_family = 0.7725`
- `mean_cosine_cross_family = 0.7777`
- `mean_cosine_same_task_type = 0.8216`
- `mean_cosine_cross_task_type = 0.7335`

Grouped read:
- same-task-type pairs are clearly more similar than cross-task-type pairs
- same-family pairs do **not** globally beat cross-family pairs
- cross-dataset same-family/same-anchor pairs still retain some structure:
  - same task type: `0.7124`
  - cross task type: `0.7399`

Interpretation:
- there is nontrivial cross-dataset token geometry
- but the dominant axis in this first controlled slice is task type, not family
- this weakens the naïve version of the hypothesis that a single family-level prior is already obvious
- it strengthens a narrower design direction:
  - any future LMA prior should likely be hierarchical or conditioned by task type
  - only after that should it try to learn finer family-level alignment

So the current branch should not jump straight to a single universal hyper-router.
The more defensible next step is:
- regression-only and classification-only token-bank analysis
- then test whether family-level structure becomes cleaner within each task regime

## Within-task split result

Regression-only slice:

```bash
PYTHONPATH=src python scripts/analyze_cross_dataset_view_tokens.py \
  --datasets california cpu_act diamonds kin8nm \
  --max-samples 384 \
  --sample-rows 48 \
  --output-dir eval/afc_cross_dataset_lma_regression_v1
```

Classification-only slice:

```bash
PYTHONPATH=src python scripts/analyze_cross_dataset_view_tokens.py \
  --datasets pendigits diabetes credit_g optdigits \
  --max-samples 384 \
  --sample-rows 48 \
  --output-dir eval/afc_cross_dataset_lma_classification_v1
```

Regression summary:
- `mean_cosine_same_family = 0.5495`
- `mean_cosine_cross_family = 0.5880`
- cross-dataset same-family/same-anchor mean cosine: `0.4631`

Classification summary:
- `mean_cosine_same_family = 0.8855`
- `mean_cosine_cross_family = 0.8914`
- cross-dataset same-family/same-anchor mean cosine: `0.8628`

Read:
- classification tokens exhibit much tighter cross-dataset geometry than regression tokens
- neither regime yet shows a simple “same family beats cross family” rule
- but the classification regime is far more coherent overall than regression

Interpretation:
- task type is not just a nuisance covariate; it appears to be a primary organizing axis for any shared prior
- a single universal prior over all tasks is weaker than a task-conditioned design
- classification is the better first target for a learned cross-dataset LMA prior
- regression likely needs either a different summary representation or stronger normalization before shared alignment will make sense

## Design implication: hierarchical or task-conditioned prior

The current evidence supports a two-stage design more than a flat hyper-router.

Proposed direction:

1. Task-context encoder.
- Input: a dataset-level sequence of view-summary tokens
- Each token should combine:
  - descriptor fields (`family`, `is_anchor`, `projection_kind`, `input_dim`, `preferred_k`)
  - token-bank summary statistics
  - simple router/usefulness statistics where available
- Model options:
  - small transformer encoder over the view-token sequence
  - or a compact GRU over a canonical expert order (`FULL`, `SUB0`, `SUB1`, `SUB2`, ...)

2. Task-conditioned prior head.
- Output: a task embedding `z_task`
- Use `z_task` to condition:
  - rotor/alignment parameters
  - router initialization
  - specialist validity priors
  - or attention bias terms over view families

3. Family-level alignment inside task regime.
- Only after conditioning on task type or task embedding should the model try to learn finer family-level alignment

Why this is better than a flat prior:
- regression and classification do not currently occupy the same token geometry
- forcing one universal prior would likely mix incompatible manifolds
- a hierarchical prior can first separate the broad regime, then share structure where it is actually reusable

Current recommendation:
- start with classification-first task-conditioned LMA
- use a small transformer encoder if permutation robustness over view order matters
- use a GRU only if we intentionally define and trust a canonical expert sequence and want a lighter recurrent prior

## Initial analysis questions

1. Do anchor views from different datasets cluster more tightly than random view pairs?
2. Do `structural_subspace` views show stable cross-dataset geometry relative to anchors?
3. Are there family-level statistics that are invariant enough to support a shared routing prior?
4. Does regression vs classification produce distinct token manifolds, or is there a mixed latent structure that a shared transformer could exploit?

## Minimum success criterion for the offline phase

This offline phase is useful only if it can show at least one of:
- family-level token similarity is measurably stronger than arbitrary cross-dataset pairs
- anchor-to-specialist geometry has recurring structure across datasets
- descriptor and token statistics reveal a natural basis for cross-dataset conditioning

If none of that appears, the branch should not escalate directly into a hyper-router implementation.

## Next checks

1. Run the token-bank extractor on a small mix of regression and classification datasets.
2. Inspect whether similarity structure is driven only by trivial descriptors such as `input_dim` or `is_anchor`.
3. If the signal survives that control, design a minimal shared prior over view families before any end-to-end hyper-router training.
