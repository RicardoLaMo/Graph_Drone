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

## Prototype: task-context encoder baselines

New prototype modules:
- `src/graphdrone_fit/task_conditioned_prior.py`
- `scripts/prototype_task_conditioned_lma.py`

These do not modify the live GraphDrone path yet.
They provide an offline prior-learning scaffold:
- input: dataset-level view-summary sequences from `task_context_examples.csv`
- encoders:
  - transformer
  - GRU
- output: a task embedding and a simple dataset-classification head

Bootstrap classification extractor:

```bash
PYTHONPATH=src python scripts/analyze_cross_dataset_view_tokens.py \
  --datasets pendigits diabetes credit_g optdigits \
  --max-samples 384 \
  --sample-rows 48 \
  --bootstrap-summaries 24 \
  --output-dir eval/afc_cross_dataset_lma_classification_bootstrap_v1
```

Prototype run:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v1 \
  --encoder both \
  --epochs 120 \
  --output-dir eval/afc_task_conditioned_lma_proto_cls_v1
```

Result:
- examples: `96`
- sequence length: `4`
- input dimension: `28`
- transformer: loss `0.00399`, accuracy `1.00`
- GRU: loss `0.01139`, accuracy `1.00`

Interpretation:
- the task-conditioned prior is learnable on the classification-first bootstrap context surface
- both encoders can fit the current small prototype task
- the transformer converged a bit better than the GRU on the same artifact

Important limit:
- this is only a learnability/prototype check
- it does **not** yet show held-out generalization across unseen datasets
- it does **not** yet show end-to-end GraphDrone improvement

Still, it is enough to justify the next research step:
- keep the classification-first task-conditioned direction
- prefer transformer as the default baseline
- move next to a stricter generalization test, not immediate integration

## Held-out generalization check

Held-out prototype run:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v1 \
  --encoder both \
  --mode leave_one_dataset_out_reconstruction \
  --epochs 150 \
  --output-dir eval/afc_task_conditioned_lma_generalization_cls_v1
```

Result summary:
- transformer:
  - mean held-out test MSE: `46.61`
  - mean generalization gap: `45.17`
- GRU:
  - mean held-out test MSE: `40.31`
  - mean generalization gap: `39.42`

Per-dataset read:
- both encoders generalize reasonably on `credit_g` and `diabetes`
- both are acceptable on `pendigits`
- both fail badly on `optdigits`

Interpretation:
- the task-conditioned prior is learnable, but cross-dataset generalization is still fragile
- the current bottleneck is not just encoder class
- dataset heterogeneity inside the classification regime is still large enough to break the prior on some held-out tasks

This changes the architectural read in an important way:
- transformer remains the default baseline for task-conditioned prior learning because it fit the in-sample context task better
- but the held-out result does **not** justify claiming transformer is already the better generalizing prior
- GRU actually had slightly lower held-out reconstruction error in this first check
- the real problem appears to be representation mismatch, especially around `optdigits`, not simply model capacity

So the next question is not “transformer or GRU?”
It is:
- what normalization or representation change makes held-out datasets look less alien to the shared prior?

Current best next step:
- inspect why `optdigits` is a strong outlier
- normalize or reparameterize the task-context features before another encoder comparison
- only then revisit encoder choice

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
