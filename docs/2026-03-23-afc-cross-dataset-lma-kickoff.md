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

## Similar-task transfer framing

Why the earlier held-out check used MSE:
- that run was not scoring classification quality
- it was scoring reconstruction of continuous task-context features
- so `MSE` was only a surrogate for representation transfer, not a claim that classification itself should be judged by MSE

That surrogate is still limited for the research question you raised.

The better question is:
- does a held-out dataset map coherently toward one or a few seen task prototypes?
- and is that mapping stable enough to support “apply the prior to a similar next task”?

## Similarity-aware held-out result

Similarity-mode run:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v1 \
  --encoder both \
  --mode leave_one_dataset_out_similarity \
  --epochs 150 \
  --output-dir eval/afc_task_conditioned_lma_similarity_cls_v1
```

Artifact:
- `eval/afc_task_conditioned_lma_similarity_cls_v1/summary.json`

Result:
- both encoders map each held-out dataset to exactly one seen dataset across all 24 bootstraps
- transformer:
  - `credit_g -> pendigits`
  - `diabetes -> pendigits`
  - `optdigits -> credit_g`
  - `pendigits -> credit_g`
- GRU:
  - same neighbor pattern

Read:
- this is not yet a healthy notion of “similar-task transfer”
- the prior is stable, but too collapsed
- it does not discover a nuanced neighborhood structure
- instead it acts like a hard prototype assignment with near-total confidence

Interpretation:
- your framing is correct: the right goal is not “one prior for all tasks”
- it is “learn task characteristics, then transfer to the next similar task”
- but the current representation still collapses that similarity relation too aggressively

So the next bottleneck is now clearer:
- we need a better similarity space before worrying about more encoder capacity

Next likely fixes:
- normalize task-context features so token magnitude and descriptor scale do not dominate
- add contrastive or metric-learning structure, not only reconstruction
- keep the task-conditioned prior, but move from hard prototype collapse toward softer task neighborhoods

## Expanded classification bank with normalized neighborhoods

Expanded bank:

```bash
PYTHONPATH=src python scripts/analyze_cross_dataset_view_tokens.py \
  --datasets segment mfeat_factors pendigits optdigits diabetes credit_g \
  --max-samples 384 \
  --sample-rows 48 \
  --bootstrap-summaries 16 \
  --output-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2
```

Normalized similarity-aware transfer:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --encoder both \
  --mode leave_one_dataset_out_similarity \
  --normalize-features \
  --epochs 150 \
  --output-dir eval/afc_task_conditioned_lma_similarity_cls_v2_norm
```

What improved:
- the prior no longer collapses into a single deterministic prototype for every held-out dataset
- the summary now exposes soft neighborhoods, for example:
  - `credit_g` is mostly near `diabetes`, with meaningful mass on `segment` and `pendigits`
  - `optdigits` is mostly near `pendigits`, but still keeps some mass on `diabetes` and `credit_g`
  - `pendigits` sits between `segment` and `optdigits`

Transformer examples:
- `credit_g`
  - top-neighbor vote fraction: `0.6875`
  - soft neighborhood: `diabetes (0.300)`, `segment (0.261)`, `pendigits (0.177)`
- `optdigits`
  - top-neighbor vote fraction: `0.8125`
  - soft neighborhood: `pendigits (0.255)`, `diabetes (0.238)`, `credit_g (0.180)`
- `pendigits`
  - top-neighbor vote fraction: `0.5000`
  - soft neighborhood: `segment (0.259)`, `optdigits (0.251)`, `diabetes (0.227)`

Interpretation:
- this is much closer to the intended “apple, banana in the fruit family” behavior
- the task-conditioned prior now expresses local neighborhoods instead of pure one-prototype collapse
- normalization mattered
- the classification-first bank is large enough to show a meaningful similarity structure

Current read on encoder choice:
- transformer and GRU now tell a broadly similar neighborhood story
- transformer remains the default baseline because it fit the context-learning tasks better overall
- but the bigger win here came from representation normalization and a larger task bank, not from encoder family alone

This is the first point where the task-conditioned LMA direction starts to look structurally useful rather than merely learnable.

## Persistent prototype bank and reuse

To avoid relearning the same dataset from scratch, this branch now has a persistent task-prototype bank layer:
- save normalized task prototypes
- save encoder state
- query for either:
  - exact dataset reuse when the same dataset is seen again
  - nearest learned task neighborhoods when the dataset is new

New support lives in:
- `src/graphdrone_fit/task_conditioned_prior.py`
- `scripts/prototype_task_conditioned_lma.py`

Exact-reuse fit on the 6-dataset normalized classification bank:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --mode fit_prototype_bank \
  --encoder both \
  --normalize-features \
  --epochs 100 \
  --output-dir eval/afc_task_prototype_bank_cls_v1
```

Artifacts:
- `eval/afc_task_prototype_bank_cls_v1/transformer_prototype_bank.json`
- `eval/afc_task_prototype_bank_cls_v1/transformer_encoder_state.pt`
- `eval/afc_task_prototype_bank_cls_v1/gru_prototype_bank.json`
- `eval/afc_task_prototype_bank_cls_v1/gru_encoder_state.pt`

Querying known datasets back against that bank:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --mode query_prototype_bank \
  --encoder both \
  --bank-dir eval/afc_task_prototype_bank_cls_v1 \
  --query-datasets credit_g diabetes optdigits pendigits \
  --output-dir eval/afc_task_prototype_bank_query_cls_v1
```

Read:
- all queried datasets returned `exact_reuse_available=true`
- transformer exact match similarities were high:
  - `credit_g`: `0.9256`
  - `diabetes`: `0.9954`
  - `optdigits`: `0.9744`
  - `pendigits`: `0.9839`
- the next-similar neighborhoods remained visible after removing the self-match:
  - `credit_g -> diabetes, segment`
  - `diabetes -> credit_g, segment`
  - `optdigits -> pendigits, mfeat_factors`
  - `pendigits -> segment, optdigits`

This is the first branch-local result that directly supports the requirement:
- if we see the same dataset again, we should reuse prior task learning
- if we see a new but related dataset, we should attach it to a learned neighborhood instead of starting from zero

## Unseen-task neighborhood lookup

To test the second half directly, a 5-dataset transformer bank was fit without `segment`, then `segment` was queried as unseen:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2_minus_segment \
  --mode fit_prototype_bank \
  --encoder transformer \
  --normalize-features \
  --epochs 100 \
  --output-dir eval/afc_task_prototype_bank_cls_minus_segment_v1

PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --mode query_prototype_bank \
  --encoder transformer \
  --bank-dir eval/afc_task_prototype_bank_cls_minus_segment_v1 \
  --query-datasets segment \
  --output-dir eval/afc_task_prototype_bank_query_segment_v1
```

Observed `segment` neighbors:
- `pendigits`: probability `0.2735`, mean similarity `0.7553`
- `diabetes`: probability `0.2459`, mean similarity `0.6491`
- `credit_g`: probability `0.2136`, mean similarity `0.5071`

Interpretation:
- the bank now supports exact dataset memory and unseen-task retrieval in the same interface
- the unseen-task retrieval is still approximate rather than semantically clean
- but this is already closer to a reusable task-family system than the earlier collapsed one-prototype behavior

Current recommendation:
- keep building the classification-first prototype bank
- add a contrastive or metric-learning objective next so neighborhoods sharpen without collapsing
- use the persistent bank as the memory substrate for any later hyper-router or cross-dataset LMA prior

## Contrastive sharpening on top of the prototype bank

The next step tested whether a supervised contrastive objective can sharpen the task neighborhood space beyond the reconstruction-trained bank.

New support:
- `supervised_contrastive_loss` in `src/graphdrone_fit/task_conditioned_prior.py`
- `fit_contrastive_prototype_bank` mode in `scripts/prototype_task_conditioned_lma.py`

Training contract:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --mode fit_contrastive_prototype_bank \
  --encoder transformer \
  --normalize-features \
  --epochs 120 \
  --contrastive-temperature 0.1 \
  --reconstruction-weight 0.25 \
  --output-dir eval/afc_task_prototype_bank_cls_contrastive_v1
```

Observed training tail:
- positive similarity: about `0.96`
- negative similarity: about `-0.14`

Interpretation:
- same-dataset task embeddings are now tightly clustered
- different datasets are pushed apart rather than merely separated by softer reconstruction geometry

Known-dataset query against the contrastive bank:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --mode query_prototype_bank \
  --encoder transformer \
  --bank-dir eval/afc_task_prototype_bank_cls_contrastive_v1 \
  --query-datasets credit_g diabetes optdigits pendigits \
  --output-dir eval/afc_task_prototype_bank_query_cls_contrastive_v1
```

Contrastive exact-match similarities:
- `credit_g`: `0.9987`
- `diabetes`: `0.9972`
- `optdigits`: `0.9986`
- `pendigits`: `0.9971`

This is materially sharper than the reconstruction-trained bank, where exact matches were already high but still in the `0.93-0.99` range.

Unseen `segment` query against a contrastive bank trained without `segment`:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2_minus_segment \
  --mode fit_contrastive_prototype_bank \
  --encoder transformer \
  --normalize-features \
  --epochs 120 \
  --contrastive-temperature 0.1 \
  --reconstruction-weight 0.25 \
  --output-dir eval/afc_task_prototype_bank_cls_minus_segment_contrastive_v1

PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --mode query_prototype_bank \
  --encoder transformer \
  --bank-dir eval/afc_task_prototype_bank_cls_minus_segment_contrastive_v1 \
  --query-datasets segment \
  --output-dir eval/afc_task_prototype_bank_query_segment_contrastive_v1
```

Observed `segment` neighbors under contrastive training:
- `credit_g`: probability `0.3158`, mean similarity `0.6040`
- `pendigits`: probability `0.2531`, mean similarity `0.3790`
- `diabetes`: probability `0.2155`, mean similarity `0.2251`
- soft-neighbor entropy: `1.5166`

Comparison to the reconstruction-trained unseen query:
- prior top neighbor was `pendigits`
- contrastive top neighbor became `credit_g`
- entropy dropped from `1.5667` to `1.5166`

Interpretation:
- contrastive training does sharpen the bank
- it improves exact reuse confidence and reduces unseen-task entropy
- but it does **not** yet guarantee a semantically cleaner task family structure
- in other words, the bank is becoming more decisive faster than it is becoming more correct

Current recommendation:
- keep the contrastive objective as the next default bank-sharpening baseline
- do not yet claim that the learned task neighborhoods are semantically aligned
- next work should add explicit neighborhood supervision or taxonomy-aware metric structure before using this bank to drive a live hyper-router

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
