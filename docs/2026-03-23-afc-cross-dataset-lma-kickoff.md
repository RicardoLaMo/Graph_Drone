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

## Metadata-driven neighborhood consistency

To avoid hardcoded task families, the next step added a metadata-driven neighborhood-consistency loss.

Important constraint:
- this does **not** use dataset-name rules
- neighborhood targets are derived algorithmically from each dataset's own task-context signatures
- the fixed dataset names seen in tests and notes are only experiment fixtures and reporting examples, not production routing logic

New support:
- `metadata_neighbor_targets` in `src/graphdrone_fit/task_conditioned_prior.py`
- `neighborhood_consistency_loss` in `src/graphdrone_fit/task_conditioned_prior.py`
- `fit_taxonomy_consistent_prototype_bank` in `scripts/prototype_task_conditioned_lma.py`

The target neighborhood matrix is built from the extracted task-context sequences:
- aggregate each dataset into a dataset signature
- compute dataset-to-dataset cosine similarities
- convert that into a soft neighborhood target distribution
- train the learned embedding neighborhoods to match that target while still keeping contrastive and reconstruction terms

Training contract:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --mode fit_taxonomy_consistent_prototype_bank \
  --encoder transformer \
  --normalize-features \
  --epochs 120 \
  --contrastive-temperature 0.1 \
  --reconstruction-weight 0.25 \
  --neighbor-weight 0.5 \
  --metadata-temperature 0.2 \
  --embedding-neighbor-temperature 0.1 \
  --output-dir eval/afc_task_prototype_bank_cls_taxonomy_v1
```

Observed training tail:
- neighbor loss fell to about `0.0042`
- positive similarity stayed high at about `0.947`
- negative similarity stayed positive but controlled at about `0.123`

Read:
- the learned bank is now matching the metadata-derived neighborhood targets closely
- this is less aggressive than contrastive-only separation
- but it is more grounded in derived task characteristics

Known-dataset query under metadata-guided training:
- exact matches remain strong, around `0.994-0.997`
- neighborhood entropy is higher than contrastive-only, roughly `1.71-1.73`

Interpretation:
- this objective gives up some confidence to preserve more neighbor structure
- that is expected if the goal is semantically useful transfer rather than maximum self-separation

Unseen `segment` query with metadata guidance:

```bash
PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2_minus_segment \
  --mode fit_taxonomy_consistent_prototype_bank \
  --encoder transformer \
  --normalize-features \
  --epochs 120 \
  --contrastive-temperature 0.1 \
  --reconstruction-weight 0.25 \
  --neighbor-weight 0.5 \
  --metadata-temperature 0.2 \
  --embedding-neighbor-temperature 0.1 \
  --output-dir eval/afc_task_prototype_bank_cls_minus_segment_taxonomy_v1

PYTHONPATH=src python scripts/prototype_task_conditioned_lma.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --mode query_prototype_bank \
  --encoder transformer \
  --bank-dir eval/afc_task_prototype_bank_cls_minus_segment_taxonomy_v1 \
  --query-datasets segment \
  --output-dir eval/afc_task_prototype_bank_query_segment_taxonomy_v1
```

Observed `segment` neighbors:
- `pendigits`: probability `0.3060`, mean similarity `0.8082`
- `credit_g`: probability `0.2287`, mean similarity `0.5162`
- `diabetes`: probability `0.2014`, mean similarity `0.3903`
- soft-neighbor entropy: `1.5561`

Comparison across the three bank variants for unseen `segment`:
- reconstruction bank:
  - top neighbor `pendigits`
  - entropy `1.5667`
- contrastive bank:
  - top neighbor `credit_g`
  - entropy `1.5166`
- metadata-guided bank:
  - top neighbor `pendigits`
  - entropy `1.5561`

Interpretation:
- metadata guidance recovered a more plausible top neighbor while staying sharper than the reconstruction-only bank
- this is the best tradeoff so far between decisiveness and semantic plausibility
- the bank is still not a true taxonomy, but it now behaves more like a learned task-family prior and less like arbitrary confident separation

Current recommendation:
- use metadata-guided neighborhood consistency as the leading task-bank objective
- keep the contrastive term, but no longer by itself
- next work should expand the dataset bank and test whether this objective remains stable when more near-neighbor datasets are added

## First live classification router-conditioning prototype

After the metadata-guided bank became the best offline prior, the next step tried a minimal live integration:
- classification only
- dataset-level prior vector only
- inject the prior into the learned router anchor token
- keep the rest of GraphDrone unchanged

New support:
- `TaskConditionedRouter` in `src/graphdrone_fit/set_router.py`
- task-prior bank loading and query helpers in `src/graphdrone_fit/task_conditioned_prior.py`
- classification router attachment in `src/graphdrone_fit/model.py`
- env/preset plumbing in `src/graphdrone_fit/presets.py`

Important wiring bug found and fixed:
- in binary classification, `_classification_router_config()` used to replace
  `bootstrap_full_only` with a fresh `SetRouterConfig(kind="noise_gate_router")`
- that silently dropped all task-prior fields
- this is now fixed by preserving the original config and only replacing the kind

Unit coverage:
- `tests/test_model_task_prior_config.py`

### Live six-dataset classification contract

```bash
GRAPHDRONE_TASK_PRIOR_BANK_DIR=eval/afc_task_prototype_bank_cls_taxonomy_v1 \
GRAPHDRONE_TASK_PRIOR_ENCODER_KIND=transformer \
GRAPHDRONE_TASK_PRIOR_STRENGTH=0.5 \
PYTHONPATH=src python scripts/run_champion_challenger.py \
  --task classification \
  --datasets credit_g diabetes mfeat_factors optdigits pendigits segment \
  --folds 0 \
  --max-samples 384 \
  --champion-version champion-cls-no-prior \
  --challenger-version challenger-cls-task-prior-v1 \
  --output-dir eval/afc_live_task_prior_cls_v1
```

Result:
- exact tie vs champion
- challenger granular report showed `router_kind=geo_poe` for all six datasets
- no `task_prior_*` diagnostics appeared

Interpretation:
- the task prior code did not fail silently
- the benchmark contract never provided a live learned-router surface for it to act on
- multiclass datasets still use static GeoPOE by design
- the binary datasets did not expose a usable learned router on that contract

### Live binary-only rerun after the config fix

```bash
GRAPHDRONE_TASK_PRIOR_BANK_DIR=eval/afc_task_prototype_bank_cls_taxonomy_v1 \
GRAPHDRONE_TASK_PRIOR_ENCODER_KIND=transformer \
GRAPHDRONE_TASK_PRIOR_STRENGTH=0.5 \
PYTHONPATH=src python scripts/run_champion_challenger.py \
  --task classification \
  --datasets credit_g diabetes \
  --folds 0 \
  --max-samples 384 \
  --champion-version champion-bin-no-prior \
  --challenger-version challenger-bin-task-prior-v2 \
  --output-dir eval/afc_live_task_prior_binary_v2
```

Result:
- exact tie again
- both binary datasets still printed:
  - `Classification router skipped: anchor-only portfolio leaves nothing to route`

Interpretation:
- after the config-preservation fix, the remaining blocker is structural
- the OOF binary portfolio still collapses to anchor-only before any learned routing stage exists
- so the task-prior mechanism is now ready, but the current benchmark contract still does not expose a place for it to act

Current conclusion:
- this branch has crossed from “offline prior only” into “live conditioning prototype”
- the task-prior mechanism is now wired correctly
- but GraphDrone’s current classification architecture does not yet provide a stable activation surface for it
- the next scale question is no longer “does the prior bank exist?”
- it is “where should a real cross-dataset hyper-router live, given that multiclass is static and binary OOF routing can collapse before conditioning?”

Current recommendation:
- do not spend more time trying to prove the prior through the current static multiclass path
- next work should build a small explicit hyper-router prototype on a classification subproblem where routing is guaranteed to exist
- alternatively, redesign the binary OOF classification portfolio so the learned router stage reliably survives

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

## Feedback-updated task bank (first closed loop)

This branch now has a first explicit outcome-feedback loop for the task bank.

New pieces:
- `scripts/update_task_prototype_bank_feedback.py`
- feedback-aware bank state inside `src/graphdrone_fit/task_conditioned_prior.py`
- live diagnostics for:
  - `task_prior_base_top_neighbor`
  - `task_prior_feedback_used`
  - `task_prior_feedback_top_source`

Contract:
- start from the metadata-guided classification bank
- run a live binary champion/challenger task-prior benchmark
- ingest `paired_task_deltas.csv`
- update the bank with outcome-weighted neighbor feedback
- rerun the same live contract with the new bank

First update artifact:

```bash
PYTHONPATH=src python scripts/update_task_prototype_bank_feedback.py \
  --analysis-dir eval/afc_cross_dataset_lma_classification_bootstrap_v2 \
  --bank-dir eval/afc_task_prototype_bank_cls_taxonomy_v2 \
  --comparison-csv eval/afc_live_task_prior_binary_v5/comparison/paired_task_deltas.csv \
  --output-dir eval/afc_task_prototype_bank_cls_feedback_v1 \
  --encoder-kind transformer \
  --feedback-blend 0.75
```

Artifacts:
- `eval/afc_task_prototype_bank_cls_feedback_v1/transformer_prototype_bank.json`
- `eval/afc_task_prototype_bank_cls_feedback_v1/feedback_update_summary.json`

Offline retrieval effect:
- exact dataset reuse now shifts according to prior outcome evidence rather than using only geometry
- `credit_g` gains positive self-bias and a smaller positive pull toward `pendigits`
- `diabetes` receives a small negative self-bias because its earlier reward was slightly negative

Live rerun:

```bash
GRAPHDRONE_TASK_PRIOR_BANK_DIR=eval/afc_task_prototype_bank_cls_feedback_v1 \
GRAPHDRONE_TASK_PRIOR_ENCODER_KIND=transformer \
GRAPHDRONE_TASK_PRIOR_STRENGTH=0.5 \
PYTHONPATH=src python scripts/run_champion_challenger.py \
  --task classification \
  --datasets credit_g diabetes \
  --folds 0 \
  --max-samples 384 \
  --champion-version champion-bin-no-prior-v6 \
  --challenger-version challenger-bin-task-prior-feedback-v6 \
  --output-dir eval/afc_live_task_prior_binary_feedback_v6
```

Result:
- decision remains `hold`
- headline improvement dropped from `0.004135` to `0.002459`
- mean log-loss guardrail still passed and improved slightly

Dataset read:
- `credit_g` got slightly worse log-loss than the static-bank run
- `diabetes` got slightly better log-loss than the static-bank run
- F1 stayed unchanged on both
- defer remained saturated near `1.0`

Most important interpretation:
- the bank is no longer static; prediction outcomes can now reshape future prior lookup
- this is real architectural progress even though the current benchmark delta is still small
- but the live route still queries the bank as an anonymous `__current__` task, so exact reuse is not fully activated during routing
- the current gain path is therefore mainly similarity-propagated feedback, not true stable dataset reuse in the live model

So the next bottleneck is narrower now:
- not whether the bank can learn from outcomes
- but whether live GraphDrone should query the bank with a persistent task signature or key, rather than only an anonymous per-run context

## Persistent task key and exact-reuse blend

The next fix was to stop querying the bank as `__current__` and pass the real benchmark dataset key through the live GraphDrone config.

Changes:
- benchmark runners now pass `dataset_key=<dataset>`
- live diagnostics now expose:
  - `task_prior_query_dataset`
  - `task_prior_exact_reuse_available`
  - `task_prior_exact_reuse_used`
  - `task_prior_exact_reuse_blend`

### Exact key only

Live rerun:

```bash
GRAPHDRONE_TASK_PRIOR_BANK_DIR=eval/afc_task_prototype_bank_cls_feedback_v1 \
GRAPHDRONE_TASK_PRIOR_ENCODER_KIND=transformer \
GRAPHDRONE_TASK_PRIOR_STRENGTH=0.5 \
PYTHONPATH=src python scripts/run_champion_challenger.py \
  --task classification \
  --datasets credit_g diabetes \
  --folds 0 \
  --max-samples 384 \
  --champion-version champion-bin-no-prior-v7 \
  --challenger-version challenger-bin-task-prior-feedback-exact-v7 \
  --output-dir eval/afc_live_task_prior_binary_feedback_exact_v7
```

Read:
- exact reuse became live:
  - `task_prior_query_dataset=credit_g|diabetes`
  - `task_prior_exact_reuse_available=1`
- but the bank still ranked `pendigits` as top neighbor on both datasets
- conclusion: knowing the task key is necessary but not sufficient; the live prior still remained dominated by neighborhood mix

### Exact-reuse blend

To make known-task memory explicit rather than implicit, the prior vector now blends:
- similarity-weighted neighborhood prior
- exact task centroid when available

Rerun:

```bash
GRAPHDRONE_TASK_PRIOR_BANK_DIR=eval/afc_task_prototype_bank_cls_feedback_v1 \
GRAPHDRONE_TASK_PRIOR_ENCODER_KIND=transformer \
GRAPHDRONE_TASK_PRIOR_STRENGTH=0.5 \
GRAPHDRONE_TASK_PRIOR_EXACT_REUSE_BLEND=0.75 \
PYTHONPATH=src python scripts/run_champion_challenger.py \
  --task classification \
  --datasets credit_g diabetes \
  --folds 0 \
  --max-samples 384 \
  --champion-version champion-bin-no-prior-v8 \
  --challenger-version challenger-bin-task-prior-feedback-exactblend-v8 \
  --output-dir eval/afc_live_task_prior_binary_feedback_exactblend_v8
```

Result:
- decision still `hold`
- mean log-loss guardrail improved slightly to `-0.001478`
- `task_prior_exact_reuse_used=1` on both datasets

Dataset read:
- `credit_g`: slightly worse than the prior feedback-only run
- `diabetes`: better than the prior feedback-only run
- F1 still flat
- defer still saturated near `1.0`

Interpretation:
- persistent task identity now works
- explicit exact-reuse blending is active
- this is another real architectural step: the live model is no longer limited to anonymous neighborhood borrowing
- but the remaining blocker is now even clearer:
  - the binary routing objective still pushes almost all mass into defer
  - so better task priors mostly show up as small calibration/log-loss shifts rather than a new routing policy
