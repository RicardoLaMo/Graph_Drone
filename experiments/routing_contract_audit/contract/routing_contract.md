# Routing Contract

## Purpose

This contract defines exactly what **routing** means for the audit and future experiments.

The goal is to remove ambiguity and prevent an implementation from drifting into:
- post-hoc weighted ensembling,
- curvature-as-feature prediction,
- or any other creative reinterpretation.

This contract is the authority. If an implementation differs from this document, the implementation is wrong.

---

## Core definition

Routing means:

> observer features control **which view is trusted** and **whether view information is isolated or interacted** **before final prediction**.

Routing is a control mechanism over representations, not a direct predictive feature.

---

## What routing is **not**

The following do **not** count as intended routing:

1. Appending curvature or observer features directly to the predictive feature vector and calling that routing.
2. Adding a “curvature view” and averaging it with other views.
3. Using observer features only in the last MLP after all view representations are already fixed and merged.
4. Using only a learned weighted ensemble over final per-view predictions/logits without an explicit routing mechanism.
5. Any implementation where observer features do not produce explicit routing variables.

If the implementation does any of the above and calls it “routing,” that is a failure of comprehension.

---

## Required routing variables

For each row `i`, compute an observer vector:

```python
g_i = [kappa_i, lid_i, lof_i, density_i, degree_i, view_stats_i...]
```

The intended routing implementation must produce:

### 1. View weights

```python
pi_i = softmax(W g_i)
```

Semantics:
- `pi_i[v]` is how much row `i` trusts view `v`
- `pi_i` must sum to 1 across views

### 2. Mode gate

```python
beta_i = sigmoid(U g_i)
```

Semantics:
- `beta_i -> 0` means **isolation**
- `beta_i -> 1` means **interaction**

This semantic direction is fixed and must be printed in the report.

---

## Representation semantics

Assume there are `V` views and each view encoder produces a row representation:

```python
rep_i_v = encoder_v(row_i, view_v)
```

These are stacked into:

```python
reps_i = [rep_i_1, rep_i_2, ..., rep_i_V]
```

Shape convention:

- `reps`: `[B, V, D]`
- `g`: `[B, G]`
- `pi`: `[B, V]`
- `beta`: `[B, 1]`

---

## Required isolation branch

Isolation must treat views as separately trusted sources and combine them only after view-level representations exist.

Minimum valid form:

```python
weighted = reps * pi.unsqueeze(-1)
iso_rep = weighted.sum(dim=1)
```

This is the minimum semantics for isolation in the audit.

Interpretation:
- each view contributes independently according to `pi`
- no extra cross-view transformation happens yet

---

## Required interaction branch

Interaction must create a jointly transformed fused representation.

Minimum valid form:

```python
inter_rep = interaction_mlp(iso_rep)
```

This is the minimum semantics for interaction in the audit.

Interpretation:
- interaction means the fused information is jointly transformed
- this differs from isolation, which stops at weighted aggregation

Note:
This is a minimal audit version, not the full GoRA/THORN formulation.
The audit only needs a faithful simplified implementation.

---

## Required final blend

The final representation must be formed by blending isolation and interaction using `beta`:

```python
final_rep = (1.0 - beta) * iso_rep + beta * inter_rep
```

Then prediction is made from `final_rep`.

This means:
- `beta = 0` -> pure isolation
- `beta = 1` -> pure interaction

This must not be reversed.

---

## Model A vs Model B contract

## Model A: `A_posthoc_combiner`

This model is the control model representing the weaker interpretation.

Allowed behavior:
- same per-view encoders as Model B
- may learn to combine view representations or predictions
- may optionally receive observer features as extra combiner input

Not allowed behavior:
- must not define explicit `pi` and `beta` with the semantics above
- must not use observer features to control isolation vs interaction

Model A represents:
> observer-assisted post-hoc combination

---

## Model B: `B_intended_router`

This model is the intended routing implementation.

Required behavior:
- uses same per-view encoders as Model A
- computes explicit `pi` from observer vector `g`
- computes explicit `beta` from observer vector `g`
- forms `iso_rep`
- forms `inter_rep`
- blends them before final prediction

Model B represents:
> observer-driven routing over views and mode

If Model B does not do all of the above, it is not compliant.

---

## Fairness constraints for the audit

To isolate routing semantics:

1. Model A and Model B must share:
   - same dataset splits
   - same views
   - same per-view encoders
   - same observer vector definition
   - same output head dimensionality where practical

2. The substantive difference must be:
   - Model A = post-hoc combination
   - Model B = explicit routing through `pi` and `beta`

3. Do not add rank/compression routing in the first pass.

---

## Required code interfaces

The implementation must define or closely match these interfaces.

```python
class ObserverRouter(nn.Module):
    def forward(self, g):
        '''
        g: [B, obs_dim]
        returns:
          pi:   [B, n_views]
          beta: [B, 1]
        '''
```

```python
class ViewEncoder(nn.Module):
    def forward(self, batch_view):
        '''
        returns:
          rep: [B, rep_dim]
        '''
```

```python
class PosthocCombinerA(nn.Module):
    def forward(self, reps, g=None):
        '''
        reps: [B, V, D]
        returns prediction
        No explicit pi/beta semantics.
        '''
```

```python
class IntendedRouterB(nn.Module):
    def forward(self, reps, g):
        '''
        reps: [B, V, D]
        g:    [B, G]
        must compute:
          pi
          beta
          iso_rep
          inter_rep
          final_rep
        then predict
        '''
```

---

## Required report outputs

The report must include all of the following.

### Routing semantics section
- one sentence defining routing
- one sentence defining `beta`
- one sentence distinguishing A vs B

### Routing behavior outputs
- mean `pi` by view
- top-1 view frequency
- routing entropy summary
- mean `beta`
- mean `beta` by curvature bin or regime
- examples of rows where `pi` differs strongly

### Audit conclusion section
The report must answer:
1. Did A and B differ architecturally in the intended way?
2. Did B behave like routing rather than post-hoc weighting?
3. Did B improve prediction?
4. If not, did B at least show the intended routing semantics?

---

## Required synthetic acceptance tests

Before any dataset experiment, the implementation must pass synthetic tests.

### Test 1: `pi` changes when `g` changes
Hold `reps` fixed, vary `g`.
Expected:
- `pi` must change

### Test 2: `beta` changes when `g` changes
Hold `reps` fixed, vary `g`.
Expected:
- `beta` must change

### Test 3: prediction changes through routing path
Hold `reps` fixed, vary `g`.
Expected:
- Model B prediction changes through `pi/beta`
- Model A may or may not change depending on design

### Test 4: `beta` semantics are correct
Construct a case with forced low `beta` and high `beta`.
Expected:
- low `beta` yields output close to `iso_rep`
- high `beta` yields output close to `inter_rep`

### Test 5: `pi` sums to 1
Expected:
- each row's `pi` sums to 1 within tolerance

If these tests fail, dataset experiments should not proceed.

---

## Non-goals of first pass

The following are explicitly excluded from first-pass routing audit:

- observer-driven rank selection
- observer-driven compression choice
- full per-head routing
- full GoRA-style pre-softmax adjacency mixing
- FGW / GW alignment inside the router

These may be added later only after routing semantics are validated.

---

## Final summary

The first-pass intended routing audit is successful only if:

1. Model B explicitly computes and uses `pi` and `beta`
2. `pi` and `beta` alter representation combination before prediction
3. synthetic tests confirm the semantics
4. the report documents routing behavior clearly

Anything less is not intended routing.
