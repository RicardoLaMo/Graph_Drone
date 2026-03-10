# P0 Design Note

This note explains why the `P0` family works and why `P0_crossfit` is provisionally ranked first in the current California benchmark table.

Relevant implementation files:
- `/Volumes/MacMini/Projects/Graph_Drone/.worktrees/mv-tabpfn-view-router/experiments/tabpfn_view_router/scripts/run_experiment.py`
- `/Volumes/MacMini/Projects/Graph_Drone/.worktrees/mv-tabpfn-view-router/experiments/tabpfn_view_router/src/data.py`
- `/Volumes/MacMini/Projects/Graph_Drone/.worktrees/mv-tabpfn-view-router/experiments/tabpfn_view_router/src/router.py`

## 1. Architecture in one sentence

`P0` is a **mixture-of-view-experts** model:
- train one `TabPFNRegressor` per view
- predict with each view independently
- learn a row-wise softmax router over those predictions using view-quality and view-agreement features

It is **not** a new TabPFN backbone. The backbone is the same TabPFN expert class applied separately to `FULL`, `GEO`, `SOCIO`, and `LOWRANK`.

## 2. Data and views

The aligned California split and view construction are in `src/data.py`.

For each row `i`, the model builds four feature subspaces:
- `FULL`
- `GEO`
- `SOCIO`
- `LOWRANK`

Each view gets its own TabPFN expert:

\[
\hat y_i^{(v)} = f_v(x_i^{(v)}), \qquad v \in \{\text{FULL},\text{GEO},\text{SOCIO},\text{LOWRANK}\}
\]

The stacked expert prediction vector is:

\[
p_i =
\begin{bmatrix}
\hat y_i^{(\text{FULL})} \\
\hat y_i^{(\text{GEO})} \\
\hat y_i^{(\text{SOCIO})} \\
\hat y_i^{(\text{LOWRANK})}
\end{bmatrix}
\in \mathbb{R}^4
\]

## 3. Routing features

The router does **not** look at raw tabular features. It looks at compact view-quality signals computed from the current row:

- per-view local label variance proxy: `sigma2_v`
- pairwise view-neighborhood overlap features: `J_flat`
- mean overlap: `mean_J`

In code, this is the `QualityFeatures` bundle in `src/data.py`.

So the router input is:

\[
r_i = [\sigma^2_i, J_i^{\text{flat}}, \bar J_i] \in \mathbb{R}^{11}
\]

with:
- `4` sigma terms
- `6` pairwise Jaccard terms
- `1` mean Jaccard term

## 4. Learned router

The learned router in `src/router.py` is:

\[
w_i = \mathrm{softmax}(\mathrm{MLP}(r_i)) \in \mathbb{R}^4
\]

and the final prediction is:

\[
\hat y_i = \sum_{v=1}^4 w_{i,v}\,\hat y_i^{(v)}
\]

Implementation details that matter:
- hidden dimension is only `32`
- final linear layer is zero-initialized
- this means training starts near uniform mixing and only departs when the data supports it

This small meta-model is enough because the heavy lifting is done by the four TabPFN experts.

## 5. Why the learned router works

The key empirical fact is that the four view experts are **not equally good**.

From the stored full report:
- `P0_FULL` is strong
- `P0_GEO` is useful but weaker
- `P0_SOCIO` and `P0_LOWRANK` are much worse alone

So the right solution is not:
- uniform averaging
- inverse-variance weighting only
- a closed-form GoRA temperature rule

The right solution is:
- trust `FULL` most of the time
- preserve meaningful `GEO` contribution
- suppress `SOCIO` and `LOWRANK` almost completely

That is exactly what the learned router does.

Average learned weights in the seed-42 full report are approximately:

\[
w \approx [0.72,\ 0.23,\ 0.03,\ 0.02]
\]

for:
- `FULL`
- `GEO`
- `SOCIO`
- `LOWRANK`

This is the core reason `P0_router` beats:
- `P0_FULL`
- `P0_uniform`
- `P0_sigma2`
- `P0_gora`

## 6. Why the analytical routes lose

### Uniform

\[
w_{i,v} = \frac{1}{4}
\]

This assumes all experts are equally reliable. They are not.

### Sigma2

The sigma-only mixer uses:

\[
w_{i,v} \propto \frac{1}{\sigma^2_{i,v} + c}
\]

This is better motivated than uniform averaging, but it is still too rigid:
- it ignores pairwise view agreement structure
- it cannot learn nonlinear combinations of the routing features

### GoRA analytical rule

The analytical GoRA-style route is:

\[
\tau_i = \frac{1}{\bar J_i + \epsilon}, \qquad
w_i = \mathrm{softmax}(-\sigma_i^2 \cdot \tau_i)
\]

This is elegant but over-constrained. It assumes a single fixed formula of `sigma2` and `mean_J` is enough. The learned router relaxes that assumption and wins.

## 7. Why `P0_crossfit` is rank 1

This is the subtle part.

`P0_crossfit` is **not meaningfully different in architecture** from `P0_router`.

The difference is in how the router is trained and evaluated.

### `P0_router`

`fit_soft_router(...)`:
1. split validation rows into router-train and router-holdout
2. train router on router-train
3. early-stop on router-holdout
4. evaluate the fitted router on the full validation set and on test

This is reasonable, but the validation set is still used both for fitting the meta-router and for reporting its validation score.

### `P0_crossfit`

`fit_crossfit_router(...)`:
1. split validation into 5 folds
2. for each fold:
   - fit the router on the other 4 folds
   - predict the held-out fold
3. assemble out-of-fold predictions for the whole validation set
4. train one final router on all validation rows for test-time prediction

So the reported validation RMSE is an **OOF estimate**, which is cleaner and less optimistic.

That is why `P0_crossfit` edges out `P0_router` in the provisional leaderboard.

The important honest conclusion is:
- `P0_crossfit` is best because it is a **cleaner protocol**
- not because it discovered a different mathematical solution

The test means are essentially tied.

## 8. Why this can beat global TabPFN

Global TabPFN is one strong expert over the full feature set.

`P0` improves on it because:
1. `FULL` already gives a very strong TabPFN expert
2. `GEO` adds complementary information
3. the router learns when the complementary geographic signal should matter
4. weak views are suppressed rather than averaged in

So the win is not “multi-view by itself.”
It is:

\[
\text{strong base expert} + \text{one complementary specialist} + \text{row-wise sparse mixture}
\]

## 9. Practical interpretation

The strongest claim supported by the code and reports is:

> Per-view TabPFN experts plus a small GoRA-style routing model produce a real and repeatable improvement over a single global TabPFN expert on California.

The weaker claim, which should be avoided, is:

> `P0_crossfit` is a fundamentally new better architecture than `P0_router`.

That is not what the code shows.

## 10. What to do next

If continuing the `P0` line, the next meaningful architectural target is **not** crossfitting itself. It is one of:
- better routing features
- stronger evaluation across more datasets / matched split sweeps
- applying the same per-view expert + router pattern in the TabArena protocol

The main mathematical idea has already been validated:

\[
\hat y_i = \sum_v w_{i,v}(r_i)\,\hat y_i^{(v)}
\]

with `w_i` learned from view-quality and view-agreement signals.
