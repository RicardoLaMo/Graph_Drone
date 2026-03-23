# GraphDrone v1.3 Regression Research Handoff

Date: 2026-03-23

Prepared from branch:
- `exp/v13-regression-program`

Prepared for:
- internal/external research teams continuing GraphDrone regression

## Why v1.3 is regression-first

The binary classification line proved that the research operating system is working:
- branchable hypotheses
- champion/challenger evaluation
- claim-first diagnostics
- threshold/mechanism analysis
- durable research memory
- handoff-ready notes with branch/SHA traceability

Regression should now use the same operating model, but not the same technical agenda.

The current regression evidence is more specific:
- useful specialists exist, but realized specialist value is often negative
- anchor contamination in the regression defer path was real and partly causal
- `california` still exposes a real router-instability / diagnostics-coverage hole
- rotor/alignment can improve geometry without improving RMSE
- local routing and allocation appear to be a stronger bottleneck than cross-dataset prior learning

So `v1.3` should be a regression-first research program focused on routing quality and causal diagnosis before broadening to new priors.

## Stable references

### Regression evidence already established

Most important notes:
- `docs/2026-03-23-afc-phase-b-anchor-exclusion.md`
- `docs/2026-03-23-afc-phase-b-residual-usefulness.md`
- `docs/2026-03-23-afc-phase-b-california-router-instability.md`
- `docs/research/current_hypotheses.md`

Most important currently established regression claims:
- `afc-b-reg-anchor-asymmetry`
- `afc-b-california-router-instability`
- `afc-b-residual-usefulness-gap`
- `afc-b-rotor-mechanism`

Interpretation:
- the regression problem is not “specialists are useless”
- it is “specialist value exists, but the router does not realize it reliably”

## v1.3 branch structure

These are the intended continuation branches for v1.3 regression.
They may initially point to the same base SHA, but each should evolve as an isolated mechanism lane.

### 1. Regression stability lane

- branch: `exp/v13-reg-stability`
- question:
  - can we make regression router failures explainable and fully diagnosable?

Scope:
- classify and surface all regression fallback modes
- remove coverage holes like `california` where predictions are finite but mechanism diagnostics are incomplete
- make `router_training_nonfinite_anchor_only` an explained state, not just a safe fallback

Success condition:
- every regression fallback case is diagnosable from artifacts without code archaeology

### 2. Regression usefulness-routing lane

- branch: `exp/v13-reg-usefulness-routing`
- question:
  - can regression routing be trained to realize positive specialist value rather than only detect that it exists?

Scope:
- make realized specialist value the primary mechanism target
- optimize expert allocation / attention quality rather than only blended prediction loss
- keep champion/challenger metric checks, but only after mechanism movement is clear

Success condition:
- positive available specialist advantage becomes positive realized specialist advantage on the validation surface, and then translates to held-out RMSE/R² improvement

### 3. Regression AFC revisit lane

- branch: `exp/v13-reg-afc-revisit`
- question:
  - after the regression circuit is better instrumented, does AFC alignment improve allocation quality rather than only token geometry?

Scope:
- rerun rotor/alignment only on top of the corrected regression circuit
- require alignment gain and allocation gain together
- reject any mechanism that improves cosine/alignment without improving realized specialist value

Success condition:
- AFC-style geometry improvements also improve regression routing usefulness

### 4. Regression meta-prior readiness lane

- branch: `exp/v13-reg-meta-prior-readiness`
- question:
  - is local regression router learning still too weak after the usefulness-routing lane, such that a task-conditioned regression prior becomes justified?

Scope:
- this is a later lane, not the first move
- only start it if local routing evidence says single-dataset learning remains the blocker
- do not force regression into the current classification-first task bank without new evidence

Success condition:
- explicit evidence that local regression routing is still the bottleneck after the circuit is improved

## What teams should do first

Recommended order:

1. `exp/v13-reg-stability`
2. `exp/v13-reg-usefulness-routing`
3. `exp/v13-reg-afc-revisit`
4. `exp/v13-reg-meta-prior-readiness`

This order matters.

Do not start with regression meta-priors.
Do not start with another broad AFC rerun.
Do not spend the next cycle on benchmark-only expansion.

The next cycle should answer:
- why and where regression routing fails
- whether allocation quality can be improved directly
- whether AFC geometry then helps the improved circuit

## Required artifacts for every v1.3 regression branch

Each branch must return:
1. branch name
2. head SHA
3. one note under `docs/`
4. main artifact paths under `eval/`
5. one durable finding added to `docs/research/findings.jsonl`
6. one status:
   - `cleared`
   - `partially_causal`
   - `open`
   - `confounded`
   - `falsified`
7. one sentence stating:
   - what hypothesis was tested
   - what changed causally
   - whether the next team should continue spending on it

Preferred commit style:
- code: `exp(v13-reg): <mechanism>`
- docs/research memory: `docs(v13-reg): record <finding>`

## Required regression mechanism diagnostics

These should be treated as first-class outputs for v1.3:
- `validation_best_specialist_advantage_score`
- `validation_weighted_specialist_advantage_score`
- `validation_defer_weighted_specialist_advantage_score`
- `validation_positive_specialist_mass`
- `validation_top_specialist_positive_rate`
- `validation_residual_usefulness_gap`
- `mean_specialist_mass`
- `mean_anchor_attention_weight`
- `router_nonfinite_fallback`

These are the minimum mechanism surface for claiming progress on regression routing.

## Benchmark contract for v1.3 regression

Decision rule:
- mechanism-first
- benchmark-second

Practical order:
1. quick regression smoke
2. fold-0 mini-full regression gate
3. full 3-fold only after mechanism movement is clear

Primary metrics:
- `rmse`
- `mae`
- `r2`

Important rule:
- quality-neutral but causally clarifying work is acceptable in v1.3 only if it closes a real hypothesis and is written into research memory

## What is already falsified or narrowed

Do not restart these blindly:

- the original residual-usefulness gap penalty formulation
  - already falsified as a direct fix
- rotor geometry alone as a success condition
  - already known to be insufficient
- broad regression AFC adoption without allocation evidence
  - not justified yet
- regression cross-dataset prior as the first move
  - not supported strongly enough yet

## Recommended checkouts

Program branch:

```bash
git checkout exp/v13-regression-program
```

If starting a lane:

```bash
git checkout exp/v13-regression-program
git checkout -b exp/v13-reg-<lane-name>
```

If using the predefined branch refs:

```bash
git checkout exp/v13-reg-stability
git checkout exp/v13-reg-usefulness-routing
git checkout exp/v13-reg-afc-revisit
git checkout exp/v13-reg-meta-prior-readiness
```

## Final current recommendation

For `v1.3`, the best next regression move is not “more AFC” and not “more benchmark.”

It is:
- stabilize regression routing diagnostics
- directly improve realized specialist value
- then re-test AFC alignment on top of that corrected circuit
