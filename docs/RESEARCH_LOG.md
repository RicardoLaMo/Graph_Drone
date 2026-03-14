# GraphDrone v1-width Research Log

Each row is one experiment. "Sprint ELO" = 8-dataset × fold-0 canary.
"Full ELO" = 43-dataset × 3-fold official run. Baseline is the last `v1.0.0-gora` full run.

## Baseline

| Label | Branch/Tag | Sprint ELO | Full ELO | Full Rank | Notes |
|-------|-----------|-----------|---------|-----------|-------|
| **v1.0.0-gora** | `tag:v1.0.0-gora` | — | **1420.3** | **19 / 58** | 43 datasets × 3 folds, 129 tasks |

---

## Experiment Log

| ID | Branch | Hypothesis | Changed Files | Sprint ELO Δ | Full ELO Δ | Status | Decision |
|----|--------|-----------|--------------|-------------|-----------|--------|----------|
| P0-A | `exp/p0-attn-fix` | `attn_weights` (not `v`) → router has true inter-expert attention | `set_router.py:59` | — | — | ✅ merged into P0-AB | — |
| P0-B | `exp/p0-loss-fn` | BCE loss for binary tasks → router optimises correct metric | `model.py:111` | — | — | ✅ merged into P0-AB | — |
| P0-AB | `exp/p0-combined-v2` | Both P0-A + P0-B together (+ NaN guard in BCE path) | `set_router.py`, `model.py` | **+35.4** | **+20.8** (1441.1, rank 18.8/58, winrate 68.7%) | ✅ keep | merged → `v1-width.1` |
| P1-A | `exp/p1-descriptor-norm` | Normalise input_dim/preferred_k in descriptor token (raw int → fraction) | `token_builder.py` | _pending_ | — | 🔄 running | — |
| P1-B | `exp/p1-snr-wire` | Wire SNR via k-NN label statistics (mean_y, var_y) → router gets reliability signal | `model.py` | _pending_ | — | 🔄 running | — |
| P1-C | `exp/p1-kappa-vec` | Vectorise kappa SVD + LID loop → batch numpy, ~1.5–5x faster on large datasets | `observers.py` | _pending_ | — | 🔄 running | — |

---

## How to Add an Entry

1. Create branch `exp/<id>-<short-name>` from `v1-width`.
2. Make the change. Commit with message `exp(<id>): <hypothesis>`.
3. Run `python scripts/run_sprint.py` → copy Sprint ELO Δ into the table.
4. If Sprint ELO Δ > 0: run full benchmark (`--retry` auto-skips cached tasks).
5. Fill Full ELO Δ and set Status = ✅ keep / ❌ reject / 🔁 iterate.
6. If ✅: `git merge --no-ff exp/<id>` into `v1-width`, tag `v1-width.<N>`.
7. If ❌: tag branch `exp/rejected/<id>`, leave for reference.

---

## Sprint Dataset Contract

Fixed 8 datasets, fold 0 only. **Never change this set between experiments.**

| Dataset | Type | Why included |
|---------|------|-------------|
| `kddcup09_appetency` | binary | Biggest GD gap vs baseline |
| `bank-marketing` | binary | Moderate gap, high traffic |
| `Diabetes130US` | binary | Large NaN dataset, GD loses badly |
| `diabetes` | binary | Small, fast, GD was competitive — catches regression |
| `concrete_compressive_strength` | regression | Small, fast regression canary |
| `airfoil_self_noise` | regression | Small, fast regression canary |
| `credit-g` | binary | Medium binary canary |
| `APSFailure` | binary | GD near-best — catches regression on large datasets |

Sprint runtime target: **< 5 min** (8 tasks × fold 0, 1 per GPU, all parallel).
