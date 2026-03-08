# California Tab Foundation Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Lock a trustworthy California champion baseline, add TabM as a second foundation baseline, align both to this repo's California protocol, and then run the first challenger experiments cleanly.

**Architecture:** Treat TabR and TabM as foundation models, not hybrid experiments. First freeze one champion and one comparator under reproducible CPU execution. Then create an aligned California protocol so future comparisons are fair. Only after that run a controlled decoder challenger, changing the readout while keeping the backbone/value path fixed.

**Tech Stack:** Python, PyTorch, FAISS CPU for TabR, TabM package/paper stack, Apple Silicon CPU-first execution, git worktrees, repo-local experiment reports and CSV artifacts.

---

## Branch / Worktree Layout

- Stable baseline branch:
  - `feature/tabr-california-baseline`
- Comparison branch:
  - `feature/california-tab-foundation-compare`
- Worktree:
  - `/Volumes/MacMini/Projects/Graph_Drone/.worktrees/california-tab-foundation-compare`

## Scope

This phase covers:
- `M4` TabR champion lock
- TabM California baseline addition
- `M5` alignment audit and aligned runs
- `C1` aligned foundation comparison
- `C2` decoder challenger

This phase does **not** cover:
- MPS optimization
- full hybrid encoder swaps
- joint view-routing redesigns

## M4: Champion Lock

### Objective

Freeze one exact champion candidate before adding comparisons.

### Champion choice

- Champion: `TabR_CPU_fixed`
- Source result:
  - [0-evaluation__0.json](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/reports/0-evaluation__0.json)
- Locked metrics:
  - test RMSE `0.3949403615264023`
  - val RMSE `0.40094805430544206`
  - best epoch `49`

### Files

- Create: `experiments/tab_foundation_compare/reports/champion_report.md`
- Create: `experiments/tab_foundation_compare/artifacts/champion_metrics.csv`
- Create: `experiments/tab_foundation_compare/artifacts/environment_snapshot.json`

### Tasks

1. Copy the exact champion metadata from the TabR branch.
2. Record the exact command, config, environment, and metrics.
3. State explicitly that CPU is the locked execution device.
4. Commit:
   - `docs: lock TabR CPU champion for California`

## TabM Foundation Baseline

### Objective

Add a second strong tabular DL baseline for California.

### Rationale

- TabM is a modern MLP-style deep baseline with good practical fit on current PyTorch.
- It gives a clean comparison point against retrieval-heavy TabR and our routing-heavy models.
- Published California performance is strong enough to matter, but not obviously stronger than TabR.

### Files

- Create: `experiments/tabm_california_baseline/README.md`
- Create: `experiments/tabm_california_baseline/scripts/check_env.py`
- Create: `experiments/tabm_california_baseline/scripts/run_tabm_california.py`
- Create: `experiments/tabm_california_baseline/scripts/export_upstream_refs.py`
- Create: `experiments/tabm_california_baseline/artifacts/upstream_reference_metrics.csv`
- Create: `experiments/tabm_california_baseline/artifacts/metrics.csv`
- Create: `experiments/tabm_california_baseline/artifacts/runtime.csv`
- Create: `experiments/tabm_california_baseline/reports/m0_environment.md`
- Create: `experiments/tabm_california_baseline/reports/m1_smoke_report.md`
- Create: `experiments/tabm_california_baseline/reports/m2_fixed_config_report.md`

### Tasks

1. Start from the upstream paper California config:
   - `/private/tmp/tabm_clone_inspect_20260308/paper/exp/tabm/california/0-evaluation/0.toml`
2. Implement a local California export path mirroring what was done for TabR.
3. Use CPU-first execution on Apple Silicon.
4. First run a smoke config.
5. Then run one fixed config.
6. Record:
   - metrics
   - runtime
   - exact environment
   - upstream comparison
7. Commit:
   - `feat: add TabM California baseline`

## M5: Alignment Audit

### Objective

Create one fair California comparison protocol across TabR, TabM, and this repo.

### Files

- Create: `experiments/tab_foundation_compare/reports/alignment_report.md`
- Create: `experiments/tab_foundation_compare/artifacts/protocol_diff.csv`
- Create: `experiments/tab_foundation_compare/artifacts/alignment_metrics.csv`

### Required audit points

- dataset source
- split ratios
- split seed
- numeric transforms
- target normalization
- metric computation
- any train/val/test leakage risks

### Decision rule

Preferred alignment direction:
- run TabR and TabM on **our** California split/protocol

Fallback:
- run repo challengers on the foundation-model local export protocol

### Tasks

1. Audit the current TabR local export protocol.
2. Audit the planned TabM local export protocol.
3. Audit our California protocol from:
   - `v3.5`
   - current California experiment lineage under `experiments/`
4. Write a diff table.
5. Implement one aligned data protocol.
6. Re-run both TabR and TabM under that aligned path.
7. Commit:
   - `feat: align California protocol for foundation baselines`

## C1: Aligned Foundation Comparison

### Objective

Compare the strongest non-hybrid baselines under one protocol.

### Ladder

- `TabR_CPU_fixed_aligned`
- `TabM_CPU_fixed_aligned`
- repo references:
  - `B1_HGBR`
  - `G2_GoRA_v1`
  - `CA_v35b`
  - strongest head-routing California variant available

### Files

- Create: `experiments/tab_foundation_compare/reports/c1_foundation_comparison.md`
- Create: `experiments/tab_foundation_compare/artifacts/c1_metrics.csv`

### Tasks

1. Build one metrics table with all aligned runs.
2. Rank by test RMSE, then by training/runtime cost.
3. State whether TabR remains champion after alignment.
4. State whether TabM closes or widens the gap versus repo models.
5. Commit:
   - `docs: add C1 aligned California foundation comparison`

## C2: Decoder Challenger

### Objective

Test whether a routing-aware decoder improves the strongest aligned foundation without changing its core backbone/value path.

### Constraint

Change the decoder only. Do not mix in encoder or retrieval redesigns.

### Preferred base

- Use whichever wins `C1`.
- Current expected base: `TabR_CPU_fixed_aligned`

### Proposed C2 design

- Baseline:
  - original foundation predictor
- Challenger:
  - gated multi-slot decoder
  - per-slot predictions `y_h`
  - row-conditional gate `a_h`
  - final prediction:
    - `y = sum_h a_h * y_h + residual_global`

### Gate inputs

- target embedding / final hidden representation
- retrieval concentration or slot concentration
- optional uncertainty/confidence summaries
- keep inputs minimal and prediction-relevant

### Files

- Create: `experiments/tab_foundation_compare/scripts/run_c2_decoder.py`
- Create: `experiments/tab_foundation_compare/reports/c2_decoder_report.md`
- Create: `experiments/tab_foundation_compare/artifacts/c2_metrics.csv`

### Tasks

1. Reproduce the aligned foundation winner unchanged.
2. Add the gated decoder variant.
3. Run at least one seed-matched comparison.
4. Classify the result:
   - better metric
   - same metric but cleaner credit assignment
   - worse metric, complexity unjustified
5. Commit:
   - `feat: add C2 decoder challenger`

## Expected Insights

- If TabR remains strongest after `M5`, retrieval/value design is genuinely useful for California.
- If TabM closes the gap after alignment, part of the TabR edge may have been protocol or optimization.
- If `C2` improves the aligned winner, decoder credit assignment remains a live bottleneck.
- If `C2` does not improve the aligned winner, the main leverage is likely upstream representation/value construction rather than readout.

## Immediate Execution Order

1. Lock `M4` champion report for TabR.
2. Add TabM California smoke runner.
3. Add TabM fixed-config run.
4. Write and execute `M5` alignment audit.
5. Run `C1` aligned comparison.
6. Only then implement `C2`.
