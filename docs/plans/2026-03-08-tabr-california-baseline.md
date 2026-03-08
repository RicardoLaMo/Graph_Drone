# TabR California Champion Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce a trustworthy TabR baseline on California Housing in this repo, on Apple Silicon, and then use it as the champion for later component-swap challengers.

**Architecture:** Keep the first milestone as close to upstream TabR as possible: upstream data format, upstream California config, CPU FAISS retrieval, and a fixed evaluation config. Only after that baseline is stable do we add an Apple/MPS device port and then controlled swaps with our own architecture pieces.

**Tech Stack:** Python, PyTorch, FAISS CPU, Optuna configs from upstream TabR, git worktrees, Apple Silicon / MPS-capable local machine.

---

## Ground Rules

- Do not mix this work with existing GoRA/head-routing branches.
- Do not trust MPS until CPU reproduction is stable.
- Do not tune first. Reproduce a shipped California evaluation config first.
- Keep champion/challenger comparisons single-change and reversible.
- Record exact environment details with every milestone.

## Branch / Worktree Layout

- Branch: `feature/tabr-california-baseline`
- Worktree: `/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline`

## Proposed Repo Layout

Create these tracked paths in this branch:

- `experiments/tabr_california_baseline/README.md`
- `experiments/tabr_california_baseline/upstream/`
- `experiments/tabr_california_baseline/scripts/`
- `experiments/tabr_california_baseline/configs/`
- `experiments/tabr_california_baseline/reports/`
- `experiments/tabr_california_baseline/artifacts/`
- `experiments/tabr_california_baseline/logs/`

Recommended contents:
- `upstream/`
  - minimal vendored TabR source or pinned snapshot metadata
- `scripts/`
  - environment check
  - California fixed-config runner
  - metrics summarizer
- `configs/`
  - copied fixed California config(s) from upstream
  - one local override config for CPU-first runs
- `reports/`
  - milestone reports
- `artifacts/`
  - metrics CSV, runtime CSV, environment JSON, copied report excerpts

## Milestone Ladder

### Milestone M0: Environment Capture

**Success condition:** We can state exactly what local environment we are running and what is incompatible with upstream as-is.

**Tasks**
1. Add a script to print Python, torch, MPS, FAISS, and platform versions.
2. Save the output to `experiments/tabr_california_baseline/artifacts/environment.json`.
3. Write a short note in `reports/m0_environment.md`.

**Expected result**
- Clear statement that this machine is Apple Silicon with no CUDA.
- Clear statement whether FAISS CPU imports cleanly.

### Milestone M1: Upstream CPU-Faiss Smoke Run

**Success condition:** A fixed California TabR config runs end-to-end on CPU with no algorithm changes.

**Tasks**
1. Vendor only the minimal upstream files needed for `bin/tabr.py` California evaluation, or pin the external checkout with a reproducible fetch script.
2. Create an isolated environment for the TabR baseline.
3. Use CPU-compatible dependency choices:
   - replace `faiss-gpu` with `faiss-cpu`
   - no CUDA packages
4. Copy one shipped California evaluation config from upstream:
   - start with `exp/tabr/california/default-evaluation/0.toml` or `exp/tabr/california/0-evaluation/0.toml`
5. Run a smoke version first:
   - reduced epochs or smaller debug subset only to confirm the pipeline
6. Save:
   - stdout/stderr log
   - wall-clock runtime
   - resulting metrics

**Expected result**
- We know whether upstream TabR can run on this machine without any MPS patch.

### Milestone M2: Fixed-Config California Reproduction

**Success condition:** We reproduce one shipped TabR California evaluation config on CPU and obtain a sensible RMSE near the upstream reference band.

**Tasks**
1. Run one fixed config without tuning.
2. Compare against upstream saved reference:
   - `default-evaluation/0`
   - `0-evaluation/0`
   - optional `2-plr-lite-evaluation/0`
3. Save:
   - `metrics.csv`
   - `runtime.csv`
   - `report.md`

**Acceptance band**
- Do not require exact equality.
- Accept if RMSE is in a reasonable neighborhood and there is no execution bug.
- If materially worse, classify:
  - environment drift
  - data mismatch
  - package/version mismatch
  - implementation issue

### Milestone M3: Apple/MPS Port Feasibility

**Success condition:** We know whether a minimal MPS port is worth doing.

**Tasks**
1. Patch device selection to prefer `mps` when available.
2. Verify that all non-FAISS tensor ops work on MPS.
3. Keep FAISS search on CPU if needed, with explicit tensor transfer boundaries.
4. Measure whether the MPS path is actually faster or more fragile than CPU.

**Decision rule**
- If MPS adds instability or complex cross-device bugs, keep champion on CPU.
- Do not merge an MPS port into champion status unless metrics match CPU.

### Milestone M4: Champion Lock

**Success condition:** We have one trusted TabR champion baseline on California.

**Tasks**
1. Choose exactly one champion:
   - `TabR_CPU_fixed`
   - or `TabR_MPS_fixed`
2. Freeze:
   - config
   - environment
   - exact command
   - saved metrics
3. Write `reports/champion_report.md`.

### Milestone M5: Challenger Alignment

**Success condition:** We can compare TabR champion to our work without hidden preprocessing/split differences.

**Tasks**
1. Audit split/preprocessing differences between:
   - upstream TabR California
   - our California pipelines
2. Create one aligned-evaluation path:
   - either run TabR on our split
   - or run our model on a TabR-style split
3. Save an alignment note before any architecture swap.

**Important**
- Do not call any model comparison fair before this step.

### Milestone M6: Controlled Component Swaps

**Success condition:** We can evaluate one challenger change at a time against the locked champion.

Recommended order:
1. `C1`: TabR champion unchanged, but evaluated on our split/preprocessing alignment
2. `C2`: TabR retrieval/value path + our head-gated decoder
3. `C3`: Our encoder/backbone + TabR-style target-conditioned value path
4. `C4`: Full hybrid only after earlier challengers are understood

For every challenger:
- one architectural change only
- compare against champion
- classify result:
  - better metric
  - same metric but cheaper/faster
  - worse metric but more interpretable
  - failure

## Exact Early Commands To Target

These are the first commands to implement once execution starts.

1. Environment probe

```bash
source .venv/bin/activate
python experiments/tabr_california_baseline/scripts/check_env.py
```

2. Upstream config capture

```bash
python experiments/tabr_california_baseline/scripts/export_upstream_refs.py
```

3. CPU smoke run

```bash
python experiments/tabr_california_baseline/scripts/run_tabr_california.py --config default-evaluation-0 --smoke --device cpu
```

4. Fixed-config full run

```bash
python experiments/tabr_california_baseline/scripts/run_tabr_california.py --config 0-evaluation-0 --device cpu
```

## Risks And Mitigations

### Risk 1: Upstream dependency drift
- **Problem:** upstream pins older torch/faiss stack than local repo
- **Mitigation:** use an isolated environment for the TabR branch; do not reuse the main repo `.venv` for final champion runs

### Risk 2: MPS is not a drop-in replacement
- **Problem:** upstream only supports `cpu/cuda`; FAISS does not provide a simple MPS backend
- **Mitigation:** keep champion CPU-first; treat MPS as optional challenger infrastructure

### Risk 3: Data mismatch invalidates comparison
- **Problem:** upstream California benchmark may not match our current split/preprocessing
- **Mitigation:** add an explicit alignment milestone before comparing TabR against our models

### Risk 4: Tuning explodes scope
- **Problem:** jumping to Optuna tuning before fixed-config reproduction burns time and muddies diagnosis
- **Mitigation:** fixed shipped config first, tuning later only if needed

## Deliverables

- `experiments/tabr_california_baseline/reports/m0_environment.md`
- `experiments/tabr_california_baseline/reports/m1_smoke_report.md`
- `experiments/tabr_california_baseline/reports/m2_fixed_config_report.md`
- `experiments/tabr_california_baseline/reports/champion_report.md`
- `experiments/tabr_california_baseline/artifacts/environment.json`
- `experiments/tabr_california_baseline/artifacts/metrics.csv`
- `experiments/tabr_california_baseline/artifacts/runtime.csv`
- `experiments/tabr_california_baseline/artifacts/upstream_reference_metrics.csv`

## Progress / Milestone Reporting Format

For each milestone, report:
- objective
- command run
- environment
- runtime
- metrics
- status: PASS / PARTIAL / FAIL
- blocker, if any
- next action

## Current Recommendation

Start with `M0 -> M1 -> M2` only.

That is the smallest path that answers the immediate question:

`Can TabR be established as a trusted California champion on this Apple Silicon machine?`
