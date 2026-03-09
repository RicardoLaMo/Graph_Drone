# H200 Migration Plan

## Goal

Move the critical California experiment lines from the local Apple M4 workflow to an H200 GPU environment with minimal rework and minimal ambiguity about which branch, commit, and command should be treated as source of truth.

This plan assumes:

- local repo root is `/Volumes/MacMini/Projects/Graph_Drone`
- GitHub remote is `https://github.com/RicardoLaMo/Graph_Drone.git`
- the main checkout is intentionally dirty and should not be used as the migration unit
- migration should happen branch-by-branch from pushed worktrees, not from the dirty root checkout

## Source Of Truth

Before reasoning about branch lineage on the GPU host, read:

- `docs/worktree_lineage.md`
- `docs/worktree_registry.json`

Those files distinguish:

- current checkout facts
- sibling worktree facts
- tracked experiment families
- branch-local research lines

## Pushed Critical Branches

These branch heads were pushed to `origin` and should be treated as the migration units.

| Branch | Why it matters | Key artifact |
| --- | --- | --- |
| `codex/mv-tabpfn-view-router` | Critical `P0` series. Current strongest branch-local California line. | `experiments/tabpfn_view_router/reports/p0_multiseed.md` |
| `codex/tabpfn-california-baseline` | Fair aligned TabPFN foundation baseline. | `experiments/tab_foundation_compare/reports/tabpfn_multiseed_e1.md` |
| `codex/mv-geo-segmentation-priors` | Confirms latent geo/community structure from latitude/longitude is useful. | `experiments/california_geo_segmentation/reports/report.md` |
| `codex/mv-tabr-gora-geo-retrieval` | Best geo-segmented retrieval-prior line in the MV-TabR-GoRA family. | `experiments/mv_tabr_gora/reports/geo_retrieval_multiseed.md` |
| `codex/mv-tabr-gora-g0g1-cohort-residual` | Negative control line for cohort residual targets; keep for design history. | `experiments/mv_tabr_gora/reports/cohort_residual/cohort_residual_results.json` |

## Migration Rules

1. Do not copy the local `.venv`.
2. Do not depend on untracked local directories such as `.gemini-cross-checks/`.
3. Reproduce from pushed branch heads plus tracked reports.
4. Keep `split_seed` and training `seed` separate in every runner.
5. Treat each worktree experiment as its own unit. Do not merge branches just to “clean up” before first GPU reproduction.

## Lowest-Cost H200 Bootstrap

Run on the GPU host after cloning the repo:

```bash
git clone https://github.com/RicardoLaMo/Graph_Drone.git
cd Graph_Drone
git fetch --all --prune
git checkout codex/mv-tabpfn-view-router
bash scripts/bootstrap_h200.sh
```

The bootstrap script:

- creates `.venv-h200`
- installs non-PyTorch dependencies from `requirements.txt`
- installs CUDA-enabled PyTorch wheels
- installs `tabpfn`
- verifies CUDA visibility

## Recommended Reproduction Order

### 1. Reproduce the strongest current result first

Branch:

- `codex/mv-tabpfn-view-router`

Commands:

```bash
source .venv-h200/bin/activate
python experiments/tabpfn_view_router/scripts/run_experiment.py \
  --split-seed 42 --seed 42 \
  --output experiments/tabpfn_view_router/reports/p0_full_h200
python experiments/tabpfn_view_router/scripts/run_experiment.py \
  --split-seed 42 --seed 41 \
  --output experiments/tabpfn_view_router/reports/p0_seed41_h200
python experiments/tabpfn_view_router/scripts/run_experiment.py \
  --split-seed 42 --seed 43 \
  --output experiments/tabpfn_view_router/reports/p0_seed43_h200
```

Expected reference band from local M4:

- `P0_FULL` mean test RMSE: about `0.3932`
- `P0_router` mean test RMSE: about `0.3790`

### 2. Reproduce aligned foundation baselines

Branch:

- `codex/tabpfn-california-baseline`

Commands:

```bash
source .venv-h200/bin/activate
python experiments/tab_foundation_compare/scripts/run_tabpfn_aligned.py --n-estimators 1
python experiments/tab_foundation_compare/scripts/run_tabpfn_aligned.py
```

Expected reference band from local M4:

- best single run: about `0.3891`
- 3-seed mean: about `0.3932`

### 3. Reproduce the strongest geo prior line

Branch:

- `codex/mv-tabr-gora-geo-retrieval`

Command pattern:

```bash
source .venv-h200/bin/activate
python experiments/mv_tabr_gora/scripts/run_geo_retrieval.py --seed 0
python experiments/mv_tabr_gora/scripts/run_geo_retrieval.py --seed 1
python experiments/mv_tabr_gora/scripts/run_geo_retrieval.py --seed 2
```

Expected local paired result:

- branch-local gain of about `+0.0055` RMSE over its paired raw baseline

## What To Change For GPU First

Minimal code changes before serious H200 runs:

1. Standardize device selection to `cpu/cuda/mps`.
2. Expose training `--seed` consistently in every runner.
3. Keep split seeds fixed when comparing model randomness.
4. Avoid branch-local hidden state in reports; write fresh GPU outputs to separate report directories.

## What Not To Change Yet

Do not do these before first GPU reproduction:

- refactor experiment folders
- merge branch-local lines into one mega-branch
- rewrite losses or architecture “while migrating”
- delete negative-result branches

The migration objective is reproducibility first, acceleration second.

## Why H200 Should Help

The H200 environment is most valuable for:

- multi-seed TabPFN and P0 sweeps
- larger candidate-pool or retrieval-temperature sweeps
- architecture ladders that were too expensive or noisy on M4
- future dynamic retrieval experiments that need more repetitions before drawing conclusions

The H200 move should therefore be used to:

- stabilize comparisons
- increase seed count
- run paired ablations faster

not to change the scientific question mid-migration.
