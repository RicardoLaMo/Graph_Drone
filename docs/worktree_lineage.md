# Worktree Lineage

This repo has two different kinds of experiment history:

1. **Tracked experiment families in the current checkout**
   - live under `experiments/`
   - should be treated as the canonical mainline tree for this checkout

2. **Branch-local research lines in sibling worktrees**
   - live under `.worktrees/` or `.claude/worktrees/`
   - may be newer, but are not automatically part of the current checkout

## Current Practical Mapping

### Current checkout — agent home
- path: `/Volumes/MacMini/Projects/Graph_Drone`
- branch: `feature/gora-v5-trust-routing`
- meaning: primary working checkout, agent home directory for all agents
- tracked experiment families:
  - `gora_tabular`
  - `mq_gora_v4`
  - `head_routing_v5`

### ~~Claude worktree: early v3 snapshot~~ [ARCHIVED — do not use as home]
- path: `.claude/worktrees/funny-davinci`
- branch: `claude/funny-davinci`
- role: early `gora_tabular` v3 snapshot at commit `1903f5a`
- status: **archived reference point only** — Claude works from the main checkout

### MV-TabR-GoRA core line
- path: `.worktrees/mv-tabr-gora`
- branch: `feature/mv-tabr-gora`
- result: A6f saved artifact **0.4063 RMSE** — not reproducible from fresh reruns (see note)
- role: California branch-local line with `A0..A6f` ablation ladder
- ⚠️ reproducibility note: fresh reruns of A6f from commit `c1ef39e` land at ~0.4295–0.4315
  (MPS non-determinism); 0.4063 artifact is orphaned — do not use as absolute target

### MV-TabR-GoRA B-series (retrieval quality: pool expansion + score biases)
- path: `.worktrees/mv-tabr-gora-rerank`
- branch: `feature/mv-tabr-gora-rerank`
- result: pool expansion K=24→96 **hurts** (+0.006 RMSE); score biases marginal
- finding: raw-space kNN can't find label-predictive far neighbors

### MV-TabR-GoRA A7a (iterative re-indexing: learned embedding-space kNN)
- path: `.worktrees/mv-tabr-gora-a7a-iterative-reindex`
- branch: `feature/mv-tabr-gora-a7a-iterative-reindex`
- result: probe ✅ (+0.011 retrieval quality), neural retrain ≈ neutral-to-negative
- Gemini-validated: weight mismatch bug fixed; post-fix evidence weaker
- finding: encoder z^v space is better for shallow retrieval, but model can't yet
  convert that into stable out-of-sample neural gains; need hybrid graph mixing

### MV-TabR-GoRA A7 candidate (earlier A7 candidate-pool attempt, superseded)
- path: `.worktrees/mv-tabr-gora-a7-candidate-rerank`
- branch: `feature/mv-tabr-gora-a7-candidate-rerank`
- role: exploratory; superseded by A7a iterative reindex

### MV-TabR-GoRA A8a (label-anchor residual prediction)
- path: `.worktrees/mv-tabr-gora-a8a-label-anchor`
- branch: `feature/mv-tabr-gora-a8a-label-anchor`
- result: A6f val=0.4510/test=0.4099 vs A8a val=0.4492/test=0.4156
- finding: val improves (−0.0018) but test worsens (+0.0057); confirms retrieval is the bottleneck

### MV-TabR-GoRA A9a (gated anchor trust)
- path: `.worktrees/mv-tabr-gora-a9a-gated-anchor`
- branch: `feature/mv-tabr-gora-a9a-gated-anchor`
- hypothesis: learned gate down-weights bad anchors; pred_v = sigmoid(gate(ctx_v)) * anchor_v + residual_v
- status: **implemented, ready to run** (smoke test ✅)

### MV-TabR-GoRA A9b (label-guided encoder pretraining)
- path: `.worktrees/mv-tabr-gora-a9b-guided-pretrain`
- branch: `feature/mv-tabr-gora-a9b-guided-pretrain`
- hypothesis: Stage 0 encoder pretraining → better kNN quality → better neural gains
- result: val=0.4882, test=0.4657 — clear regression; Stage 0 collapses direction-encoding metric structure
- finding: pretraining destroys T(z_i^v − z_j^v) space; label-predictive embeddings ≠ retrieval-useful embeddings

### MV-TabR-GoRA A10a (d_model 64→256 scale)
- path: `.worktrees/mv-tabr-gora-a10a-scale256`
- branch: `feature/mv-tabr-gora-a10a-scale256`
- hypothesis: architecture is capacity-bottlenecked; scaling d_model attacks the remaining gap directly
- status: **ready to run** (smoke test ✅, no code changes — use --d_model 256 flag)
- run: `python experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A6f --d_model 256 --output reports/a10a_d256`

### MV-TabR-GoRA A10b (dynamic kNN — co-evolving graph)
- path: `.worktrees/mv-tabr-gora-a10b-dynamic-knn`
- branch: `feature/mv-tabr-gora-a10b-dynamic-knn`
- hypothesis: rebuild per-view kNN from current z^v embeddings every 10 epochs — graph and encoder co-evolve (TabR-like)
- status: **ready to run** (smoke test ✅ — rebuild fires correctly, ~1.4s per rebuild)
- run: `python experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A6f A10b --rebuild_interval 10 --output reports/a10b_dynknn`

### MV-TabR-GoRA C1/E0 (consumption bottleneck diagnosis)
- path: `.worktrees/mv-tabr-gora-c1e0`
- branch: `feature/mv-tabr-gora-c1e0`
- base: `feature/mv-tabr-gora` (A6f, clean static-graph baseline)
- hypothesis: remaining gap to TabR is in consumption, not retrieval
  - C1 tests: does collapsing K=24 neighbours to one ctx_v too early hurt?
    MultiSlotAttention M=2: anchor query + learned per-slot offsets → gated sum
  - E0 tests: does single decoder head variance hurt?
    MultiHeadTaskHead M=4: 4 parallel 2-layer MLPs, averaged output
  - C1E0: both together (interaction check)
- static graph throughout — no retrieval changes
- smoke test: ✅ A6f/C1/E0/C1E0 all pass (param deltas verified)
- status: **scaffolding complete — ready for full run**
- run: `python .worktrees/mv-tabr-gora-c1e0/experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A6f C1 E0 C1E0`

### MV-TabR-GoRA E1+D0 (gated decoder + deep view encoder)
- path: `.worktrees/mv-tabr-gora-e1d0`
- branch: `feature/mv-tabr-gora-e1d0`
- base: `feature/mv-tabr-gora-c1e0` (inherits A6f/C1/E0/C1E0 baselines)
- hypothesis:
  - E1: gate M=4 decoder head weights on {pi, mean_J} → routing co-trains with head selection
  - D0: 2-layer GELU ViewEncoder → GEO view discovers implicit cluster structure
  - Motivated by geo segmentation experiment: H5 structure-only priors (0.4251) show geometry carries signal
- ablations: E1 (gated heads), D0 (deep encoder), E1D0 (both)
- smoke test: ✅ A6f/E1/D0/E1D0 pass, param deltas verified (+12K/+16K/+29K)
- result: A6f=0.4081 | E0=0.4091 | E1=0.4115 | D0=0.4161 | E1D0=0.4121 (test RMSE)
- finding: consumption side exhausted — A6f stays best; see findings for root cause analysis
- status: **complete**

### MV-TabR-GoRA G2 geo pool-mix + G0 cohort residual (Codex branch)
- path: `.worktrees/mv-tabr-gora-g0g1-cohort-residual`
- branch: `codex/mv-tabr-gora-g0g1-cohort-residual`
- experiments:
  - **G2_geo_poolmix96** (geo retrieval): replace k_seg=12 of K=24 FULL+GEO neighbors
    with nearest same-KMeans-96-segment training points; same total budget; no arch change
  - **G3_random_poolmix96** (control): same budget, random replacements → hurts
  - **G0_cohort_residual** (failed): residual target + segment_mean in GEO combined → ep=3 collapse
- **G2 results (3-seed paired, canonical Huber pipeline)**:
  - A6f_raw mean: 0.4295 ± 0.0015 test RMSE (reproducible fresh baseline)
  - G2 mean: **0.4241 ± 0.0021** test RMSE
  - paired gain: **+0.0055 ± 0.0010** (consistent across all 3 seeds; G3 control hurts → community structure specifically helps)
- **G2 is the new reproducible champion** at 0.4241 mean test RMSE
- G0 finding: two confounders (residual target + GEO feature change combined); needs clean split
- **G0a** (scaffolded, ready to run): `append_seg_mean_to_geo=False` → residual target only; kNN geometry identical to A6f
- **G0c** (scaffolded, ready to run): G0a + `normalize_by_residual_std=True` → Huber delta=1.0 calibrated to residual std (~0.83)
- run: `cd .worktrees/mv-tabr-gora-g0g1-cohort-residual && python experiments/mv_tabr_gora/scripts/run_cohort_residual.py --variants raw g0a g0c`
- status: **G2 champion registered; G0a/G0c scaffolded (smoke ✅), full run pending**

### TabPFN View Router (per-view TabPFN experts + GoRA-style routing)
- path: `.worktrees/mv-tabpfn-view-router`
- branch: `codex/mv-tabpfn-view-router`
- hypothesis: 4 per-view TabPFN experts (FULL/GEO/SOCIO/LOWRANK) + learned soft router on
  {sigma2_v, J_flat, mean_J} outperforms single global TabPFN
- experiments:
  - **P0_FULL** (single global TabPFN): mean=0.3932 ± 0.0038
  - **P0_uniform** (equal-weight mix): mean=0.4379 — worse than P0_FULL
  - **P0_sigma2** (inv-sigma2, no val labels): mean=0.4380 — worse than P0_FULL
  - **P0_gora** (GoRA analytical formula, no val labels): mean=0.5196 — **fails** (too aggressive winner-take-all)
  - **P0_router** (learned router, val meta-training): mean=0.3790 ± 0.0016
  - **P0_crossfit** (5-fold OOF, clean protocol): mean=0.3790 ± 0.0016 — **validates P0_router**
- **Key findings:**
  - P0_gora failure: raw sigma2/J formula → winner-take-all, picks wrong views; routing signal requires learning
  - P0_crossfit = P0_router at all 3 seeds → router not memorizing val labels; test gain is real
  - 0.3790 < TabR (0.3829) < P0_FULL (0.3932): per-view specialisation + routing beats single global model
- **status: complete (3-seed, clean OOF validation)**
- report: `experiments/tabpfn_view_router/reports/p0_multiseed_ab.md`

### MV-TabR-GoRA DRST (dynamic candidate-pool retriever + EdgeMLP encoding)
- path: `.worktrees/mv-tabr-gora-drst`
- branch: `feature/mv-tabr-gora-drst`
- base: `feature/mv-tabr-gora` (A6f codebase — avoids A9/A10 cruft)
- hypothesis: Q·K_ema re-ranking inside fixed pool (C=96) + EdgeMLP edge encoding
  fixes the static-direction-encoding / dynamic-graph incompatibility
- ablations: B1 (cand-pool rerank), B2 (+ EdgeMLP), B3 (+ CrossViewMixer)
- smoke test: ✅ B1 and B2 pass (KL loss visible, no shape errors)
- status: **scaffolding complete — ready for full run**
- run: `cd .worktrees/mv-tabr-gora-drst && python experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A6f B1`

## Agent Assignments

| Agent | Home | Writes to | Role |
|---|---|---|---|
| Claude | main checkout (`feature/gora-v5-trust-routing`) | `.claude/` | Implementation |
| Gemini | reads only | `.gemini-cross-checks/` | Cross-validation |
| Codex | TBD | `.codex/` (reserved) | TBD |

For details see `docs/agents.md`.

## How Agents Should Use This

Before inferring experiment lineage:
1. read `docs/worktree_registry.json`
2. read this file
3. distinguish:
   - current checkout facts
   - sibling worktree facts (research lines)

Do not describe sibling worktree experiments as part of the current checkout unless you say so explicitly.

Do not treat `claude/funny-davinci` as the active working location — it is archived.

## Refresh Rule

After adding/removing worktrees or switching the main checkout branch, rerun:

```bash
source .venv/bin/activate
python scripts/audit_worktrees.py --write docs/worktree_registry.json
```

Then commit `docs/worktree_registry.json` with: `chore: refresh worktree registry`.
