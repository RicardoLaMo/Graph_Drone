# Graph_Drone Agent Notes

## Workspace Identity
- Repo root and current working directory: `/Volumes/MacMini/Projects/Graph_Drone`
- GitHub remote: `https://github.com/RicardoLaMo/Graph_Drone.git`
- Repository README title is currently `GraphHNM`, but the folder and git repository name are `Graph_Drone`.
- Current checkout branch: `feature/gora-v5-trust-routing` (verified at commit 848bc32)

## Worktree Source Of Truth
- Before inferring experiment lineage, read:
  - `docs/worktree_lineage.md`
  - `docs/worktree_registry.json`
- The registry distinguishes:
  - current checkout facts
  - sibling worktree facts
  - tracked experiment families vs branch-local research lines
- Do not describe sibling worktree experiments as part of the current checkout unless stated explicitly.

## Working Environment
- Python project with a checked-in local environment at `.venv` using Python 3.12.
- Core dependencies in `requirements.txt`: NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn, PyTorch, torchvision, torchaudio, OpenML, JupyterLab, and `tqdm`.
- The repo is set up for Apple Silicon acceleration when available. `src/verify_environment.py` checks `torch.backends.mps` and falls back to CPU.
- Common activation pattern:
  ```bash
  source .venv/bin/activate
  ```

## Repo Layout
- `experiments/_shared_routing_curvature/src/`: shared routing, observer, graph-building, and evaluation utilities.
- `experiments/california_routing_curvature/`: tracked California Housing routing-curvature experiment.
- `experiments/mnist_routing_curvature/`: tracked MNIST-784 routing-curvature experiment.
- `src/verify_environment.py`: environment and MPS verification script.
- `docs/git_doe_strategy.md`: local branching guidance for design-of-experiments work.
- `data/`, `models/`: ignored by default via `.gitignore` except for `.gitkeep`.

## Canonical Experiment Entry Points
- California:
  ```bash
  source .venv/bin/activate
  python experiments/california_routing_curvature/scripts/run_experiment.py
  ```
- MNIST subset:
  ```bash
  source .venv/bin/activate
  python experiments/mnist_routing_curvature/scripts/run_experiment.py
  ```
- MNIST full:
  ```bash
  source .venv/bin/activate
  python experiments/mnist_routing_curvature/scripts/run_experiment.py --full
  ```

## Current Experiment Design
- The active tracked branch tests the hypothesis that curvature is more useful as a routing prior than as a direct predictive feature.
- Shared routing components live in `experiments/_shared_routing_curvature/src/routing_models.py`:
  - `ObserverRouter`: maps observer features to per-row view weights plus an isolation/interaction gate.
  - `ViewCombiner`: blends routed view representations.
  - `NoRouterCombiner`: learned ablation without observer routing.
- Shared observer features are computed in `experiments/_shared_routing_curvature/src/observer.py`:
  - local PCA residual curvature (`kappa`)
  - LID proxy
  - LOF
  - local density

## Current Tracked Results
- California report: `experiments/california_routing_curvature/reports/report.md`
  - Best tracked graph/routing model in the current report is `C8_Routed` with RMSE `0.5250`.
  - Report verdict: routing helped on California, but the isolation/interaction gate did not.
- MNIST report: `experiments/mnist_routing_curvature/reports/report.md`
  - Best tracked graph model in the current report is `M5_PCA` with accuracy `0.942`.
  - Report verdict: routing did not meaningfully exploit hidden geometry on MNIST.

## Git And Tree State
- Local branches present during audit:
  - `main`
  - `exp/algo/benchmark-setup`
  - `exp/research/curvature-hypothesis`
  - `feature/mnist-geometry-sanity-check`
  - `feature/routing-curvature-dual-datasets`
- No git submodules were configured during the audit.
- There is no tracked `.github/` directory in the current checkout.

## Important Current Caveats
- The IDE context referenced paths such as `experiments/california_curvature/run_all.py` and `experiments/california_curvature/06_curvature_models.py`, but those paths do not exist in the current checkout.
- The current tracked California experiment directory is `experiments/california_routing_curvature/`.
- There is an untracked workspace at `experiments/mnist_geometry_sanity/`, but during this audit it appeared to contain artifacts and `__pycache__` entries without the corresponding source `.py` files or report markdown in the working tree.
- Additional untracked content existed at the repo root in `artifacts/` during the audit.

## Agent Operating Guidance
- Verify whether a path from the editor actually exists before assuming it is part of the current branch.
- Prefer the tracked routing-curvature experiment directories over stale `california_curvature` references.
- Do not delete or overwrite untracked experiment outputs unless explicitly asked.
- When validating the environment, use:
  ```bash
  source .venv/bin/activate
  python src/verify_environment.py
  ```

## GoRA-Tabular v1-v3 Learning
- The active MQ-GoRA / GoRA-Tabular lineage is under `experiments/gora_tabular/`, not the routing-curvature folders.
- Commit lineage to remember:
  - `df3ea03`: v1 GoRA-Tabular, routing inside attention, models `B0/B1/G0/G1/G2/G3/G4`
  - `1d5ce04` then `9b12459`: v2 joint-view kNN plus disagreement regularizer, models `G5/G6/G3p`
  - `1903f5a` then `f65fc20` then `fb53bf9`: v3 MQ-GoRA, models `G7/G8/G9/G10`, then bugfix/perf stabilization
- v1/v2/v3 runner entrypoints:
  - `experiments/gora_tabular/scripts/run_gora.py`
  - `experiments/gora_tabular/scripts/run_gora_v2.py`
  - `experiments/gora_tabular/scripts/run_gora_v3.py`

## v3 Data And Views
- California v3:
  - dataset: `fetch_california_housing()`
  - preprocessing: `float32`, `log1p` on columns `[2, 4]`, then `RobustScaler`
  - split: 70/15/15 via two `train_test_split` calls, not stratified
  - views: `FULL`, `GEO = X[:, [6, 7]]`, `SOCIO = X[:, [0, 1, 2, 3, 4]]`, `LOWRANK = PCA(4)`
  - primary v1 carry-forward view for `G2`: `GEO`
- MNIST v3:
  - dataset: OpenML `mnist_784`
  - preprocessing: optional stratified subset, then `StandardScaler`
  - split: stratified 70/15/15
  - views: `FULL`, `BLOCK` (16 block means from 4x4 grid of 7x7 patches), `PCA = PCA(50)`
  - primary v1 carry-forward view for `G2`: `PCA`

## v3 Observer And Topology Semantics
- Observer vector `g_i` is routing-only, not appended to predictive features.
- Implemented observer channels are:
  - `kappa`: local PCA residual non-flatness
  - `lid`: local intrinsic dimensionality proxy
  - `lof`: local outlier factor
  - per-view normalized mean kNN distance proxies
- v2/v3 routing semantics depend on `build_joint_neighbourhood()`:
  - each view nominates `k_per_view=5` neighbors independently
  - union pool size is at most `M * 5`
  - `view_mask` marks whether a pooled neighbor belongs to each view
  - `agree_score` is fraction of pooled neighbors shared by at least two views
- This matters: peaked `pi` means isolate one view's neighborhood; flat `pi` means interact across views.

## v3 Model Family
- Baselines:
  - `B0`: MLP
  - `B1`: HistGradientBoosting
  - `B2`: TabPFN, only when constraints allow
- v1 family:
  - `G0`: standard transformer over fetched neighborhood, no graph bias
  - `G1`: single fixed-view routing
  - `G2`: original GoRA routing, `pi` and `tau` inside attention logits
  - `G3`: uniform `pi` ablation
  - `G4`: shuffled-geometry ablation
- v2 family:
  - `G5`: `G2` plus joint-view kNN pool
  - `G6`: `G5` plus disagreement-aligned routing regularizer
- v3 family uses `MQGoraTransformer` flags:
  - `G7`: ViewSpecificEmbedder + ManifoldReader avg-pool + RichMoERouter
  - `G8`: `G7` + LabelContextEncoder
  - `G9`: `G8` + ManifoldTeacher query `z_anc`
  - `G10`: `G9` + AlphaGate prediction fusion
- Important semantic note: v3 has `pi`, `tau`, and `alpha`; it does not have `beta`.

## v3 Objectives
- Teacher objective (`ManifoldTeacher`):
  - `L_agree`: predict normalized `agree_score`
  - `L_label`: predict per-view label centroids
  - `L_centroid`: reconstruct primary-view neighborhood centroid in feature space
- Student objective:
  - task loss: `MSE` for California, `CrossEntropy` for MNIST
  - optional v2 routing regularizer aligning routing entropy to `agree_score`
  - optional v3 alpha auxiliary loss `MSE(alpha, agree_score)`
- Training detail that matters for interpretation:
  - label context is available during train batches
  - validation and inference explicitly drop `lbl_nei`
  - this train/inference mismatch is mild for MNIST one-hot labels and riskier for California continuous targets

## Saved v3 Results To Anchor Future Work
- California saved metrics (`experiments/gora_tabular/artifacts/california_v3_metrics.csv`):
  - `B1_HGBR=0.4433`
  - `G2_GoRA_v1=0.4546`
  - `G7_RichCtx=0.4929`
  - `G8_LabelCtx=0.5169`
  - `G9_Teacher=0.5099`
  - `G10_Full=0.5209`
- MNIST saved metrics (`experiments/gora_tabular/artifacts/mnist_v3_metrics.csv`):
  - `B1_HGBR=0.9580`
  - `G2_GoRA_v1=0.9300`
  - `G7_RichCtx=0.9267`
  - `G8_LabelCtx=0.9307`
  - `G9_Teacher=0.9347`
  - `G10_Full=0.9380`
- Saved report conclusions:
  - California: every v3 rich-context variant is worse than carried-forward `G2`; GEO remains dominant across heads.
  - MNIST: `G8 -> G9 -> G10` is a real monotonic improvement ladder; `G10` partially restores head diversity with one head preferring `FULL`.
  - Neither dataset beats `B1_HGBR` in the saved v3 outputs.
- Agreement regime difference is important:
  - California mean `agree_score` is about `0.112`, indicating strong view disagreement
  - MNIST mean `agree_score` is about `0.372`, indicating more overlap between views
- Per-bin saved outputs show:
  - California high-curvature bin is where GoRA's relative routing signal is strongest, but it still trails stronger baselines
  - MNIST `G10` improves most in the high-curvature bin, but still trails `B1`

## Practical Interpretation For Future Work
- Do not describe v3 as a general win. The actual repo evidence is narrower:
  - v1 proved routing has signal relative to uniform/random routing
  - v2 showed joint-kNN topology did not help and often blurred specialization
  - v3 improved MNIST incrementally through richer neighborhood reading, but harmed California
- California failure mode to keep in mind:
  - persistent GEO dominance
  - low inter-view agreement
  - likely overfitting from continuous label context and its train/inference mismatch
- MNIST success mode to keep in mind:
  - view structure is real enough for partial head specialization
  - teacher-guided reading and alpha fusion helped incrementally
  - gains are real inside the GoRA family but still below the strongest tabular baseline

## Latest MV-TabR-GoRA California Research Line (Branch-Local)

The most recent California research is in sibling worktrees under `feature/mv-tabr-gora*` branches:

**Core Lineage:**
1. `feature/mv-tabr-gora` (`.worktrees/mv-tabr-gora`): A0-A6 ablation ladder, A6f champion at 0.4063 RMSE
2. `feature/mv-tabr-gora-rerank` (`.worktrees/mv-tabr-gora-rerank`): B-series (pool expansion + score biases); found larger K hurts
3. `feature/mv-tabr-gora-a7a-iterative-reindex` (`.worktrees/mv-tabr-gora-a7a-iterative-reindex`): A7a iterative re-indexing (learned embedding-space kNN reindex)

**Key Finding (B-series → A7a transition):**
- B-series: Pool expansion (K 24→96) hurt performance by +0.006 RMSE; score biases partially recover but can't overcome dilution
- Root cause: Raw-space kNN finds geometrically-close but not label-predictive far neighbors
- A7a hypothesis: Train A6f encoder → extract embeddings → rebuild kNN in embedding space → retrain A6f on better-quality neighbors
- A7a result: Check `feature/mv-tabr-gora-a7a-iterative-reindex` branch for iterative reindex probe and analysis

**Before reasoning about current state of A-series experiments:**
- Read: `docs/worktree_lineage.md` and `docs/worktree_registry.json`
- Do NOT assume sibling worktree experiments are part of current checkout unless explicitly stated
- Distinguish: current checkout (`feature/gora-v5-trust-routing`) vs branch-local California research lines (`feature/mv-tabr-gora*`)
