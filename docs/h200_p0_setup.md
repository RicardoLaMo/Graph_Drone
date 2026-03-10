# H200 P0 Setup

This branch is self-contained for H200 bring-up of the `P0` TabPFN view-router line and now assumes the validated single shared H200 environment model.

Scope:

- branch: `codex/mv-tabpfn-view-router`
- experiment: `experiments/tabpfn_view_router/`
- objective: boot the H200 runtime, validate PyTorch/TabPFN/Hugging Face/OpenML access, and keep git/Codex operations lightweight and reproducible

## Files

- `migration/bootstrap_h200.sh`: creates the canonical `.venv-h200` conda-prefix env with CUDA PyTorch, GPU FAISS, TabPFN, and CLI extras
- `migration/activate_h200_env.sh`: activates the H200 env and applies runtime guardrails
- `migration/setup_git_and_cli.sh`: configures git defaults and checks GitHub/Hugging Face CLI auth
- `migration/validate_h200_stack.py`: smoke checks for torch, git, Hugging Face, and OpenML
- `migration/codex_project_skills_manifest.json`: minimal Codex skill subset to carry over to the H200 host

## Bootstrap Order

1. Clone and checkout this branch.

```bash
git clone https://github.com/RicardoLaMo/Graph_Drone.git
cd Graph_Drone
git checkout codex/mv-tabpfn-view-router
```

2. Build the Python environment.

```bash
bash migration/bootstrap_h200.sh
```

3. Activate the runtime.

```bash
source migration/activate_h200_env.sh
```

Notes:

- `GRAPH_DRONE_GPU_COUNT` now defaults to `8`, so the activation wrapper exposes the whole H200 box unless you narrow it
- `GRAPH_DRONE_GPU_POOL` can pin the physical GPU order explicitly, for example `7,6,5,4,3,2,1,0`
- `GRAPHDRONE_OPENML_GRAPHDRONE_GPU_SPAN=1` is the max-throughput mixed-suite default; set it to `4` for GraphDrone-only sweeps that should run as two 4-GPU jobs on an 8-GPU node
- `GRAPHDRONE_OPENML_MAX_CONCURRENT_JOBS` defaults to `8`, so the OpenML queue will pack one job per free GPU when enough tasks are pending
- bootstrap now uses the validated conda path: `torch 2.6.x`, `CUDA 12.6`, `faiss-gpu-cuvs 1.14.1`
- the target is one shared `.venv-h200` env, not separate `.venv-foundation*` variants
- Hugging Face cache and TabPFN model cache are kept under `.cache/` in the repo by default

## Git And Auth

Run:

```bash
bash migration/setup_git_and_cli.sh
```

Optional environment variables before running it:

```bash
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="you@example.com"
```

What it does:

- sets safe repo-local git defaults (`fetch.prune`, `pull.rebase`, `push.autoSetupRemote`, `rerere.enabled`)
- registers the repo as a global safe directory
- reports whether `gh` and `hf` are installed and authenticated

Important:

- `gh` is not installed by the repo bootstrap script because it is typically managed at the OS level
- if `gh` is missing, install it with the host package manager and run `gh auth login`
- if Hugging Face auth is missing, run `hf auth login`
- for transient validation without persisting credentials, export `HF_TOKEN` for the current shell before `--hf-smoke`
- if Hugging Face CLI reports offline mode, unset `HF_HUB_OFFLINE` before retrying auth
- plain git over the repo's SSH remote is validated separately from the optional `gh` CLI

## Shared-Env Model

- repo-root canonical env: `.venv-h200`
- sibling worktrees should point at that env via local shims when needed
- `migration/activate_h200_env.sh` supports both classic virtualenv activation and conda-prefix layouts
- do not create new branch-local Python envs unless the shared H200 env is proven insufficient

## Validation

Minimal runtime check:

```bash
python migration/validate_h200_stack.py --torch-smoke --git-smoke
```

Hugging Face auth check:

```bash
python migration/validate_h200_stack.py --hf-smoke
```

Transient token-based Hugging Face check:

```bash
export HF_TOKEN=...
python migration/validate_h200_stack.py --hf-smoke
unset HF_TOKEN
```

OpenML metadata check:

```bash
python migration/validate_h200_stack.py --openml-smoke
```

OpenML data-path check using the branch-relevant California dataset:

```bash
python migration/validate_h200_stack.py --openml-smoke --openml-download --openml-dataset-id 44024
```

## P0 OpenML Run

Branch-local multiseed launcher:

```bash
bash migration/run_p0_openml_multiseed.sh
```

Summary after the runs:

```bash
python migration/summarize_p0_openml.py
```

Useful overrides:

```bash
P0_SEEDS="41 42 43" \
P0_GPU_GROUPS="7,6,5,4;3,2,1,0" \
bash migration/run_p0_openml_multiseed.sh
```

```bash
P0_SMOKE=1 \
P0_GPU_GROUPS="7,6,5,4" \
bash migration/run_p0_openml_multiseed.sh
```

## Operational Reflections

- For `P0`, one run has four independent view experts, so `4` GPUs is the practical per-run ceiling in `per_view` mode.
- To use all `8` H200s efficiently, the best pattern is two concurrent seeds:
  `7,6,5,4` and `3,2,1,0`.
- The activation script now defaults to high-index GPUs first so ad hoc runs avoid the more crowded low-index devices.
- The validated shared env on this host is `torch 2.6.0`, `tabpfn 6.3.1`, `openml 0.15.1`, and GPU `faiss 1.14.1`.
- OpenML California should use `did=44024` on this branch. Its feature matrix matches `fetch_california_housing()` exactly.
- The OpenML target in `did=44024` is `log1p(price)`, so the loader converts it back with `expm1` before scoring.
- Git over the SSH remote is sufficient for normal branch work. `gh` is helpful, but not required for `git fetch/pull/push`.
- Hugging Face validation is safest with a transient `HF_TOKEN` in the shell unless you explicitly want persistent CLI auth on the host.

## Minimal Codex Skill Carry-Over

Do not port the full local skill directory to H200 by default.

Use only the branch-relevant subset listed in:

- `migration/codex_project_skills_manifest.json`

Current minimal set:

- `h200-gpu-ops`
- `git-codebase-ops`
- `git-tree-manager`
- `experiment-design-tracker`
- `results-manager`
- `gemini-validator-reconciler` (optional)

## Acceptance Checklist

- `.venv-h200` exists and imports `torch`, `tabpfn`, `huggingface_hub`, and `openml`
- `source migration/activate_h200_env.sh` completes without errors
- `python migration/validate_h200_stack.py --torch-smoke --git-smoke` passes
- `python migration/validate_h200_stack.py --hf-smoke` confirms Hugging Face CLI availability
- `python migration/validate_h200_stack.py --openml-smoke --openml-download --openml-dataset-id 44024` succeeds on the target host
- `git ls-remote origin HEAD` succeeds; set `user.name` and `user.email` before committing
- `gh auth status` is optional but recommended if you want the GitHub CLI on the host
- `hf auth whoami` or `HF_TOKEN=... python migration/validate_h200_stack.py --hf-smoke` succeeds before long runs
- `bash migration/run_p0_openml_multiseed.sh` and `python migration/summarize_p0_openml.py` complete without manual path edits
