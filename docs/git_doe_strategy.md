# Git Tree Strategy for Design of Experiments (DOE)

When testing different algorithms or graphs on the `Graph_Drone` project, we want to maintain an organized repository where experimental features don't destabilize the `main` branch. 

## 1. Branch Naming Conventions
Use descriptive prefixes for your branches to quickly identify the nature of the experiment:

- **Algorithm Experiments**: `exp/algo/<algorithm-name>` 
  *Example: `exp/algo/z-gw-attention`*
- **Data/Graph Tests**: `exp/data/<dataset-topic>`
  *Example: `exp/data/california-housing`*
- **Hyperparameter tuning**: `exp/hyper/<parameter-tune>`
  *Example: `exp/hyper/learning-rate-decay`*

## 2. Typical Workflow for an Experiment
1. **Sync Main**: Ensure your local `main` branch is up to date:
   ```bash
   git switch main
   git pull origin main
   ```
2. **Create Branch**: Create a new branch for the experiment:
   ```bash
   git switch -c exp/algo/new-model
   ```
3. **Iterate & Commit**: Write code, run tests, save models, and commit early and often.
   ```bash
   git add .
   git commit -m "feat: added simple GNN model for baseline testing"
   ```
4. **Push Work**: Push your branch to GitHub to back up your experiment.
   ```bash
   git push -u origin exp/algo/new-model
   ```

## 3. Completing an Experiment (Validation Gate)
Once the experiment has concluded:
- If **successful** and you want to merge it into the production codebase, create a Pull Request (PR) to merge into `main`. Ensure all unit tests pass before merging.
- If **unsuccessful** or just a sandbox test, leave the branch as a historical record of what didn't work. Do not delete it unless it's pure clutter.

## 4. Using the `agile-workspace-sync` Skill
If multiple agents are collaborating, the `agile-workspace-sync` skill can be utilized to automatically check out branches, safely sync with remote, and manage PRs without disrupting the primary environment. Agents can spin up temporary features or fixes, validate them against the `tests/` suite, and push them gracefully.

## 5. Artifacts and Results 
Save outputs of an experiment (e.g., plots, tables, metrics) in `docs/experiments/` or `notebooks/`. Track any relevant serialized models in `models/` but remember they are git-ignored by default to keep the repository light. If a model *must* be tracked, use Git LFS or a cloud storage bucket.
