# California Housing: Routing Curvature Experiment

> ⚠️ **Isolation notice**: This experiment is completely isolated from prior California Housing experiments. No prior files modified.

## Branch
`feature/routing-curvature-dual-datasets`

## Hypothesis
Curvature is NOT a direct predictor. It is a **routing prior** that guides which row-level view to trust (isolation) or whether to fuse views (interaction).

## Run
```bash
source .venv/bin/activate
python experiments/california_routing_curvature/scripts/run_experiment.py
```

## Models
| ID | Model | Role |
|----|-------|------|
| C0 | Mean | Floor |
| C1 | MLP | Tabular |
| C2 | HGBR | Strong tabular |
| C3 | XGBoost | Strong tabular |
| C4 | TabPFN | Strong tabular (subset) |
| C5 | GraphSAGE FULL | Single-view graph |
| C6 | Uniform ensemble | Multi-view, no learning |
| C7 | Learned combiner (no observer) | Multi-view ablation |
| C8 | Observer-routed combiner | **Main test** |
| C9 | Observer-routed + isolation/interaction | **Full routing** |

## Warning signs
- C8/C9 no better than C7 → routing adds nothing
- C7 no better than C5 → multi-view adds nothing
- All graph models trail C2/C3 significantly → graph on tabular is premature
