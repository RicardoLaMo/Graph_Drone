# MNIST-784: Routing Curvature Experiment

> ⚠️ **Isolation notice**: Completely isolated from all prior MNIST experiments.

## Branch
`feature/routing-curvature-dual-datasets`

## Hypothesis
Curvature as a **routing prior** — deciding which view to trust per row — should exploit MNIST's hidden pixel geometry better than using curvature as a direct feature.

## Run
```bash
source .venv/bin/activate
python experiments/mnist_routing_curvature/scripts/run_experiment.py          # 10k subset
python experiments/mnist_routing_curvature/scripts/run_experiment.py --full   # 70k
```

## Models
| ID | Model | Role |
|----|-------|------|
| M0 | Majority | Floor |
| M1 | MLP | Tabular |
| M2 | HGBR | Strong tabular |
| M3 | XGBoost | Strong tabular |
| M4 | TabPFN | Subset (documented) |
| M5 | GraphSAGE FULL | Single-view |
| M6 | Uniform ensemble | Multi-view |
| M7 | Learned combiner (no observer) | Ablation |
| M8 | Observer-routed combiner | **Main test** |
| M9 | Observer-routed + isolation/interaction | **Full routing** |

## Warning signs
- M8/M9 no better than M7 → routing adds nothing
- All graph models below MLP → hidden geometry not exploited
