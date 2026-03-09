# California Geo Segmentation

This experiment tests whether latent "zip/community" structure can be recovered
from California Housing latitude/longitude alone, using train-safe geo
segmentation priors.

It intentionally starts shallow:

- derive pseudo-neighborhood segments from raw latitude/longitude
- compute train-only segment target statistics
- test whether those priors improve over the raw-feature HGBR baseline

The goal is to validate geo segmentation signal before pushing it into
MV-TabR-GoRA retrieval or routing.

Run:

```bash
cd /Volumes/MacMini/Projects/Graph_Drone/.worktrees/mv-geo-segmentation-priors
source ../../../.venv/bin/activate
python experiments/california_geo_segmentation/scripts/run_experiment.py
```

Smoke:

```bash
cd /Volumes/MacMini/Projects/Graph_Drone/.worktrees/mv-geo-segmentation-priors
source ../../../.venv/bin/activate
python experiments/california_geo_segmentation/scripts/run_experiment.py --smoke
```
