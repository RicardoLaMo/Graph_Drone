---
name: gd-bench
description: Pre-flight check + smart benchmark execution. Shows exactly what will be reused from cache vs re-run, confirms version, inspects GPU, and prints the right command to run. Prevents redundant TabPFN re-runs and stale GD results.
argument-hint: "[regression|classification|both|quick]  default: both"
allowed-tools: Read, Bash, Grep
---

$ARGUMENTS is the engine scope: `regression`, `classification`, `both`, or `quick`. Default: `both`.

## Step 1: Read version string (source of truth = the script)

```
!`grep "GRAPHDRONE_VERSION" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/scripts/run_geopoe_benchmark.py | head -3`
```

Record the current `GRAPHDRONE_VERSION`. This is the version that will be used to key GD cache entries.

## Step 2: Inspect the cache — what will run vs reuse

```
!`ls -la /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache/ 2>/dev/null | head -60 || echo "Cache directory empty or missing"`
```

For each file in the cache directory, the naming pattern is `<dataset>__fold<N>__<method>.json`.

- Files matching `*__tabpfn.json`: baselines, keyed with `baseline_v1` — **always reusable across GD version bumps**
- Files matching `*__graphdrone.json`: read the `cache_key` field inside each file

```
!`for f in /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache/*__graphdrone.json; do echo "$(basename $f): $(python3 -c "import json,sys; d=json.load(open('$f')); print(d.get('cache_key','?')[:8], d.get('status','?'))" 2>/dev/null)"; done 2>/dev/null | head -30 || echo "No graphdrone cache files found"`
```

Report a 3-column summary:
```
Dataset+Fold          TabPFN cache    GD cache (vs current version)
california fold=0     REUSE           RE-RUN  (stale: old_version → new_version)
california fold=1     REUSE           CACHED
...
```

Count: "X/18 TabPFN tasks cached. Y/18 GD tasks cached for current version. Z GD tasks will re-run."

## Step 3: GPU pre-flight

```
!`nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"`
```

```
!`nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>/dev/null || echo "No active GPU processes"`
```

Report:
- GPU model and total VRAM
- Free VRAM right now
- Whether free VRAM is sufficient for the planned run (TabPFN + 4 GD experts ≈ 8–12 GB peak; H200 = 80–141 GB → can run 4–10 dataset tasks in parallel if scripts supported it)
- Flag any other processes occupying GPU memory

## Step 4: Version bump check

Ask: **Has any model code changed since the last benchmark run without a version bump?**

```
!`cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research && git log --oneline --since="3 days ago" -- src/ scripts/run_geopoe_benchmark.py | head -20`
```

If any commits to `src/` exist that are **newer** than the last version bump commit, warn:
> "MODEL CODE WAS CHANGED after the last GRAPHDRONE_VERSION bump. The cache will serve stale GD results unless you bump the version string."

If it looks clean, confirm: "Version string is current. GD cache entries with this version are valid."

## Step 5: Print the exact command to run

Based on $ARGUMENTS, print the correct command:

```bash
cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research

# What will actually re-run given current cache state:
# TabPFN: 0 tasks (all cached with baseline_v1)
# GD: <Z> tasks (uncached for version <GRAPHDRONE_VERSION>)

PYTHONPATH=src python scripts/run_geopoe_benchmark.py \
  --tasks <regression|classification|all> \
  --folds 0 1 2 \
  --cache-dir eval/geopoe_cache \
  --output-dir eval/geopoe_benchmark
```

**H200 note**: The script runs datasets sequentially. If Z > 6 re-runs are needed and free VRAM > 40 GB, consider running regression and classification in parallel in two terminals — they use independent cache files and non-overlapping GPU peaks:
```bash
# Terminal 1:
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks regression --folds 0 1 2 &

# Terminal 2:
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks classification --folds 0 1 2 &
```

## Step 6: Baseline coverage check

If any TabPFN cache files are **missing** (new dataset added, or cache was wiped), flag them specifically:
> "WARNING: TabPFN baseline missing for [dataset, fold]. This will re-run and may take time. Consider running baselines-only first: `--datasets <name> --methods tabpfn` (if supported) or let the full run populate them once."

Once TabPFN caches are populated with `baseline_v1` key they will never re-run again regardless of GD version bumps.
