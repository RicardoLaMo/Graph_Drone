---
name: gd-cache
description: Inspect and manage the benchmark cache. Shows what's cached per method+version, identifies stale GD entries, and protects baseline caches from accidental deletion. Run before any benchmark to understand cache state.
argument-hint: "[status|prune-gd|show-baselines]  default: status"
allowed-tools: Read, Bash, Grep
---

$ARGUMENTS is `status` (default), `prune-gd` (remove stale GD caches only), or `show-baselines` (detail TabPFN cached results).

## Step 1: Read current version

```
!`grep "GRAPHDRONE_VERSION" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/scripts/run_geopoe_benchmark.py | head -2`
```

## Step 2: Scan all cache files

```
!`ls /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache/*.json 2>/dev/null | wc -l && echo "files in cache" || echo "Cache directory empty or missing"`
```

```
!`for f in /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache/*.json; do [ -f "$f" ] && python3 -c "
import json, sys, os
d = json.load(open('$f'))
name = os.path.basename('$f')
key = d.get('cache_key', '?')[:8]
status = d.get('status', '?')
method = 'tabpfn' if 'tabpfn' in name else 'graphdrone' if 'graphdrone' in name else 'unknown'
elapsed = d.get('elapsed', 0) or 0
print(f'{name:55s}  {method:12s}  {key}  {status:6s}  {elapsed:.0f}s')
" 2>/dev/null; done | sort`
```

## Step 3: Build the status report

### 3a. Baseline (TabPFN) cache summary

TabPFN cache entries use key `baseline_v1` — **version-independent**. They should NEVER be deleted unless the TabPFN library version changes.

Count:
- Total TabPFN cache files expected: 18 (6 regression + 6 classification datasets × 3 folds) + possible quick/partial runs
- Files present: N
- Files missing: list them
- Any failed: list them (status=fail means TabPFN crashed on that dataset/fold)

### 3b. GraphDrone cache summary

GD cache entries are keyed by `GRAPHDRONE_VERSION`. Entries from old versions are stale.

For each GD cache file, extract the embedded `cache_key` and check whether it matches the current version's expected key (recompute: `sha256(f"{dataset}|{fold}|graphdrone|{GRAPHDRONE_VERSION}")[:8]`).

Report:
- **Current-version GD entries** (valid): N files
- **Stale GD entries** (old version): list them with their version signature
- **Failed GD entries**: list them

### 3c. Compute cost summary

```
!`python3 -c "
import json, os, glob
cache_dir = '/home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache'
files = glob.glob(os.path.join(cache_dir, '*.json'))
total_tabpfn = 0; total_gd = 0; n_tabpfn = 0; n_gd = 0
for f in files:
    try:
        d = json.load(open(f))
        e = d.get('elapsed') or 0
        if 'tabpfn' in os.path.basename(f):
            total_tabpfn += e; n_tabpfn += 1
        elif 'graphdrone' in os.path.basename(f):
            total_gd += e; n_gd += 1
    except: pass
print(f'TabPFN: {n_tabpfn} cached, total compute time={total_tabpfn/60:.1f} min')
print(f'GraphDrone: {n_gd} cached, total compute time={total_gd/60:.1f} min')
print(f'Total compute saved by cache: {(total_tabpfn+total_gd)/60:.1f} min')
" 2>/dev/null || echo "Could not compute cache stats"`
```

## If $ARGUMENTS = prune-gd

Remove only stale GraphDrone cache files (those whose embedded `cache_key` doesn't match current GRAPHDRONE_VERSION).

**NEVER delete TabPFN cache files** — they represent hours of compute that will never need to re-run.

```bash
# Identify stale GD files first (dry run — show what would be deleted):
python3 -c "
import json, os, glob, hashlib
cache_dir = '/home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache'
version = 'CURRENT_VERSION_HERE'  # replace with actual
stale = []
for f in glob.glob(os.path.join(cache_dir, '*__graphdrone.json')):
    try:
        d = json.load(open(f))
        name = os.path.basename(f)
        # parse dataset and fold from filename
        parts = name.replace('__graphdrone.json','').split('__fold')
        if len(parts) == 2:
            ds, fold = parts[0], int(parts[1])
            expected_key = hashlib.sha256(f'{ds}|{fold}|graphdrone|{version}'.encode()).hexdigest()[:16]
            cached_key = d.get('cache_key','')
            if cached_key != expected_key:
                stale.append((f, cached_key[:8], expected_key[:8]))
    except: pass
for f, old, new in stale:
    print(f'STALE: {os.path.basename(f)}  (cached={old} expected={new})')
print(f'Total stale: {len(stale)}')
"
```

Before deleting anything, show the dry-run list and ask the user to confirm.

## If $ARGUMENTS = show-baselines

Read and display the actual cached metric values for TabPFN across all datasets and folds:

```
!`python3 -c "
import json, os, glob
cache_dir = '/home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache'
files = sorted(glob.glob(os.path.join(cache_dir, '*__tabpfn.json')))
print(f'{'File':50s}  {'Status':6s}  Metrics')
print('-'*90)
for f in files:
    try:
        d = json.load(open(f))
        m = d.get('metrics', {})
        status = d.get('status','?')
        name = os.path.basename(f)
        metric_str = '  '.join(f'{k}={v:.4f}' for k,v in m.items())
        print(f'{name:50s}  {status:6s}  {metric_str}')
    except Exception as e:
        print(f'{os.path.basename(f):50s}  ERROR: {e}')
" 2>/dev/null`
```

This shows the stable TabPFN baselines that GD is being compared against. These numbers are fixed as long as the `baseline_v1` key is in place.

## Final summary

Print:
```
=== Cache Status ===
TabPFN baselines : N/18 cached, M failed, K missing  [NEVER DELETE]
GD current version (<GRAPHDRONE_VERSION>) : N/18 cached, M stale
Compute saved    : X.X min TabPFN + Y.Y min GD = Z.Z min total

If running benchmark now:
  TabPFN tasks that will RE-RUN : <list missing/failed>
  GD tasks that will RE-RUN     : <list stale/missing>
  Expected additional GPU time  : ~X min
```
