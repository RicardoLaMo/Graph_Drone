---
name: gd-status
description: Verify GraphDrone ground truth by reading actual result files. Reports current ELOs, version strings, and detects any discrepancy between code, benchmark output, and documentation.
allowed-tools: Read, Bash, Grep
---

Read the actual state of GraphDrone from disk — not from memory or prior conversation. Report findings.

## 1. Version strings in code

Read these three sources and report exact values:

**Benchmark script version:**
```
!`grep -n "GRAPHDRONE_VERSION" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/scripts/run_geopoe_benchmark.py | head -5`
```

**Package version (pyproject.toml):**
```
!`grep "^version" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/pyproject.toml`
```

**VERSIONS.md current entry:**
```
!`grep -A 12 "Current production" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/Graph_Drone_research/VERSIONS.md 2>/dev/null || grep -A 12 "Current production" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/VERSIONS.md`
```

## 2. Latest benchmark results from disk

Read actual output files — these are ground truth, not cached memory.

**ELO ranking (most recent run):**
```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/elo_ranking.csv 2>/dev/null || echo "FILE NOT FOUND: eval/geopoe_benchmark/elo_ranking.csv"`
```

**Last 60 lines of run log:**
```
!`tail -60 /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/run_log.txt 2>/dev/null || echo "FILE NOT FOUND: eval/geopoe_benchmark/run_log.txt"`
```

**Report summary:**
```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/report.txt 2>/dev/null || echo "FILE NOT FOUND: eval/geopoe_benchmark/report.txt"`
```

## 3. CLAUDE.md documented ELOs

```
!`grep -A 6 "Current best ELO" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/Graph_Drone_research/CLAUDE.md 2>/dev/null || grep -A 6 "Current best ELO" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/CLAUDE.md`
```

## 4. Git state

```
!`cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research && git log --oneline -8 && echo "---" && git tag --sort=-creatordate | head -5 && echo "---BRANCH:" && git branch`
```

## 5. Analysis

After reading all of the above, report:

- **Current GRAPHDRONE_VERSION** in the benchmark script
- **Regression ELO**: what the actual result files show vs what CLAUDE.md documents
- **Classification ELO**: same
- **Discrepancy check**: does the version string in the benchmark script match the version in VERSIONS.md?
- **Cache warning**: if GRAPHDRONE_VERSION has NOT been bumped since last code change, flag it
- **Any stale cached results**: if `elo_ranking.csv` exists but benchmark has been re-run with a newer version string, flag it
- **Recommendation**: is it safe to start new improvement work, or must something be resolved first?

DO NOT rely on Claude memory for ELO numbers. The numbers in the result files are ground truth.
