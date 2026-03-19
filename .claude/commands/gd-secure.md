---
name: gd-secure
description: Lock a successful benchmark result. Reads actual result files, updates CLAUDE.md ELO table, writes a memory file, creates a git tag, and commits documentation. Run this immediately after a successful benchmark run.
argument-hint: "<engine> <new_version_string>  e.g.: regression v1-geopoe-2026.03.20a"
allowed-tools: Read, Write, Edit, Bash, Grep
---

$ARGUMENTS contains the engine and version string to lock, e.g. `regression v1-geopoe-2026.03.20a` or `classification v1-geopoe-2026.03.20b` or `both v1-geopoe-2026.03.20c`.

## Step 1: Read actual benchmark results

**CRITICAL: Read the files. Do not use memory or prior conversation ELOs.**

```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/elo_ranking.csv 2>/dev/null || echo "MISSING: elo_ranking.csv — cannot secure without results"`
```

```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/report.txt 2>/dev/null || echo "MISSING: report.txt"`
```

```
!`tail -30 /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/run_log.txt 2>/dev/null || echo "MISSING: run_log.txt"`
```

If any of these files are missing, **STOP and report** — do not proceed with securing phantom results.

## Step 2: Verify version string matches $ARGUMENTS

Parse $ARGUMENTS: the first word is the engine (`regression`, `classification`, or `both`); the second word is the expected version string (e.g. `v1-geopoe-2026.03.20b`).

```
!`grep "GRAPHDRONE_VERSION" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/scripts/run_geopoe_benchmark.py | head -3`
```

Compare only the version string part of $ARGUMENTS against the version string in the script. If they do not match, **STOP** and report: "Script has version X but you passed version Y. Did you forget to bump GRAPHDRONE_VERSION before running the benchmark?"

## Step 3: Extract ELOs from result files

From the files read in Step 1, identify:
- Regression ELO for GraphDrone and TabPFN (if engine=regression or both)
- Classification ELO for GraphDrone and TabPFN (if engine=classification or both)
- Number of tasks completed / total tasks
- Date of the run

## Step 4: Update CLAUDE.md

Read `/home/wliu23/projects/GraphDrone2/Graph_Drone_research/CLAUDE.md` and update:
1. The ELO table at the top (date, version string, GD ELO, TabPFN ELO, tasks)
2. The engine description section if the architecture changed
3. The ELO history table at the bottom (add a new row)

Only update the engine that was benchmarked. Do not modify the other engine's row unless it was also run.

## Step 5: Update VERSIONS.md

Read `/home/wliu23/projects/GraphDrone2/Graph_Drone_research/VERSIONS.md` and update the current version entry with the new ELOs.

## Step 6: Write memory

Write a memory file at `/home/wliu23/.claude/projects/-home-wliu23-projects-GraphDrone2/memory/benchmark_final_results.md` with the new results. Use this format:

```
---
name: benchmark_final_results_<tag>
description: Definitive benchmark results for <version> — reg ELO <X>, clf ELO <Y> on geopoe benchmark (<date>)
type: project
---

## Definitive benchmark results (<version>, <date>, main)

[ELO table from result files]

## Numbers to ignore (stale / misleading)

[Any prior ELOs that should not be used as targets anymore]

**How to apply:** These are ground truth for this version. Any future regression improvement must beat <reg_ELO>; classification must beat <clf_ELO>.
```

Also update the index line for `benchmark_final_results.md` in `/home/wliu23/.claude/projects/-home-wliu23-projects-GraphDrone2/memory/MEMORY.md`.

## Step 7: Commit and tag

```bash
cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research
git add CLAUDE.md VERSIONS.md
git commit -m "docs: lock benchmark results for $ARGUMENTS

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git tag <version_tag>
```

Use a tag like `v1.18` or increment based on what changed. Report the tag name.

**Do NOT push** unless the user explicitly says to push.

## Step 8: Summary report

Print a single locked-state summary:

```
=== SECURED: GraphDrone <engine> (<version>) ===
Regression ELO : GD <X> vs TabPFN <Y>   [tasks: N/36]
Classification ELO: GD <X> vs TabPFN <Y> [tasks: N/36]
Git tag        : <tag> on main
CLAUDE.md      : updated
Memory         : benchmark_final_results.md updated
Cache key      : <GRAPHDRONE_VERSION> (must match next run)

BEATS TO BEAT:
  Regression   : must exceed <reg_ELO>
  Classification: must exceed <clf_ELO>
```
