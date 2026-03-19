---
name: gd
description: GraphDrone master command. Reads current state and tells you exactly what to do next. The only skill you need to remember. Use /gd <goal> to go straight to a workflow.
argument-hint: "[status|bench|report|improve <engine>|secure <engine> <version>|pr|sync|cache|help]"
allowed-tools: Read, Write, Edit, Bash, Grep
---

## What this does

Read the live project state, figure out where you are in the workflow, and either answer your question or route you to the right action with exact commands.

**If $ARGUMENTS is empty** — run a quick state snapshot and tell the user what to do next.
**If $ARGUMENTS starts with a known subcommand** — jump straight to that workflow (details below).

---

## State snapshot (always run this first)

```
!`echo "=== GD VERSION ===" && grep "GRAPHDRONE_VERSION" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/scripts/run_geopoe_benchmark.py | head -2`
```

```
!`echo "=== BRANCH + GIT SYNC ===" && cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research && git branch --show-current && git fetch origin --quiet 2>/dev/null; AHEAD=$(git log origin/main..main --oneline 2>/dev/null | wc -l); BEHIND=$(git log main..origin/main --oneline 2>/dev/null | wc -l); echo "Local main: ${AHEAD} commit(s) ahead, ${BEHIND} commit(s) behind GitHub"`
```

```
!`echo "=== LAST 5 COMMITS ===" && cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research && git log --oneline -5`
```

```
!`echo "=== TAGS (local vs GitHub) ===" && cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research && LOCAL_TAGS=$(git tag --sort=-creatordate | head -5) && echo "Local: $LOCAL_TAGS" && echo "GitHub:" && git ls-remote --tags origin 2>/dev/null | grep -v '\^{}' | awk '{print $2}' | sed 's|refs/tags/||' | sort -r | head -5`
```

```
!`echo "=== BENCHMARK RESULTS ===" && cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/elo_ranking.csv 2>/dev/null || echo "No results yet"`
```

```
!`echo "=== CACHE STATUS ===" && ls /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache/*.json 2>/dev/null | wc -l && echo "cache files" || echo "Cache empty"`
```

```
!`echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"`
```

---

## Routing logic

After reading the state snapshot, determine the situation and respond:

### Situation A: No benchmark results yet
> "No `elo_ranking.csv` found. You need to run a baseline benchmark first.
> Run: `PYTHONPATH=src python scripts/run_geopoe_benchmark.py --folds 0 1 2`"

### Situation B: On main branch, results exist, no uncommitted model changes
> "Current state: GD regression ELO=XXXX, classification ELO=XXXX (vs TabPFN XXXX/XXXX).
> You're on main with clean results.
> **To improve:** `/gd improve <regression|classification>`
> **To analyze results:** `/gd report`
> **To check cache before running:** `/gd bench`"

### Situation C: On a feature branch (feat/*)
> "You're on branch `<branch>` — an improvement is in progress.
> **If you've changed model code:** bump GRAPHDRONE_VERSION in the script, then run the benchmark.
> **If benchmark is done:** `/gd secure <engine> <version>` to lock results and merge."

### Situation D: GRAPHDRONE_VERSION in script doesn't match latest git tag
> "WARNING: Version string <X> is in the script but the latest git tag is <Y>.
> If you changed model code, bump the version before running or you'll get stale cached GD results."

### Situation E: Local main is ahead of GitHub (unpushed commits)
> "Local main has N commit(s) not yet on GitHub. Run `/gd sync` to push them."

### Situation F: Local tags missing from GitHub
> "Tag <X> exists locally but not on GitHub. Run `/gd sync` to push all tags."

### Situation G: Local main is behind GitHub (someone else pushed)
> "WARNING: GitHub has N commit(s) not in your local main. Run `git pull origin main` before starting any improvement work."

---

## Subcommands

### `/gd status`
Read actual result files and verify everything is consistent. Full ground-truth check.

Read:
- `eval/geopoe_benchmark/elo_ranking.csv` — actual ELOs
- `eval/geopoe_benchmark/report.txt` — full report
- `CLAUDE.md` ELO table
- `GRAPHDRONE_VERSION` in benchmark script
- Latest git tags

Report any discrepancy between what the files say and what CLAUDE.md documents.
Add: "DO NOT rely on memory for ELOs — only these files are ground truth."

---

### `/gd bench [regression|classification|both]`
Pre-flight check before running the benchmark.

1. Show `GRAPHDRONE_VERSION`
2. Scan cache: how many TabPFN tasks cached (reusable) vs GD tasks that need re-running
3. Check GPU free memory
4. Check for model code commits since last version bump
5. Print the exact command to run

```
!`for f in /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_cache/*.json; do [ -f "$f" ] && python3 -c "import json,os; d=json.load(open('$f')); n=os.path.basename('$f'); m='tabpfn' if 'tabpfn' in n else 'gd'; print(m, n[:40], d.get('status','?'))" 2>/dev/null; done | sort | head -40`
```

Summary line: "TabPFN: X/18 cached (will reuse). GD: Y/18 cached for current version. Z tasks will re-run (~N min on H200)."

---

### `/gd report [regression|classification|both]`
Analyze results from `results_granular.csv`. Produce:

1. **Per-dataset table**: GD vs TabPFN on every metric (R²+RMSE for regression; F1+AUC-ROC+PR-AUC for classification)
2. **ELO interpretation**: With N matchups, each flip = ~20 ELO. Show implied win rate vs observed win rate.
3. **Volatile matchups**: datasets where GD wins 2/3 folds with margin < 0.005 — these could flip
4. **Improvement targets**: top 2 datasets where closing the gap would most move ELO

```
!`cat /home/wliu23/projects/GraphDrone2/Graph_Drone_research/eval/geopoe_benchmark/results_granular.csv 2>/dev/null || echo "MISSING"`
```

Key point to explain: "ELO is a ranking signal over win/loss matchups, not a direct measure of metric magnitude. A GD win by R²=0.001 counts the same as a win by R²=0.05. Use the per-dataset metric table to understand margin quality."

---

### `/gd improve <engine>`
Start a structured improvement branch. Engine = `regression` or `classification`.

1. Pull latest main from GitHub first: `git pull origin main`
2. Read ground truth ELOs from result files (not memory)
3. Check DO NOT rules from CLAUDE.md:

```
!`grep -A 55 "DO NOT rules" /home/wliu23/projects/GraphDrone2/Graph_Drone_research/CLAUDE.md`
```

4. Create branch and immediately push to GitHub:
```bash
git checkout -b feat/<engine>-improvement-$(date +%Y%m%d)
git push -u origin feat/<engine>-improvement-$(date +%Y%m%d)
```

5. Write improvement plan with: baseline to beat, hypothesis, files to change, new version string, rollback plan
6. **Wait for user confirmation before touching any code**

When the improvement is ready to review, use `/gd pr` to open a pull request.

DO NOT rules by engine (from CLAUDE.md — read above, don't rely on memory):
- Regression: no `bootstrap_full_only`, no CB/XGB, no missing residual penalty, don't cite 1514.7
- Classification: no `contextual_transformer` router, no GORA, no CB/XGB

---

### `/gd secure <engine> <version_string>`
Lock a successful result and sync to GitHub. Engine = `regression`, `classification`, or `both`. Version string = e.g. `v1-geopoe-2026.03.20a`.

Parse $ARGUMENTS: first word = engine, second word = version string to lock.

1. Read `elo_ranking.csv` and `report.txt` — **STOP if missing**
2. Verify version string in the benchmark script matches the second word of $ARGUMENTS
3. Update CLAUDE.md ELO table (only the engine that was run; leave the other engine's row untouched)
4. Update VERSIONS.md
5. Write memory file at `~/.claude/projects/-home-wliu23-projects-GraphDrone2/memory/benchmark_final_results.md`
6. Commit and tag:
```bash
git add CLAUDE.md VERSIONS.md
git commit -m "docs: lock benchmark results for <version_string>"
git tag <version_tag>   # e.g. v1.19 — increment from last tag
```
7. **Push to GitHub** — confirm with user first, then:
```bash
git push origin main
git push origin <version_tag>
```
8. Print locked summary:
```
=== SECURED: GraphDrone <engine> (<version>) ===
Regression ELO    : GD X vs TabPFN Y  [N/36 tasks]
Classification ELO: GD X vs TabPFN Y  [N/36 tasks]
Git tag           : <tag> — pushed to GitHub ✓
CLAUDE.md         : updated ✓
Memory            : benchmark_final_results.md updated ✓

BEATS TO BEAT NEXT TIME:
  Regression    : must exceed X
  Classification: must exceed X
```

---

### `/gd pr`
Open a pull request from the current feature branch to main on GitHub.

1. Confirm the current branch is a `feat/*` branch (not main)
2. Push any unpushed commits: `git push origin <branch>`
3. Read `elo_ranking.csv` to get the new ELOs for the PR description
4. Read the last 10 commits on this branch vs main for the change summary
5. Create the PR:
```bash
gh pr create \
  --base main \
  --title "<engine> improvement: <metric> ELO X → Y (<version>)" \
  --body "$(cat <<'EOF'
## What changed
- Engine: <regression|classification>
- GRAPHDRONE_VERSION: <version>
- Key change: <one-line description>

## Benchmark results
| Engine | GD ELO | TabPFN ELO | vs baseline |
|---|---|---|---|
| Regression | X | Y | +/- Z |
| Classification | X | Y | +/- Z |

## DO NOT violations checked
- [ ] No bootstrap_full_only for regression
- [ ] No CB/XGB in either engine
- [ ] No contextual_transformer for classification
- [ ] MSE residual penalty present (regression)
- [ ] GRAPHDRONE_VERSION bumped before benchmark run

## Test
- [ ] Smoke test: `python tests/test_smoke_v118.py`
- [ ] Benchmark: results in eval/geopoe_benchmark/

🤖 Generated with Claude Code
EOF
)"
```
6. Print the PR URL.

---

### `/gd sync`
Push all local-only commits and tags to GitHub.

1. Check what's unpushed:
```
!`cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research && git fetch origin --quiet 2>/dev/null; echo "Commits to push:"; git log origin/main..main --oneline; echo "Tags to push:"; git tag | while read t; do git ls-remote --tags origin "refs/tags/$t" 2>/dev/null | grep -q "$t" || echo "  $t (local only)"; done`
```

2. Show what will be pushed and ask for confirmation.

3. On confirmation:
```bash
git push origin main
git push origin --tags   # push all local tags not yet on GitHub
```

4. Confirm: "Pushed N commits and M tags to GitHub/RicardoLaMo/Graph_Drone."

---

### `/gd cache [status|prune-gd|show-baselines]`
Manage benchmark cache.

- `status` (default): list all cache files, classify as tabpfn/gd, show stale vs current, compute total saved compute time
- `prune-gd`: dry-run list of stale GD cache files (those not matching current version) — ask for confirmation before deleting. **Never delete TabPFN files.**
- `show-baselines`: print the actual cached TabPFN metric values for all datasets/folds

---

### `/gd help`
Print this reference card:

```
/gd                          — state snapshot + what to do next
/gd status                   — verify ground truth from files
/gd bench [scope]            — pre-flight: what will re-run vs reuse cache
/gd report [scope]           — ELO + R²/F1/AUC per dataset, volatility
/gd improve <engine>         — create branch, plan, push to GitHub
/gd secure <engine> <ver>    — lock result: docs + git tag + push
/gd pr                       — open GitHub PR from current feature branch
/gd sync                     — push unpushed commits and tags to GitHub
/gd cache [action]           — inspect/prune benchmark cache

Full workflow:
  /gd sync                          ← start: pull latest + push any pending
  /gd bench                         ← check cache before running
  run benchmark                     ← PYTHONPATH=src python scripts/...
  /gd report                        ← understand results
  /gd improve <engine>              ← branch + plan (pushes branch to GitHub)
  [implement + bump version]
  /gd bench                         ← confirm what will re-run
  run benchmark
  /gd report                        ← verify improvement
  /gd secure <engine> <ver>         ← lock + push tag + push main
  /gd pr                            ← open PR feat/* → main
  [merge PR on GitHub]
```
