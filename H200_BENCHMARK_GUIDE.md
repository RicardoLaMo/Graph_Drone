# H200 Parallel TabArena Benchmark Guide

**Objective:** Run full TabArena validation (51 datasets × 3 folds = 153 tasks) efficiently on H200 GPUs

**Estimated Runtime:** 4-6 hours on GPUs 1-5
**GPU Memory:** H200 has 141GB per GPU (vs A100 40GB) - excellent for this workload
**Expected Throughput:** ~2-3 datasets per GPU per hour

---

## 🚀 Quick Start

### Launch the benchmark:

```bash
cd /home/wliu23/projects/GraphDrone2/Graph_Drone_research
bash scripts/launch_h200_benchmark.sh
```

This will:
1. ✅ Verify conda environment (h200_tabpfn)
2. ✅ Check GPU availability
3. ✅ Launch benchmark in detached tmux session
4. ✅ Start GPU monitoring window
5. ✅ Stream real-time logs

### Monitor progress:

```bash
# Attach to benchmark window (see logs in real-time)
tmux attach-session -t h200_benchmark_YYYYMMDD_HHMMSS

# Inside tmux:
# - Window 0: Benchmark execution (currently selected)
# - Window 1: GPU monitoring dashboard
# - Window 2: Log stream
```

### Quick test (before full run):

```bash
bash scripts/launch_h200_benchmark.sh --quick
```

This runs 5 datasets × 1 fold (~15-20 minutes) to verify setup.

---

## 🏗️ Architecture & Optimization

### Multi-GPU Distribution

```
Tasks (153 total)
    ├─ GPU 1: Tasks 0, 5, 10, 15, ... (round-robin)
    ├─ GPU 2: Tasks 1, 6, 11, 16, ...
    ├─ GPU 3: Tasks 2, 7, 12, 17, ...
    ├─ GPU 4: Tasks 3, 8, 13, 18, ...
    └─ GPU 5: Tasks 4, 9, 14, 19, ...
```

**Load balancing:** Each GPU gets ~31 tasks (153 ÷ 5 = 30.6)

### Parallel Execution

- **5 worker processes** (one per GPU)
- **ProcessPoolExecutor** manages task queue
- **H200 memory optimization** prevents OOM
  - H200: 141GB per GPU (plenty of headroom)
  - Model + TabArena + buffer: ~50GB per GPU
  - 2-3 concurrent fits per GPU possible

### Task Scheduling

```python
for fold in [0, 1, 2]:
    for dataset_idx in range(51):
        gpu_id = gpu_ids[task_id % n_gpus]
        # Submit task to worker on that GPU
```

This ensures:
- ✅ All GPUs keep working continuously
- ✅ No GPU sits idle while others work
- ✅ Automatic load balancing
- ✅ Fair distribution across different fold groups

### Memory Efficiency

H200 vs A100:
- **H200:** 141GB per GPU (3.5× more than A100)
- **Per-task memory:** ~15-20GB (fit + validation + inference)
- **Concurrent tasks:** Can fit 6-7 per GPU if needed
- **Safety margin:** Running 1-2 per GPU ensures stability

---

## 📊 Expected Performance

### Timeline

| Phase | Est. Time | Tasks |
|-------|-----------|-------|
| Setup & first task | 2-3 min | 1 |
| Initial ramp-up | 10-15 min | ~10 |
| Steady state | 3-5 hours | 140 |
| Final tasks + aggregation | 10-20 min | 2 |
| **Total** | **4-6 hours** | **153** |

### GPU Utilization

Expected during steady state:
- **GPU Util:** 85-95% (compute)
- **Memory:** 60-70% (plenty of headroom)
- **Power:** ~350-400W per H200

### Task Duration

- **Small dataset (50-1000 rows):** 20-30 seconds
- **Medium dataset (1000-10k rows):** 40-60 seconds
- **Large dataset (10k-100k rows):** 60-120 seconds
- **Average:** ~45 seconds per task

**Total compute time:** 153 × 45s ÷ 60 = 115 minutes ÷ 5 GPUs = 23 minutes per GPU
**With overhead:** 4-6 hours actual time

---

## 🔧 Customization Options

### Launch with custom GPUs:

```bash
# Use only GPUs 0-3 (if you want to save GPU 4-5)
python scripts/run_tabarena_h200_parallel.py \
    --gpus 0 1 2 3 \
    --datasets 51 \
    --folds 3 \
    --workers 4
```

### Enable checkpointing for recovery:

```bash
# Automatically enabled by launch script
bash scripts/launch_h200_benchmark.sh
# Checkpoint saved every task in checkpoints/tabarena_YYYYMMDD_HHMMSS/
```

If interrupted, the script will resume from where it left off.

### Quick test parameters:

```bash
# 5 datasets, 1 fold (~15 min)
bash scripts/launch_h200_benchmark.sh --quick

# Or manually:
python scripts/run_tabarena_h200_parallel.py \
    --datasets 5 \
    --folds 1 \
    --gpus 1 2 3 4 5 \
    --checkpoint
```

---

## 📈 Monitoring & Logging

### GPU Monitoring Dashboard

Auto-launched in tmux window "gpu-monitor":

```
GPU Status at 2026-03-16 14:30:45
======================================
GPU 1: NVIDIA H100 | GPU Util: 92% | Mem: 85614/141288MB
GPU 2: NVIDIA H100 | GPU Util: 88% | Mem: 79842/141288MB
GPU 3: NVIDIA H100 | GPU Util: 95% | Mem: 91234/141288MB
GPU 4: NVIDIA H100 | GPU Util: 87% | Mem: 76543/141288MB
GPU 5: NVIDIA H100 | GPU Util: 90% | Mem: 88765/141288MB
```

Updates every 5 seconds.

### Log File

Location: `logs/tabarena_h200_parallel.log`

Sample output:
```
2026-03-16 14:30:45 | INFO     | Loading TabArena context...
2026-03-16 14:30:48 | INFO     | Created 153 tasks (51 datasets × 3 folds)
2026-03-16 14:30:48 | INFO     | Using GPUs: [1, 2, 3, 4, 5]
2026-03-16 14:30:48 | INFO     | Running 153 pending tasks...
2026-03-16 14:30:52 | INFO     | [GPU 1] Task 0: airfoil_self_noise (fold 0)
2026-03-16 14:31:04 | INFO     | [GPU 1] ✅ Task 0 completed (12.1s)
2026-03-16 14:31:05 | INFO     | Progress: 1/153 (0.7%) | Completed: 1, Failed: 0
```

### Real-time log stream

Tmux window "logs" shows tail of logs as they're written.

---

## ⚠️ Troubleshooting

### Benchmark not starting

```bash
# Check if h200_tabpfn environment exists
conda env list | grep h200_tabpfn

# If missing, create it (should already exist)
conda create -n h200_tabpfn python=3.11 pytorch::pytorch pytorch::pytorch-cuda=12.1 -c pytorch -c nvidia
```

### GPU not found

```bash
# Check available GPUs
nvidia-smi

# If GPUs 1-5 don't exist, adjust launch script:
bash scripts/launch_h200_benchmark.sh  # or
python scripts/run_tabarena_h200_parallel.py --gpus 0 1 2 3 4
```

### Out of memory error

This is very unlikely on H200 (141GB), but if it happens:
- Reduce workers: `--workers 3`
- This serializes more tasks, reduces peak memory

### Benchmark hangs

- Check GPU temperatures: `nvidia-smi -l 1` (should be <80°C)
- Check CPU load: `top` (should be moderate, not maxed)
- Restarted checkpoint: `--checkpoint` will resume from last completed task
- Kill session: `tmux kill-session -t session_name`

### Task failures during benchmark

The script logs failures to console and checkpoint. Minor failures are OK:
- 1-2 failures: Likely transient, benchmark continues
- 5+ failures: Investigate logs for pattern
- All failures: Check that GraphDrone is importable: `python -c "from graphdrone_fit import GraphDrone"`

---

## 📊 Expected Results

### Output Files

After completion, you'll have:

```
eval/tabarena_h200_parallel/
├── leaderboard.csv          # 153 rows (51 × 3)
├── metrics_summary.json     # ELO, statistics
├── benchmark_report.txt     # Summary statistics
├── raw_results/             # Individual task results
└── ...

checkpoints/
└── tabarena_YYYYMMDD_HHMMSS/
    └── task_status.json     # Checkpoint for recovery
```

### Leaderboard CSV Format

```
dataset,fold,PC_MoE_elo,baseline_elo,delta
airfoil_self_noise,0,1485.2,1458.9,26.3
airfoil_self_noise,1,1482.1,1458.9,23.2
airfoil_self_noise,2,1488.5,1458.9,29.6
...
```

### Analysis

After benchmark:
```python
import pandas as pd
df = pd.read_csv("eval/tabarena_h200_parallel/leaderboard.csv")

# Overall ELO change
baseline_elo = df['baseline_elo'].mean()
new_elo = df['PC_MoE_elo'].mean()
delta = new_elo - baseline_elo

print(f"Baseline ELO: {baseline_elo:.1f}")
print(f"New ELO: {new_elo:.1f}")
print(f"Delta: {delta:+.1f}")

# Top improvements
top_wins = df.nlargest(10, 'delta')
print("\nTop 10 datasets with largest improvements:")
print(top_wins[['dataset', 'delta']])
```

---

## 🎯 Next Steps After Benchmark

1. **Analyze results**
   ```bash
   python eval/tabarena_h200_parallel/analyze_results.py
   ```

2. **Update evidence files**
   ```bash
   cp eval/tabarena_h200_parallel/leaderboard.csv reports/
   # Update reports/pr_0_metrics.json with new ELO
   ```

3. **Commit results**
   ```bash
   git add reports/ eval/tabarena_h200_parallel/
   git commit -m "docs: add full TabArena benchmark results"
   ```

4. **Open PR on GitHub**
   - Use .github/PULL_REQUEST_TEMPLATE.md
   - Include evidence files
   - Link to this benchmark guide

5. **Merge decision**
   - If ELO > baseline: Proceed to merge
   - If ELO < baseline: Investigate regression
   - If neutral: Check multiclass-specific improvements

---

## 💾 Checkpoint System

### How it works

After each task completes, a checkpoint is saved:

```json
{
  "timestamp": "2026-03-16T14:30:45.123456",
  "total_tasks": 153,
  "tasks": [
    {"task_id": 0, "dataset": "...", "fold": 0, "status": "completed", "duration": 12.1},
    {"task_id": 1, "dataset": "...", "fold": 0, "status": "running", "duration": null},
    ...
  ]
}
```

### Recovery

If benchmark is interrupted:
1. Script detects checkpoint file
2. Loads task status
3. Skips completed tasks
4. Resumes from pending tasks

No data loss, no repeated work.

---

## 📝 Summary

**H200 Optimization leverages:**
- ✅ 141GB GPU memory (3.5× more than A100)
- ✅ Multi-GPU parallelization (GPUs 1-5)
- ✅ Intelligent task distribution (round-robin)
- ✅ Checkpoint/recovery system
- ✅ Real-time monitoring (GPU + logs)
- ✅ ProcessPoolExecutor for true parallelism

**Result:** Full TabArena validation (4-6 hours) with:
- ~85-95% GPU utilization
- Fair load distribution
- Fault tolerance
- Complete observability

---

## 🚀 **Ready? Launch it!**

```bash
bash scripts/launch_h200_benchmark.sh
```

Then monitor:
```bash
tmux attach-session -t h200_benchmark_YYYYMMDD_HHMMSS
```

**Status: Ready for deployment!**
