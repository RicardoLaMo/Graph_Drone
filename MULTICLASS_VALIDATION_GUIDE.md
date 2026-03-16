# Multiclass Refactor Validation Guide

**Branch:** `exp/multi-classification-refactor`
**Status:** Ready for full TabArena validation
**Created:** 2026-03-16

---

## ✅ Completed

### 1. Implementation
- [x] ProbabilisticCategoricalMoE (PC-MoE) implemented in `src/graphdrone_fit/`
- [x] Token builder supports PET tokens (probability-enabled tokens)
- [x] Router uses cross-entropy loss for classification
- [x] Probability-mixture integration in `defer_integrator.py`
- [x] `predict_proba()` support in `portfolio_loader.py` and `expert_factory.py`

### 2. Local Validation
- [x] Classification test passes: **96.67% accuracy on Iris** ✅
- [x] Validation script created: `validation_scripts/test_classification.py`
- [x] Quick benchmark script: `validation_scripts/quick_benchmark.py`

### 3. Evidence Files
- [x] `reports/pr_0_summary.md` — Experiment objective, hypothesis, scope, results
- [x] `reports/pr_0_metrics.json` — Structured metrics (accuracy, device, seed)

### 4. Documentation
- [x] Commit message follows format: `exp(multi-classification): refactor...`
- [x] Benchmark runner script: `scripts/run_full_tabarena_benchmark.py`

---

## 📋 Next Steps: Full TabArena Validation

Per **GOVERNANCE.md**, the multiclass refactor needs evidence from full TabArena benchmark:
- All 51 datasets
- 3 folds per dataset = 153 total runs
- ELO ranking vs baseline
- Complete leaderboard

### Step 1: Quick Local Test (5 min)

Test on local datasets before committing GPU time:

```bash
# Test on wine and breast_cancer datasets
PYTHONPATH=src python3 validation_scripts/quick_benchmark.py

# Expected: Good accuracy on binary classification before multiclass focus
```

### Step 2: Full TabArena Benchmark (4-6 hours)

Run the complete benchmark to get final ELO metrics:

```bash
# Activate TabArena environment (required)
conda activate h200_tabpfn

# Run full benchmark: all 51 datasets, 3 folds
python scripts/run_full_tabarena_benchmark.py \
    --datasets 51 \
    --folds 3 \
    --output-dir eval/tabarena_full \
    --device cuda

# This generates:
# - eval/tabarena_full/leaderboard.csv (complete results)
# - eval/tabarena_full/metrics_summary.json (ELO, statistics)
```

**Expected runtime:** 4-6 hours on 6x A100 40GB GPUs
**Disk space needed:** ~10-15 GB for evaluation artifacts

### Step 3: Quick Test Mode (if GPU time limited)

For testing without full GPU commitment:

```bash
# Quick test: 5 datasets, 1 fold (~15-20 min)
python scripts/run_full_tabarena_benchmark.py --quick

# This generates:
# - eval/tabarena_quick/leaderboard.csv (5 dataset sample)
# - eval/tabarena_quick/metrics_summary.json
```

**Use when:**
- Debugging issues (before full run)
- Time-constrained testing
- Verifying setup works

---

## 📊 Interpreting Results

### Leaderboard CSV Format

The output `leaderboard.csv` contains:
- **task_name** — Dataset name
- **fold** — Fold number (0-2)
- **PC_MoE_elo** — ELO rating for PC-MoE run
- **baseline_elo** — TabArena official baseline
- **delta** — PC_MoE_elo - baseline_elo (positive = improvement)

### Metrics JSON

```json
{
  "experiment": "PC-MoE Multiclass Refactor",
  "timestamp": "2026-03-16T...",
  "mean_elo": 1485.3,
  "std_elo": 45.2,
  "datasets_tested": 51,
  "folds_per_dataset": 3,
  "total_runs": 153
}
```

### Decision Criteria

**✅ MERGE if:**
- Mean ELO > baseline ELO (positive delta)
- No major regressions on individual datasets
- Multiclass datasets show clear improvements
- Polish_companies (imbalanced) improves significantly

**⚠️ INVESTIGATE if:**
- Mean ELO neutral (±5 ELO)
- Mixed results (some datasets ↑, some ↓)
- Regression on imbalanced datasets

**❌ DO NOT MERGE if:**
- Mean ELO < baseline (negative delta > 10)
- Major regression on any dataset (delta < -20)
- Crashes or errors during benchmark

---

## 🔧 Troubleshooting

### TabArena not found
```bash
# Ensure you're in h200_tabpfn environment
conda activate h200_tabpfn
python -c "from tabarena import Experiment; print('OK')"
```

### GPU out of memory
```bash
# Reduce n_estimators in the script
# Or run with --quick flag first
```

### Benchmark times out
```bash
# Run a subset to test setup
python scripts/run_full_tabarena_benchmark.py --datasets 5 --folds 1
```

---

## 📈 Expected Performance

Based on architecture changes:

**Multiclass datasets (should improve):**
- iris (3-class): baseline ~95%, expecting +2-4%
- Other multiclass datasets: expecting +1-3% improvement

**Binary datasets (should maintain):**
- Binary accuracy: expecting ±1% (stable or slight improvement)
- ROC AUC: expecting ±1%

**Overall ELO impact:**
- Baseline ELO: ~1458 (current standard)
- PC-MoE expected: +10-30 ELO (conservative estimate)
- If much lower: debug token design or router training

---

## 📝 Updating Evidence Files

After benchmark completes, update evidence:

```bash
# 1. Copy leaderboard to evidence
cp eval/tabarena_full/leaderboard.csv reports/

# 2. Extract baseline and new ELO
python << 'EOF'
import pandas as pd
import json

# Read leaderboard
df = pd.read_csv("eval/tabarena_full/leaderboard.csv")

# Calculate metrics
baseline_elo = df['baseline_elo'].mean()
new_elo = df['PC_MoE_elo'].mean()
delta = new_elo - baseline_elo

# Update metrics file
metrics = {
    "experiment": "pc-moe-multiclass",
    "baseline_elo": float(baseline_elo),
    "new_elo": float(new_elo),
    "delta_elo": float(delta),
    "datasets_tested": 51,
    "folds": 3,
    "total_runs": 153,
    "hardware": "6x A100 40GB",
    "reproduction_cmd": "conda run -n h200_tabpfn python scripts/run_full_tabarena_benchmark.py --datasets 51 --folds 3"
}

with open("reports/pr_0_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Baseline ELO: {baseline_elo:.1f}")
print(f"New ELO: {new_elo:.1f}")
print(f"Delta: {delta:+.1f}")
EOF

# 3. Commit evidence
git add reports/
git commit -m "docs: add full TabArena benchmark results (51 datasets × 3 folds)"
```

---

## 🚀 Integration Timeline

1. **Today:** Classification test passes ✅
2. **Tonight/Tomorrow:** Run full TabArena (4-6 hours, ~40 GPU hours)
3. **After completion:** Update evidence files, commit results
4. **Review stage:** Submit PR with complete evidence package
5. **Merge decision:** Review results against GOVERNANCE.md criteria

---

## 📚 References

- **GOVERNANCE.md** — Full requirements for experiment merging
- **scripts/run_full_tabarena_benchmark.py** — Full benchmark runner
- **validation_scripts/test_classification.py** — Local test verification
- **reports/pr_0_summary.md** — Experiment summary

---

## ✅ Validation Checklist

Before submitting PR:

- [ ] Local classification test passes (96.67% on Iris)
- [ ] Full TabArena benchmark completes successfully
- [ ] `eval/tabarena_full/leaderboard.csv` exists
- [ ] `reports/pr_0_metrics.json` updated with ELO metrics
- [ ] Mean ELO > baseline (or well-justified if not)
- [ ] No major regressions (< 10 ELO per dataset)
- [ ] Commit message follows format: `exp(scope): subject`
- [ ] Evidence files present: `reports/pr_0_*.md`, `reports/pr_0_*.json`
- [ ] PR template populated with results
- [ ] Ready for human review and approval

---

**Status:** Ready to proceed with full TabArena validation
**Next action:** Run `scripts/run_full_tabarena_benchmark.py` in h200_tabpfn environment
