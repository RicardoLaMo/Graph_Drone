"""
Sprint validator for GraphDrone v1-width experiments.

Fixed 8-dataset × fold-0 canary that runs in < 5 min on 8 GPUs.
Compares against ALL TabArena baselines via compare_on_tabarena().
Use this BEFORE committing any experiment to the full 43-dataset run.

Usage:
    # Normal sprint (uses cache; set ignore_cache=True to force re-run):
    conda run -n h200_tabpfn python scripts/run_sprint.py

    # Force fresh run (ignores any cached results):
    conda run -n h200_tabpfn python scripts/run_sprint.py --fresh

    # Print what the last sprint produced (no re-run):
    conda run -n h200_tabpfn python scripts/run_sprint.py --results-only
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "tabarena" / "tabarena"))

# ── Fixed sprint contract ────────────────────────────────────────────────────
SPRINT_DATASETS = [
    "kddcup09_appetency",           # biggest GD gap, binary, large
    "bank-marketing",               # moderate gap, binary
    "Diabetes130US",                # large binary with NaNs, GD loses
    "diabetes",                     # small binary — regression canary
    "concrete_compressive_strength",# small regression canary
    "airfoil_self_noise",           # small regression canary
    "credit-g",                     # medium binary canary
    "APSFailure",                   # large binary, GD near-best — regression canary
]
SPRINT_FOLDS = [0]
N_GPUS = 8

EXPNAME = str(ROOT / "experiments" / "tabarena_sprint")
EVAL_DIR = ROOT / "eval" / "tabarena_sprint"
WORKER_DIR = ROOT / "experiments" / "tabarena_sprint" / "_worker_results"
# ─────────────────────────────────────────────────────────────────────────────


def _worker(worker_idx: int, dataset: str, ignore_cache: bool):
    """Run one dataset on one GPU."""
    from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    from graphdrone_fit.adapters.tabarena import GraphDroneTabArenaAdapter

    ctx = TabArenaContext()
    task_metadata = ctx.task_metadata

    methods = [
        Experiment(
            name="GraphDrone_v1w_Sprint",
            method_cls=GraphDroneTabArenaAdapter,
            method_kwargs={"n_estimators": 8, "router_kind": "noise_gate_router"},
        )
    ]

    runner = ExperimentBatchRunner(expname=EXPNAME, task_metadata=task_metadata)
    results: list[dict[str, Any]] = runner.run(
        datasets=[dataset],
        folds=SPRINT_FOLDS,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    out = WORKER_DIR / f"worker_{worker_idx:02d}.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"[Sprint Worker {worker_idx} | {dataset}] Done → {out}")


def _aggregate(baseline_elo: float = 1455.7):
    import pandas as pd
    from tabarena.nips2025_utils.end_to_end import EndToEnd
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    from tabarena.website.website_format import format_leaderboard

    ctx = TabArenaContext()
    task_metadata = ctx.task_metadata
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    pkls = sorted(WORKER_DIR.glob("worker_*.pkl"))
    if not pkls:
        print("No worker results found. Run sprint first.")
        sys.exit(1)

    all_results: list[dict[str, Any]] = []
    for p in pkls:
        with open(p, "rb") as f:
            all_results.extend(pickle.load(f))

    e2e = EndToEnd.from_raw(results_lst=all_results, task_metadata=task_metadata, cache=False, cache_raw=False)
    e2e_results = e2e.to_results()

    leaderboard: pd.DataFrame = e2e_results.compare_on_tabarena(
        output_dir=EVAL_DIR,
        only_valid_tasks=True,
        use_model_results=True,
        new_result_prefix="GD_",
    )
    leaderboard.to_csv(EVAL_DIR / "sprint_leaderboard.csv", index=False)

    gd = leaderboard[leaderboard["method"].str.startswith("GD_")]
    lb_top = leaderboard[["method", "elo", "rank", "winrate"]].sort_values("elo", ascending=False)

    print("\n" + "=" * 70)
    print("SPRINT RESULTS  (8 datasets × fold 0)")
    print("=" * 70)
    print(format_leaderboard(df_leaderboard=leaderboard).to_markdown(index=False))

    if not gd.empty:
        r = gd.iloc[0]
        delta = r["elo"] - baseline_elo
        sign = "+" if delta >= 0 else ""
        print("\n" + "─" * 70)
        print(f"  GraphDrone Sprint ELO : {r['elo']:.1f}  ({sign}{delta:.1f} vs baseline {baseline_elo})")
        print(f"  Sprint Rank           : {r['rank']:.1f} / {len(leaderboard)}")
        print(f"  Win Rate              : {r['winrate']*100:.1f}%")
        print(f"\n  → Paste into RESEARCH_LOG.md:  Sprint ELO Δ = {sign}{delta:.1f}")
        print("─" * 70)


def _master(ignore_cache: bool):
    script = str(ROOT / "scripts" / "run_sprint.py")
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"Sprint: {len(SPRINT_DATASETS)} datasets × fold {SPRINT_FOLDS[0]}")
    procs = []
    for i, dataset in enumerate(SPRINT_DATASETS):
        gpu = i % N_GPUS
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OPENBLAS_NUM_THREADS"] = "4"
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        log = log_dir / f"sprint_{i:02d}_{dataset.replace('-','_')}.log"
        lf = open(log, "w")
        cmd = [sys.executable, script, "--worker", str(i), "--dataset", dataset]
        if ignore_cache:
            cmd.append("--fresh")
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((i, dataset, p, lf, log))
        print(f"  GPU {gpu}: {dataset}  (PID {p.pid})")

    print("\nWaiting...")
    failed = []
    for i, dataset, p, lf, log in procs:
        ret = p.wait(); lf.close()
        status = "✓" if ret == 0 else f"✗ exit {ret}"
        print(f"  [{status}] {dataset}")
        if ret != 0:
            failed.append(dataset)

    if failed:
        print(f"\nWARNING: failed: {failed}")
    print("\nAggregating...")
    _aggregate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--fresh", action="store_true", help="Ignore cache, force re-run")
    parser.add_argument("--results-only", action="store_true", help="Print last sprint results without re-running")
    args = parser.parse_args()

    if args.results_only:
        _aggregate()
    elif args.worker is not None:
        _worker(worker_idx=args.worker, dataset=args.dataset, ignore_cache=args.fresh)
    else:
        _master(ignore_cache=args.fresh)
