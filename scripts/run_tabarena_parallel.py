"""
Parallel TabArena benchmark runner for GraphDrone.
Spawns 8 worker subprocesses, one per GPU, then aggregates ELO results.

Usage:
    # Full run (master mode):
    conda run -n h200_tabpfn python scripts/run_tabarena_parallel.py

    # Single worker (called internally by master):
    conda run -n h200_tabpfn python scripts/run_tabarena_parallel.py --worker 3 --gpu 3
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

EXPNAME = str(ROOT / "experiments" / "tabarena_full")
EVAL_DIR = ROOT / "eval" / "tabarena_full"
WORKER_RESULTS_DIR = ROOT / "experiments" / "tabarena_full" / "_worker_results"
N_GPUS = 8
FOLDS = [0, 1, 2]  # All 3 folds for every dataset


def get_eligible_datasets():
    """Return (names, tids) filtered to binary + regression only."""
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    ctx = TabArenaContext()
    meta = ctx.task_metadata
    mask = meta["problem_type"].isin(["binary", "regression"])
    filtered = meta[mask].sort_values("tid").reset_index(drop=True)
    return list(filtered["name"]), list(filtered["tid"]), ctx.task_metadata


def split_into_chunks(lst, n):
    """Split list into n roughly-equal chunks."""
    k, m = divmod(len(lst), n)
    chunks = []
    idx = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        chunks.append(lst[idx: idx + size])
        idx += size
    return chunks


# ─────────────────────────────────────────────────────────────────
# WORKER MODE: runs a dataset chunk on a single GPU
# ─────────────────────────────────────────────────────────────────
def run_worker(worker_idx: int, gpu_id: int):
    from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    from graphdrone_fit.adapters.tabarena import GraphDroneTabArenaAdapter

    names, tids, task_metadata = get_eligible_datasets()
    chunks = split_into_chunks(names, N_GPUS)
    my_datasets = chunks[worker_idx]

    if not my_datasets:
        print(f"[Worker {worker_idx}] No datasets assigned, exiting.")
        return

    print(f"[Worker {worker_idx} | GPU {gpu_id}] Assigned {len(my_datasets)} datasets: {my_datasets}")
    print(f"[Worker {worker_idx}] Folds: {FOLDS}  →  {len(my_datasets) * len(FOLDS)} tasks")

    methods = [
        Experiment(
            name="GraphDrone_v2026.03_Consolidated",
            method_cls=GraphDroneTabArenaAdapter,
            method_kwargs={"n_estimators": 8, "router_kind": "noise_gate_router"},
        )
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=EXPNAME, task_metadata=task_metadata)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=my_datasets,
        folds=FOLDS,
        methods=methods,
        ignore_cache=False,
    )

    # Persist this worker's results so the master can aggregate
    WORKER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WORKER_RESULTS_DIR / f"worker_{worker_idx:02d}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results_lst, f)
    print(f"[Worker {worker_idx}] Done. Results saved to {out_path}")


# ─────────────────────────────────────────────────────────────────
# AGGREGATE MODE: merge all worker results → ELO leaderboard
# ─────────────────────────────────────────────────────────────────
def aggregate():
    import pandas as pd
    from tabarena.nips2025_utils.end_to_end import EndToEnd
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    from tabarena.website.website_format import format_leaderboard

    _, _, task_metadata = get_eligible_datasets()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all worker result pickles
    pkl_files = sorted(WORKER_RESULTS_DIR.glob("worker_*.pkl"))
    if not pkl_files:
        print("ERROR: No worker result files found. Make sure workers completed.")
        sys.exit(1)

    print(f"Aggregating results from {len(pkl_files)} worker files...")
    all_results: list[dict[str, Any]] = []
    for p in pkl_files:
        with open(p, "rb") as f:
            chunk = pickle.load(f)
        print(f"  {p.name}: {len(chunk)} results")
        all_results.extend(chunk)

    print(f"Total results: {len(all_results)}")

    end_to_end = EndToEnd.from_raw(
        results_lst=all_results,
        task_metadata=task_metadata,
        cache=False,
        cache_raw=False,
    )
    end_to_end_results = end_to_end.to_results()

    print(f"\nNew Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=EVAL_DIR,
        only_valid_tasks=True,
        use_model_results=True,
        new_result_prefix="GD_",
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)

    print("\n" + "=" * 80)
    print("GRAPHDRONE TABARENA FULL LEADERBOARD (Binary + Regression, 3 folds)")
    print("=" * 80)
    print(leaderboard_website.to_markdown(index=False))

    # Also save leaderboard as CSV for easy reading
    lb_csv = EVAL_DIR / "tabarena_leaderboard_full.csv"
    leaderboard.to_csv(lb_csv, index=False)
    print(f"\nFull leaderboard saved to: {lb_csv}")

    # Print GraphDrone's row specifically
    gd_row = leaderboard[leaderboard["method"].str.startswith("GD_")]
    if not gd_row.empty:
        print("\n--- GraphDrone Summary ---")
        for _, r in gd_row.iterrows():
            print(f"  Method      : {r['method']}")
            print(f"  ELO         : {r['elo']:.1f}  (+{r['elo+']:.1f} / -{r['elo-']:.1f})")
            print(f"  Rank        : {r['rank']:.1f}")
            print(f"  Win Rate    : {r['winrate']*100:.1f}%")
            print(f"  Improvability: {r['improvability']:.4f}")


# ─────────────────────────────────────────────────────────────────
# MASTER MODE: spawn 8 worker subprocesses, then aggregate
# ─────────────────────────────────────────────────────────────────
def master():
    names, _, _ = get_eligible_datasets()
    print(f"Datasets (binary + regression): {len(names)}")
    print(f"Folds: {FOLDS}  →  {len(names) * len(FOLDS)} total tasks")
    print(f"GPUs: {N_GPUS}  →  ~{len(names) * len(FOLDS) // N_GPUS} tasks/GPU")
    print()

    chunks = split_into_chunks(names, N_GPUS)
    for i, chunk in enumerate(chunks):
        print(f"  GPU {i}: {len(chunk)} datasets → {chunk}")

    print("\nLaunching workers...")
    script = str(ROOT / "scripts" / "run_tabarena_parallel.py")
    procs = []
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    for i in range(N_GPUS):
        if not chunks[i]:
            print(f"  [GPU {i}] No datasets, skipping.")
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        log_path = log_dir / f"worker_{i:02d}.log"
        log_file = open(log_path, "w")
        cmd = [
            sys.executable, script,
            "--worker", str(i),
            "--gpu", str(i),
        ]
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((i, p, log_file, log_path))
        print(f"  [GPU {i}] PID {p.pid} | log: {log_path}")

    print(f"\nAll {len(procs)} workers launched. Waiting for completion...")
    print("(Monitor with: tail -f logs/worker_XX.log)\n")

    failed = []
    for gpu_id, p, log_file, log_path in procs:
        ret = p.wait()
        log_file.close()
        status = "OK" if ret == 0 else f"FAILED (exit {ret})"
        print(f"  [GPU {gpu_id}] {status}")
        if ret != 0:
            failed.append(gpu_id)

    if failed:
        print(f"\nWARNING: Workers on GPU(s) {failed} failed. Check logs/worker_XX.log")
        print("Attempting to aggregate results from successful workers...")

    print("\nAggregating results and computing ELO...")
    aggregate()


# ─────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=None,
                        help="Run as worker N (called by master, do not use directly)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id for this worker (informational; actual GPU set by CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip workers, just aggregate existing results from a previous run")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate()
    elif args.worker is not None:
        run_worker(worker_idx=args.worker, gpu_id=args.gpu)
    else:
        master()
