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

EXPNAME = str(ROOT / "experiments" / "tabarena_p0ab")
EVAL_DIR = ROOT / "eval" / "tabarena_p0ab"
WORKER_RESULTS_DIR = ROOT / "experiments" / "tabarena_p0ab" / "_worker_results"
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


def get_incomplete_datasets():
    """Return names of datasets that don't yet have all 3 folds completed."""
    names, tids, task_metadata = get_eligible_datasets()
    exp_dir = Path(EXPNAME) / "data" / "GraphDrone_P0AB"
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    ctx = TabArenaContext()
    meta = ctx.task_metadata
    name_to_tid = dict(zip(meta["name"], meta["tid"].astype(int)))
    incomplete = []
    for name in names:
        tid = name_to_tid.get(name)
        if tid is None:
            incomplete.append(name)
            continue
        all_done = all(
            (exp_dir / str(tid) / str(fold) / "results.pkl").exists()
            for fold in FOLDS
        )
        if not all_done:
            incomplete.append(name)
    return incomplete, task_metadata


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
def run_worker(worker_idx: int, gpu_id: int, explicit_datasets: list[str] | None = None):
    from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    from graphdrone_fit.adapters.tabarena import GraphDroneTabArenaAdapter

    if explicit_datasets is not None:
        # Retry mode: datasets passed explicitly by master
        _, _, task_metadata = get_eligible_datasets()
        my_datasets = explicit_datasets
    else:
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
            name="GraphDrone_P0AB",
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
def master(retry: bool = False):
    if retry:
        incomplete, _ = get_incomplete_datasets()
        names_to_run = incomplete
        print(f"RETRY MODE: {len(names_to_run)} incomplete datasets")
        log_suffix = "_retry"
    else:
        names_to_run, _, _ = get_eligible_datasets()
        print(f"Datasets (binary + regression): {len(names_to_run)}")
        log_suffix = ""

    print(f"Folds: {FOLDS}  →  {len(names_to_run) * len(FOLDS)} total tasks")
    n_workers = min(N_GPUS, len(names_to_run))
    print(f"GPUs: {n_workers}  →  ~{max(1, len(names_to_run) * len(FOLDS) // n_workers)} tasks/GPU")
    print()

    chunks = split_into_chunks(names_to_run, n_workers)
    for i, chunk in enumerate(chunks):
        print(f"  GPU {i}: {len(chunk)} datasets → {chunk}")

    print("\nLaunching workers...")
    script = str(ROOT / "scripts" / "run_tabarena_parallel.py")
    procs = []
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    for i in range(n_workers):
        if not chunks[i]:
            print(f"  [GPU {i}] No datasets, skipping.")
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        # Cap OpenBLAS/OMP threads per worker to avoid hitting the 128-thread limit
        # when 8 workers each try to use all cores simultaneously.
        env["OPENBLAS_NUM_THREADS"] = "4"
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        log_path = log_dir / f"worker_{i:02d}{log_suffix}.log"
        log_file = open(log_path, "w")
        cmd = [
            sys.executable, script,
            "--worker", str(i),
            "--gpu", str(i),
            "--datasets", ",".join(chunks[i]),
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
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names for this worker (overrides chunk assignment)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip workers, just aggregate existing results from a previous run")
    parser.add_argument("--retry", action="store_true",
                        help="Re-run only incomplete datasets (after fixing failures)")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate()
    elif args.worker is not None:
        explicit = args.datasets.split(",") if args.datasets else None
        run_worker(worker_idx=args.worker, gpu_id=args.gpu, explicit_datasets=explicit)
    elif args.retry:
        master(retry=True)
    else:
        master(retry=False)
