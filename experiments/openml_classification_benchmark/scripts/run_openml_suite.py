from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_classification_benchmark.src.openml_tasks import available_dataset_keys


MODEL_SCRIPTS = {
    "GraphDrone": SCRIPT_DIR / "run_graphdrone_fit_openml.py",
    "TabPFN": SCRIPT_DIR / "run_tabpfn_openml.py",
    "TabR": SCRIPT_DIR / "run_tabr_openml.py",
    "TabM": SCRIPT_DIR / "run_tabm_openml.py",
}


def resolve_shared_python() -> Path:
    for candidate_root in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = candidate_root / ".venv-h200" / "bin" / "python"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate shared .venv-h200/bin/python from the worktree path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue OpenML classification runs onto all currently free H200 GPUs")
    parser.add_argument("--datasets", nargs="+", default=available_dataset_keys())
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=["GraphDrone", "TabPFN", "TabM"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--graphdrone-max-train-samples", type=int, default=0)
    parser.add_argument("--tabpfn-max-train-samples", type=int, default=0)
    parser.add_argument("--graphdrone-router-token-encoder", choices=("field_aware", "flat"), default="field_aware")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--gpu-order", choices=["high-first", "low-first"], default=os.environ.get("GRAPH_DRONE_GPU_ORDER", "high-first"))
    parser.add_argument("--gpu-memory-free-threshold-mib", type=int, default=int(os.environ.get("GRAPHDRONE_OPENML_GPU_MEMORY_FREE_THRESHOLD_MIB", "4096")))
    parser.add_argument("--gpu-util-free-threshold", type=int, default=int(os.environ.get("GRAPHDRONE_OPENML_GPU_UTIL_FREE_THRESHOLD", "10")))
    parser.add_argument("--max-concurrent-jobs", type=int, default=int(os.environ.get("GRAPHDRONE_OPENML_MAX_CONCURRENT_JOBS", "0")))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_classification_benchmark" / "reports",
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def parse_gpu_indices(text: str) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        idx = int(stripped)
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


def ordered_gpu_indices(indices: list[int], *, gpu_order: str) -> list[int]:
    return sorted(indices, reverse=gpu_order == "high-first")


def parse_gpu_status(output: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in output.strip().splitlines():
        idx_str, mem_used_str, util_str = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(idx_str),
                "memory_used_mib": int(mem_used_str),
                "utilization_gpu": int(util_str),
            }
        )
    return rows


def query_gpu_status() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    return parse_gpu_status(output)


def resolve_gpu_pool(
    *,
    gpu_arg: str,
    env_gpu_pool: str,
    gpu_order: str,
    gpu_status: list[dict[str, int]],
) -> list[int]:
    if gpu_arg == "auto":
        indices = parse_gpu_indices(env_gpu_pool) if env_gpu_pool.strip() else [row["index"] for row in gpu_status]
    elif gpu_arg == "all":
        indices = [row["index"] for row in gpu_status]
    else:
        indices = parse_gpu_indices(gpu_arg)
    return ordered_gpu_indices(indices, gpu_order=gpu_order)


def discover_free_gpus(
    *,
    gpu_pool: list[int],
    gpu_status: list[dict[str, int]],
    gpu_order: str,
    memory_free_threshold_mib: int,
    util_free_threshold: int,
) -> list[int]:
    allowed = set(gpu_pool)
    free: list[int] = []
    for row in gpu_status:
        idx = row["index"]
        if idx not in allowed:
            continue
        if row["memory_used_mib"] <= memory_free_threshold_mib and row["utilization_gpu"] <= util_free_threshold:
            free.append(idx)
    return ordered_gpu_indices(free, gpu_order=gpu_order)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    shared_python = resolve_shared_python()

    gpu_status = query_gpu_status()
    gpu_pool = resolve_gpu_pool(
        gpu_arg=args.gpus,
        env_gpu_pool=os.environ.get("GRAPH_DRONE_GPU_POOL", ""),
        gpu_order=args.gpu_order,
        gpu_status=gpu_status,
    )
    if not gpu_pool:
        raise SystemExit("No GPUs resolved for the requested pool")

    tasks = deque()
    for dataset in args.datasets:
        for fold in args.folds:
            for model in args.models:
                tasks.append({"dataset": dataset, "fold": fold, "model": model})

    running: dict[int, dict[str, object]] = {}
    manifest: list[dict[str, object]] = []

    while tasks or running:
        finished: list[int] = []
        for gpu_index, info in running.items():
            proc = info["proc"]
            if proc.poll() is not None:
                manifest.append(
                    {
                        "gpu": gpu_index,
                        "dataset": info["dataset"],
                        "fold": info["fold"],
                        "model": info["model"],
                        "returncode": proc.returncode,
                        "command": info["command"],
                    }
                )
                finished.append(gpu_index)
        for gpu_index in finished:
            running.pop(gpu_index, None)

        occupied = set(running.keys())
        available_pool = discover_free_gpus(
            gpu_pool=gpu_pool,
            gpu_status=query_gpu_status(),
            gpu_order=args.gpu_order,
            memory_free_threshold_mib=args.gpu_memory_free_threshold_mib,
            util_free_threshold=args.gpu_util_free_threshold,
        )
        available = [gpu for gpu in available_pool if gpu not in occupied]
        capacity = args.max_concurrent_jobs if args.max_concurrent_jobs > 0 else len(gpu_pool)
        while tasks and available and len(running) < capacity:
            task = tasks.popleft()
            gpu_index = available.pop(0)
            model = str(task["model"])
            script = MODEL_SCRIPTS[model]
            cmd = [
                str(shared_python),
                str(script),
                "--dataset",
                str(task["dataset"]),
                "--repeat",
                str(args.repeat),
                "--fold",
                str(task["fold"]),
                "--seed",
                str(args.seed),
                "--split-seed",
                str(args.split_seed),
                "--output-root",
                str(args.output_root),
            ]
            if model in {"GraphDrone", "TabPFN"}:
                cmd.extend(["--device", "auto"])
            if model == "GraphDrone" and args.graphdrone_max_train_samples > 0:
                cmd.extend(["--max-train-samples", str(args.graphdrone_max_train_samples)])
            if model == "GraphDrone":
                cmd.extend(["--router-token-encoder", args.graphdrone_router_token_encoder])
            if model == "TabPFN" and args.tabpfn_max_train_samples > 0:
                cmd.extend(["--max-train-samples", str(args.tabpfn_max_train_samples)])
            if args.smoke:
                cmd.append("--smoke")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            log_path = args.output_root / f"queue__{task['dataset']}__{task['model']}__fold{task['fold']}.log"
            log_handle = log_path.open("w")
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu_index] = {
                "proc": proc,
                "dataset": task["dataset"],
                "fold": task["fold"],
                "model": task["model"],
                "command": cmd,
            }

        if tasks or running:
            time.sleep(args.poll_seconds)

    (args.output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(
        [str(shared_python), str(SCRIPT_DIR / "summarize_openml_suite.py"), "--input-root", str(args.output_root)],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
