#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenML regression benchmark suite.")
    parser.add_argument("--datasets", nargs="+", default=["miami_housing", "houses"])
    parser.add_argument("--models", nargs="+", default=["graphdrone", "tabr", "tabm", "tabpfn"])
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--graphdrone-device", default="")
    parser.add_argument("--graphdrone-max-gpus", type=int, default=4)
    parser.add_argument("--graphdrone-parallel-workers", type=int, default=0)
    parser.add_argument("--tabr-device-policy", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--tabm-device-policy", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--tabpfn-device", default="cuda")
    parser.add_argument("--tabpfn-max-train-samples", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "openml_regression_benchmark")
    return parser.parse_args()


def _free_gpu_indices(max_count: int) -> list[int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    candidates: list[int] = []
    for line in result.stdout.strip().splitlines():
        index_text, memory_used_text, utilization_text = [chunk.strip() for chunk in line.split(",")]
        if int(memory_used_text) <= 1024 and int(utilization_text) <= 10:
            candidates.append(int(index_text))
    candidates.sort(reverse=True)
    return candidates[:max_count]


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def _single_gpu_env() -> dict[str, str]:
    env = dict(os.environ)
    free_gpus = _free_gpu_indices(max_count=1)
    if free_gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
    return env


def main() -> None:
    args = parse_args()
    summary_rows: list[dict[str, object]] = []
    for dataset_key in args.datasets:
        for fold in args.folds:
            for model in args.models:
                if model == "graphdrone":
                    graphdrone_device = args.graphdrone_device
                    if not graphdrone_device:
                        free_gpus = _free_gpu_indices(max_count=args.graphdrone_max_gpus)
                        graphdrone_device = ",".join(f"cuda:{idx}" for idx in free_gpus) if free_gpus else "auto"
                    command = [
                        sys.executable,
                        "experiments/openml_regression_benchmark/scripts/run_graphdrone_openml.py",
                        "--dataset-key",
                        dataset_key,
                        "--repeat",
                        str(args.repeat),
                        "--fold",
                        str(fold),
                        "--val-seed",
                        str(args.val_seed),
                        "--device",
                        graphdrone_device,
                        "--parallel-workers",
                        str(args.graphdrone_parallel_workers or max(1, graphdrone_device.count("cuda:"))),
                    ]
                    if args.smoke:
                        command.append("--smoke")
                    _run(command)
                elif model == "tabr":
                    command = [
                        sys.executable,
                        "experiments/openml_regression_benchmark/scripts/run_tabr_openml.py",
                        "--dataset-key",
                        dataset_key,
                        "--repeat",
                        str(args.repeat),
                        "--fold",
                        str(fold),
                        "--val-seed",
                        str(args.val_seed),
                        "--device-policy",
                        args.tabr_device_policy,
                    ]
                    if args.smoke:
                        command.append("--smoke")
                    _run(command, env=_single_gpu_env() if args.tabr_device_policy == "auto" else None)
                elif model == "tabm":
                    command = [
                        sys.executable,
                        "experiments/openml_regression_benchmark/scripts/run_tabm_openml.py",
                        "--dataset-key",
                        dataset_key,
                        "--repeat",
                        str(args.repeat),
                        "--fold",
                        str(fold),
                        "--val-seed",
                        str(args.val_seed),
                        "--device-policy",
                        args.tabm_device_policy,
                    ]
                    if args.smoke:
                        command.append("--smoke")
                    _run(command, env=_single_gpu_env() if args.tabm_device_policy == "auto" else None)
                elif model == "tabpfn":
                    env = _single_gpu_env()
                    command = [
                        sys.executable,
                        "experiments/openml_regression_benchmark/scripts/run_tabpfn_openml.py",
                        "--dataset-key",
                        dataset_key,
                        "--repeat",
                        str(args.repeat),
                        "--fold",
                        str(fold),
                        "--val-seed",
                        str(args.val_seed),
                        "--max-train-samples",
                        str(args.tabpfn_max_train_samples),
                        "--device",
                        args.tabpfn_device,
                    ]
                    if args.smoke:
                        command.append("--smoke")
                    _run(command, env=env)
                summary_rows.append({"dataset": dataset_key, "repeat": args.repeat, "fold": fold, "model": model})

    output_path = args.output_root / "artifacts" / "suite_runs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "repeat", "fold", "model"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
