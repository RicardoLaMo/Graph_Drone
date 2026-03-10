#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aligned foundation baselines across split seeds.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tabr", "tabm", "tabpfn"],
        default=["tabr", "tabm"],
    )
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--tabpfn-max-train-samples", type=int, default=0)
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument("--tabpfn-device", default="cpu")
    parser.add_argument(
        "--tabm-device-policy",
        choices=["auto", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--tabr-device-policy",
        choices=["auto", "cpu"],
        default="auto",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands: list[list[str]] = []
    for split_seed in args.split_seeds:
        if "tabr" in args.models:
            commands.append(
                [
                    sys.executable,
                    "experiments/tab_foundation_compare/scripts/run_tabr_aligned.py",
                    "--seed",
                    str(split_seed),
                    "--device-policy",
                    args.tabr_device_policy,
                    *([] if not args.smoke else ["--smoke"]),
                ]
            )
        if "tabm" in args.models:
            commands.append(
                [
                    sys.executable,
                    "experiments/tabm_california_baseline/scripts/run_tabm_california.py",
                    "--seed",
                    str(split_seed),
                    "--device-policy",
                    args.tabm_device_policy,
                    *([] if not args.smoke else ["--smoke"]),
                ]
            )
        if "tabpfn" in args.models:
            commands.append(
                [
                    sys.executable,
                    "experiments/tab_foundation_compare/scripts/run_tabpfn_aligned.py",
                    "--seed",
                    str(split_seed),
                    "--max-train-samples",
                    str(args.tabpfn_max_train_samples),
                    "--n-estimators",
                    str(args.tabpfn_n_estimators),
                    "--device",
                    args.tabpfn_device,
                    *([] if not args.smoke else ["--smoke"]),
                ]
            )

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
