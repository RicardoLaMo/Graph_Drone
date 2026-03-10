from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.src.foundation_config import prepare_foundation_config
from experiments.openml_regression_benchmark.src.openml_tasks import (
    available_dataset_keys,
    build_openml_regression_split,
    dataset_run_tag,
    split_summary,
    write_foundation_dataset,
)
from experiments.tab_foundation_compare.src.runtime_support import (
    default_foundation_python,
    default_tabm_upstream_root,
)
from experiments.tabm_california_baseline.scripts.run_tabm_california import _patch_tabm_for_modern_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabM on OpenML regression task splits")
    parser.add_argument("--dataset", choices=available_dataset_keys(), required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=default_tabm_upstream_root(REPO_ROOT),
    )
    parser.add_argument(
        "--venv-python",
        type=Path,
        default=default_foundation_python(REPO_ROOT),
    )
    parser.add_argument("--config", default="auto")
    parser.add_argument("--device-policy", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = build_openml_regression_split(
        args.dataset,
        repeat=args.repeat,
        fold=args.fold,
        split_seed=args.split_seed,
        smoke=args.smoke,
    )
    run_name = dataset_run_tag(args.dataset, repeat=args.repeat, fold=args.fold, smoke=args.smoke)
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    local_data_dir = output_dir / f"tabm_dataset__{run_name}"
    write_foundation_dataset(local_data_dir, split)

    resolved_config = (
        "diamond/0-evaluation/0"
        if args.config == "auto" and split.X_cat_train is not None
        else "house/0-evaluation/0"
        if args.config == "auto"
        else args.config
    )
    source_config = args.upstream_root / "exp" / "tabm" / f"{resolved_config}.toml"
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    local_config = output_dir / "tabm_config.toml"
    prepare_foundation_config(
        source_config=source_config,
        output_config=local_config,
        data_path=str(local_data_dir),
        seed=args.seed,
        smoke=args.smoke,
        amp=False,
        cat_policy="ordinal" if split.X_cat_train is not None else None,
    )

    _patch_tabm_for_modern_torch(args.upstream_root)

    env = os.environ.copy()
    if args.device_policy == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONFAULTHANDLER", "1")

    stdout_path = output_dir / "tabm.stdout.log"
    stderr_path = output_dir / "tabm.stderr.log"
    started_at = time.time()
    command = [str(args.venv_python), "bin/model.py", str(local_config), "--force"]
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        result = subprocess.run(
            command,
            cwd=args.upstream_root,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    run_output_dir = local_config.with_suffix("")
    upstream_report = json.loads((run_output_dir / "report.json").read_text())
    payload = {
        "run_name": run_name,
        "model": "TabM",
        "seed": args.seed,
        "dataset": split_summary(split),
        "config": resolved_config,
        "device_policy": args.device_policy,
        "duration_seconds": round(time.time() - started_at, 3),
        "metrics": upstream_report["metrics"],
        "best_step": upstream_report.get("best_step"),
        "upstream_report_path": str(run_output_dir / "report.json"),
    }
    (output_dir / "tabm_results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
