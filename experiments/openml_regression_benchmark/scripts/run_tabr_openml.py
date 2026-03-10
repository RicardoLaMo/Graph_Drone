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
from experiments.tab_foundation_compare.scripts.run_tabr_aligned import _patch_tabr_for_modern_env
from experiments.tab_foundation_compare.src.runtime_support import (
    default_foundation_python,
    default_single_gpu_cuda_visible_devices,
    default_tabr_upstream_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabR on OpenML regression task splits")
    parser.add_argument("--dataset", choices=available_dataset_keys(), required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=default_tabr_upstream_root(REPO_ROOT),
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


def resolve_upstream_report_path(run_output_dir: Path) -> Path | None:
    expected = run_output_dir / "report.json"
    if expected.exists():
        return expected
    candidates = sorted(
        run_output_dir.parent.glob(f"{run_output_dir.name}*/report.json"),
        key=lambda path: str(path),
    )
    return candidates[0] if candidates else None


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
    local_data_dir = output_dir / f"tabr_dataset__{run_name}"
    write_foundation_dataset(local_data_dir, split)

    resolved_config = (
        "diamond/0-evaluation/0"
        if args.config == "auto" and split.X_cat_train is not None
        else "house/0-evaluation/0"
        if args.config == "auto"
        else args.config
    )
    source_config = args.upstream_root / "exp" / "tabr" / f"{resolved_config}.toml"
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    local_config = output_dir / "tabr_config.toml"
    prepare_foundation_config(
        source_config=source_config,
        output_config=local_config,
        data_path=str(local_data_dir),
        seed=args.seed,
        smoke=args.smoke,
        cat_policy="ordinal" if split.X_cat_train is not None else None,
    )

    _patch_tabr_for_modern_env(args.upstream_root)

    env = os.environ.copy()
    env["PROJECT_DIR"] = str(args.upstream_root)
    if args.device_policy == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", default_single_gpu_cuda_visible_devices())
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("TQDM_DISABLE", "1")

    stdout_path = output_dir / "tabr.stdout.log"
    stderr_path = output_dir / "tabr.stderr.log"
    started_at = time.time()
    command = [str(args.venv_python), "bin/tabr.py", str(local_config), "--force"]
    run_output_dir = local_config.with_suffix("")
    upstream_report_path: Path | None = None
    result = None
    for attempt in range(2):
        with stdout_path.open("a") as stdout_handle, stderr_path.open("a") as stderr_handle:
            if attempt:
                stderr_handle.write("[wrapper] retrying TabR because upstream report.json was missing after a zero exit code\n")
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
        upstream_report_path = resolve_upstream_report_path(run_output_dir)
        if upstream_report_path is not None:
            break
        time.sleep(1.0)

    if upstream_report_path is None:
        raise FileNotFoundError(
            f"TabR finished with exit code 0 but report.json was not found under {run_output_dir}"
        )
    upstream_report = json.loads(upstream_report_path.read_text())
    payload = {
        "run_name": run_name,
        "model": "TabR",
        "seed": args.seed,
        "dataset": split_summary(split),
        "config": resolved_config,
        "device_policy": args.device_policy,
        "duration_seconds": round(time.time() - started_at, 3),
        "metrics": upstream_report["metrics"],
        "best_epoch": upstream_report.get("best_epoch"),
        "upstream_report_path": str(upstream_report_path),
    }
    (output_dir / "tabr_results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
