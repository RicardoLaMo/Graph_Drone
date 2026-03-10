from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.src.openml_tasks import (
    available_dataset_keys,
    build_openml_regression_split,
    dataset_run_tag,
    split_summary,
)
from experiments.tab_foundation_compare.src.autogluon_baseline import predict_autogluon_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoGluon baseline on OpenML regression task splits")
    parser.add_argument("--dataset", choices=available_dataset_keys(), required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--num-cpus", default="4")
    parser.add_argument("--num-gpus", default="auto")
    parser.add_argument("--fit-strategy", choices=["sequential", "parallel"], default="sequential")
    parser.add_argument("--device-policy", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports",
    )
    return parser.parse_args()


def parse_resource_arg(value: str) -> int | str:
    if value == "auto":
        return value
    return int(value)


def write_report(path: Path, *, run_name: str, dataset_summary: dict[str, object], metrics: dict) -> None:
    lines = [
        "# AutoGluon OpenML Report",
        "",
        f"- run: `{run_name}`",
        f"- dataset: `{dataset_summary['dataset_name']}` ({dataset_summary['dataset_key']})",
        f"- OpenML dataset id: `{dataset_summary['dataset_id']}`",
        f"- OpenML task id: `{dataset_summary['task_id']}`",
        f"- repeat / fold: `{dataset_summary['repeat']}` / `{dataset_summary['fold']}`",
        f"- rows train / val / test: `{dataset_summary['train_rows']}` / `{dataset_summary['val_rows']}` / `{dataset_summary['test_rows']}`",
        "",
        "## Result",
        "",
        f"- test RMSE `{metrics['test']['rmse']:.4f}`",
        f"- val RMSE `{metrics['val']['rmse']:.4f}`",
        f"- test MAE `{metrics['test']['mae']:.4f}`",
        f"- test R2 `{metrics['test']['r2']:.4f}`",
        f"- train samples used `{metrics['train_samples_used']}`",
        f"- time_limit `{metrics['time_limit']}`",
        f"- presets `{metrics['presets']}`",
        f"- num_cpus `{metrics['num_cpus']}` / num_gpus `{metrics['num_gpus']}`",
        f"- fit_strategy `{metrics['fit_strategy']}`",
        f"- duration_seconds `{metrics['duration_seconds']}`",
        f"- best_model `{metrics['best_model']}`",
        "",
        "## Leaderboard Head",
        "",
    ]
    for row in metrics["leaderboard_test_head"]:
        lines.append(f"- {row}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    split = build_openml_regression_split(
        args.dataset,
        repeat=args.repeat,
        fold=args.fold,
        split_seed=args.split_seed,
        smoke=args.smoke,
    )

    max_train_samples = None if args.max_train_samples <= 0 else args.max_train_samples
    time_limit = 45.0 if args.smoke else args.time_limit
    if args.smoke and max_train_samples is None:
        max_train_samples = 1024

    num_cpus = parse_resource_arg(args.num_cpus)
    num_gpus = 0 if args.device_policy == "cpu" else parse_resource_arg(args.num_gpus)
    if num_gpus == "auto":
        visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        num_gpus = len(visible_devices) if visible_devices else 0

    run_name = dataset_run_tag(args.dataset, repeat=args.repeat, fold=args.fold, smoke=args.smoke)
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = predict_autogluon_regression(
        split=split,
        seed=args.seed,
        max_train_samples=max_train_samples,
        time_limit=time_limit,
        presets=args.presets,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        fit_strategy=args.fit_strategy,
        artifacts_path=output_dir / "autogluon_artifacts",
        verbosity=1 if args.smoke else 2,
    ).metrics

    payload = {
        "run_name": run_name,
        "model": "AutoGluon",
        "seed": args.seed,
        "dataset": split_summary(split),
        "metrics": metrics,
    }
    (output_dir / "autogluon_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(output_dir / "autogluon_report.md", run_name=run_name, dataset_summary=payload["dataset"], metrics=metrics)


if __name__ == "__main__":
    main()
