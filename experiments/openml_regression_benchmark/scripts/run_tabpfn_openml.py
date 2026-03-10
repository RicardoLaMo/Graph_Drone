from __future__ import annotations

import argparse
import json
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
from experiments.tab_foundation_compare.src.tabpfn_baseline import evaluate_tabpfn_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabPFN baseline on OpenML regression task splits")
    parser.add_argument("--dataset", choices=available_dataset_keys(), required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_regression_benchmark" / "reports",
    )
    return parser.parse_args()


def write_report(path: Path, *, run_name: str, dataset_summary: dict[str, object], metrics: dict) -> None:
    lines = [
        "# TabPFN OpenML Report",
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
        f"- n_estimators `{metrics['n_estimators']}`",
        f"- device `{metrics['device']}`",
    ]
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
    if args.smoke and max_train_samples is None:
        max_train_samples = 1024
    metrics = evaluate_tabpfn_regression(
        split=split,
        seed=args.seed,
        max_train_samples=max_train_samples,
        n_estimators=2 if args.smoke else args.n_estimators,
        max_eval_rows=512 if args.smoke else None,
        device=args.device,
    )

    run_name = dataset_run_tag(args.dataset, repeat=args.repeat, fold=args.fold, smoke=args.smoke)
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "model": "TabPFN",
        "seed": args.seed,
        "dataset": split_summary(split),
        "metrics": metrics,
    }
    (output_dir / "tabpfn_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(output_dir / "tabpfn_report.md", run_name=run_name, dataset_summary=payload["dataset"], metrics=metrics)


if __name__ == "__main__":
    main()
