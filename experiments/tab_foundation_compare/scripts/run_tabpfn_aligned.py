from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tab_foundation_compare.src.aligned_california import build_aligned_california_split
from experiments.tab_foundation_compare.src.runtime_support import (
    ALIGNED_CANONICAL_SEED,
    seed_aware_run_name,
)
from experiments.tab_foundation_compare.src.tabpfn_baseline import (
    evaluate_tabpfn_regression,
    write_tabpfn_json,
    write_tabpfn_report,
)

TABR_ON_OUR_SPLIT = 0.3829
TABM_ON_OUR_SPLIT = 0.4290
A6F_RMSE = 0.4063


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "tab_foundation_compare",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Use full aligned train split when <= 0; otherwise subsample train rows to this cap.",
    )
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _write_comparison_csv(
    path: Path,
    *,
    test_rmse: float,
    val_rmse: float,
    train_samples: int,
    device: str,
) -> None:
    runtime_note = f"TabPFN {device} aligned to repo California split; train cap={train_samples}"
    rows = [
        {
            "stage": "C1",
            "model": "TabPFN_on_our_split",
            "test_rmse": f"{test_rmse:.10f}",
            "val_rmse": f"{val_rmse:.10f}",
            "best_step_or_epoch": "",
            "source": "current aligned run snapshot",
            "notes": runtime_note,
        },
        {
            "stage": "C1",
            "model": "TabR_on_our_split",
            "test_rmse": f"{TABR_ON_OUR_SPLIT:.10f}",
            "val_rmse": "",
            "best_step_or_epoch": "",
            "source": "prior aligned run snapshot",
            "notes": "Reference from comparison branch",
        },
        {
            "stage": "C1",
            "model": "TabM_on_our_split",
            "test_rmse": f"{TABM_ON_OUR_SPLIT:.10f}",
            "val_rmse": "",
            "best_step_or_epoch": "",
            "source": "prior aligned run snapshot",
            "notes": "Reference from comparison branch",
        },
        {
            "stage": "C1",
            "model": "A6f",
            "test_rmse": f"{A6F_RMSE:.10f}",
            "val_rmse": "",
            "best_step_or_epoch": "",
            "source": "mv-tabr-gora branch-local reference",
            "notes": "Current MV-TabR-GoRA champion",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    art_dir = args.output_root / "artifacts"
    rep_dir = args.output_root / "reports"
    art_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    split = build_aligned_california_split(seed=args.seed)
    max_eval_rows = 128 if args.smoke else None
    full_train = args.max_train_samples <= 0
    if args.smoke:
        max_train_samples = 512 if full_train else min(args.max_train_samples, 512)
    else:
        max_train_samples = None if full_train else args.max_train_samples
    n_estimators = 2 if args.smoke else args.n_estimators

    metrics = evaluate_tabpfn_regression(
        split=split,
        seed=args.seed,
        max_train_samples=max_train_samples,
        n_estimators=n_estimators,
        max_eval_rows=max_eval_rows,
        device=args.device,
    )
    if args.smoke:
        base_name = "tabpfn_aligned"
    elif full_train:
        base_name = "tabpfn_aligned__full"
    else:
        base_name = f"tabpfn_aligned__train{metrics['train_samples_used']}"
    run_name = seed_aware_run_name(
        base_name,
        args.seed,
        canonical_seed=ALIGNED_CANONICAL_SEED,
        smoke=args.smoke,
    )
    write_tabpfn_json(rep_dir / f"{run_name}.json", metrics)
    write_tabpfn_report(
        rep_dir / f"{run_name}.md",
        metrics,
        tabr_rmse=TABR_ON_OUR_SPLIT,
        tabm_rmse=TABM_ON_OUR_SPLIT,
        a6f_rmse=A6F_RMSE,
    )
    _write_comparison_csv(
        art_dir / ("tabpfn_comparison__smoke.csv" if args.smoke else "tabpfn_comparison.csv"),
        test_rmse=metrics["test"]["rmse"],
        val_rmse=metrics["val"]["rmse"],
        train_samples=metrics["train_samples_used"],
        device=metrics["device"],
    )
    print(
        {
            "model": "TabPFN_on_our_split",
            "smoke": args.smoke,
            "test_rmse": round(metrics["test"]["rmse"], 4),
            "val_rmse": round(metrics["val"]["rmse"], 4),
            "train_samples_used": metrics["train_samples_used"],
            "n_estimators": metrics["n_estimators"],
        }
    )


if __name__ == "__main__":
    main()
