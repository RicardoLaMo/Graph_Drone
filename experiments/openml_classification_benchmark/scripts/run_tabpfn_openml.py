from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_classification_benchmark.src.openml_tasks import (
    available_dataset_keys,
    build_openml_classification_split,
    dataset_run_tag,
    limit_train_rows,
    split_summary,
)
from src.graphdrone_fit.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabPFN baseline on OpenML classification task splits")
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
        default=REPO_ROOT / "experiments" / "openml_classification_benchmark" / "reports",
    )
    return parser.parse_args()


def main() -> None:
    from tabpfn import TabPFNClassifier

    args = parse_args()
    split = build_openml_classification_split(
        args.dataset,
        repeat=args.repeat,
        fold=args.fold,
        split_seed=args.split_seed,
        smoke=args.smoke,
    )
    split = limit_train_rows(
        split,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )
    if args.device == "cpu" and len(split.X_train) > 1000:
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    model = TabPFNClassifier(
        n_estimators=2 if args.smoke else args.n_estimators,
        random_state=args.seed,
        device=args.device,
        ignore_pretraining_limits=len(split.X_train) > 1000,
        n_preprocessing_jobs=1,
    )
    model.fit(split.X_train, split.y_train)
    val_proba = model.predict_proba(split.X_val).astype(np.float32)
    test_proba = model.predict_proba(split.X_test).astype(np.float32)
    val_metrics = classification_metrics(split.y_val, val_proba, class_labels=tuple(range(len(split.class_labels))))
    test_metrics = classification_metrics(split.y_test, test_proba, class_labels=tuple(range(len(split.class_labels))))

    run_name = dataset_run_tag(args.dataset, repeat=args.repeat, fold=args.fold, smoke=args.smoke)
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "tabpfn_predictions.npz", val_proba=val_proba, test_proba=test_proba)
    payload = {
        "run_name": run_name,
        "model": "TabPFN",
        "seed": args.seed,
        "dataset": split_summary(split),
        "rows": [
            {
                "model": "TabPFN",
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "val_macro_f1": val_metrics["f1_macro"],
                "test_macro_f1": test_metrics["f1_macro"],
                "val_roc_auc": val_metrics.get("roc_auc", val_metrics.get("roc_auc_ovr_macro")),
                "test_roc_auc": test_metrics.get("roc_auc", test_metrics.get("roc_auc_ovr_macro")),
                "val_pr_auc": val_metrics.get("pr_auc", val_metrics.get("pr_auc_ovr_macro")),
                "test_pr_auc": test_metrics.get("pr_auc", test_metrics.get("pr_auc_ovr_macro")),
                "val_log_loss": val_metrics["log_loss"],
                "test_log_loss": test_metrics["log_loss"],
                "notes": "official TabPFN classification baseline",
            }
        ],
        "metrics": {"val": val_metrics, "test": test_metrics},
        "prediction_type": "probs",
        "train_samples_used": int(len(split.X_train)),
        "n_estimators": int(2 if args.smoke else args.n_estimators),
        "device": args.device,
    }
    (output_dir / "tabpfn_results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
