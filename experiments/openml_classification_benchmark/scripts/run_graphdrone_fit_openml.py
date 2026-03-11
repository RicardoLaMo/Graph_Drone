from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_classification_benchmark.src.graphdrone_fit_adapter import (
    build_classification_expert_plan,
    build_classification_quality_encodings,
)
from experiments.openml_classification_benchmark.src.openml_tasks import (
    available_dataset_keys,
    build_openml_classification_split,
    dataset_run_tag,
    limit_train_rows,
    split_summary,
)
from src.graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig, classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphDrone.fit classification benchmark runner")
    parser.add_argument("--dataset", choices=available_dataset_keys(), required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_classification_benchmark" / "reports",
    )
    return parser.parse_args()


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)

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
        max_train_samples=args.max_train_samples or (1024 if args.smoke else 0),
        seed=args.seed,
    )
    expert_plan = build_classification_expert_plan(
        split,
        seed=args.seed,
        n_estimators=2 if args.smoke else args.n_estimators,
        n_preprocessing_jobs=args.n_preprocessing_jobs,
        device=args.device,
    )
    model = GraphDrone(
        GraphDroneConfig(
            portfolio=None,
            full_expert_id=expert_plan.full_expert_id,
            task_type="classification",
            class_labels=split.class_labels,
            router=SetRouterConfig(kind="contextual_sparse_mlp", sparse_top_k=2),
        )
    )
    model.fit(split.X_train, split.y_train, expert_specs=expert_plan.specs)
    quality_encodings = build_classification_quality_encodings(split, expert_plan)
    router_fit_summary = model.fit_router(
        split.X_val,
        split.y_val,
        quality_features=quality_encodings["val"],
    )

    batch_val = model.predict_experts(split.X_val)
    batch_test = model.predict_experts(split.X_test)
    graphdrone_val = model.predict(
        split.X_val,
        quality_features=quality_encodings["val"],
        return_diagnostics=True,
    )
    graphdrone_test = model.predict(
        split.X_test,
        quality_features=quality_encodings["test"],
        return_diagnostics=True,
    )
    if graphdrone_val.probabilities is None or graphdrone_test.probabilities is None:
        raise RuntimeError("Classification GraphDrone run did not produce probabilities")

    internal_rows: list[dict[str, object]] = []
    for idx, expert_id in enumerate(batch_val.expert_ids):
        expert_val_proba = batch_val.predictions[:, idx, :]
        expert_test_proba = batch_test.predictions[:, idx, :]
        expert_val_pred = expert_val_proba.argmax(axis=1).astype(int)
        expert_test_pred = expert_test_proba.argmax(axis=1).astype(int)
        val_metrics = classification_metrics(
            split.y_val,
            expert_val_pred,
            expert_val_proba,
            class_labels=split.class_labels,
        )
        test_metrics = classification_metrics(
            split.y_test,
            expert_test_pred,
            expert_test_proba,
            class_labels=split.class_labels,
        )
        internal_rows.append(
            {
                "expert_id": expert_id,
                "role": expert_plan.expert_role_map.get(expert_id, expert_id),
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "val_log_loss": val_metrics["log_loss"],
                "test_log_loss": test_metrics["log_loss"],
                "notes": "internal expert diagnostics only; not a public benchmark row",
            }
        )

    val_metrics = classification_metrics(
        split.y_val,
        graphdrone_val.predictions,
        graphdrone_val.probabilities,
        class_labels=split.class_labels,
    )
    test_metrics = classification_metrics(
        split.y_test,
        graphdrone_test.predictions,
        graphdrone_test.probabilities,
        class_labels=split.class_labels,
    )
    rows = [
        {
            "model": "GraphDrone",
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "val_log_loss": val_metrics["log_loss"],
            "test_log_loss": test_metrics["log_loss"],
            "notes": "contextual_sparse_mlp over generic internal experts",
        }
    ]

    run_name = dataset_run_tag(args.dataset, repeat=args.repeat, fold=args.fold, smoke=args.smoke)
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(output_dir / "metrics.csv", rows)
    np.savez(
        output_dir / "graphdrone_predictions.npz",
        val_pred=np.asarray(graphdrone_val.predictions),
        test_pred=np.asarray(graphdrone_test.predictions),
        val_proba=np.asarray(graphdrone_val.probabilities),
        test_proba=np.asarray(graphdrone_test.probabilities),
    )
    report = {
        "run_name": run_name,
        "dataset": split_summary(split),
        "runtime": {
            "requested_device": args.device,
        },
        "rows": rows,
        "quality_feature_names": list(quality_encodings["val"].feature_names),
        "router_fit_summary": router_fit_summary,
        "graphdrone_val_diagnostics": graphdrone_val.diagnostics,
        "graphdrone_test_diagnostics": graphdrone_test.diagnostics,
    }
    internal_payload = {
        "run_name": run_name,
        "dataset": split_summary(split),
        "rows": internal_rows,
        "expert_ids": list(batch_test.expert_ids),
        "expert_role_map": expert_plan.expert_role_map,
        "descriptors": [descriptor.to_dict() for descriptor in expert_plan.descriptors],
    }
    (output_dir / "graphdrone_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "graphdrone_internal_experts.json").write_text(json.dumps(internal_payload, indent=2) + "\n")
    (output_dir / "graphdrone_fit_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
