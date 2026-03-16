#!/usr/bin/env python3
"""
H200 Parallel Validation - Run extended benchmark across multiple GPUs

Launches extended benchmark (4 datasets: Wine, Breast Cancer, Digits, Segment)
in parallel on GPUs 1-5 to demonstrate H200 multi-GPU optimization.

Usage:
    python scripts/run_h200_validation_parallel.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from tabpfn import TabPFNClassifier

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | GPU %(gpu_id)d | %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    if n_classes == 2:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        pr_auc = average_precision_score(y_true, y_proba[:, 1])
    else:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        roc_auc = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
        pr_auc = average_precision_score(y_true_bin, y_proba, average="macro")

    return {"acc": acc, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc}


def load_segment():
    """Load Segment dataset from OpenML."""
    from sklearn.datasets import fetch_openml
    logger.info("Fetching Segment dataset (OpenML)...", extra={"gpu_id": 0})
    df = fetch_openml(data_id=40984, as_frame=True, parser='auto')
    X = df.data.select_dtypes(include=[np.number]).values
    le = LabelEncoder()
    y = le.fit_transform(df.target)
    return type('Data', (), {'data': X, 'target': y})


def run_benchmark_on_gpu(gpu_id: int, dataset_name: str) -> dict:
    """Run benchmark on specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start_time = time.time()

    try:
        logger.info(f"Starting {dataset_name} benchmark", extra={"gpu_id": gpu_id})

        # Load dataset
        if dataset_name == "Wine":
            data = load_wine()
            n_classes = 3
            max_samples = 500
        elif dataset_name == "Breast Cancer":
            data = load_breast_cancer()
            n_classes = 2
            max_samples = 1000
        elif dataset_name == "Digits":
            data = load_digits()
            n_classes = 10
            max_samples = 1500
        elif dataset_name == "Segment":
            data = load_segment()
            n_classes = 7
            max_samples = 2000
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        X, y = data.data, data.target

        # Subsample if needed
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}", extra={"gpu_id": gpu_id})

        # Train GraphDrone
        logger.info("Training GraphDrone...", extra={"gpu_id": gpu_id})
        gd_config = GraphDroneConfig(
            problem_type="classification",
            n_classes=n_classes,
            router=SetRouterConfig(kind="contextual_transformer_router")
        )
        gd_model = GraphDrone(gd_config)
        gd_model.fit(X_train, y_train)
        gd_proba = gd_model.predict(X_test)
        gd_pred = np.argmax(gd_proba, axis=1)
        gd_metrics = compute_metrics(y_test, gd_pred, gd_proba, n_classes)

        # Train TabPFN
        logger.info("Training TabPFN...", extra={"gpu_id": gpu_id})
        tabpfn = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=1)
        tabpfn.fit(X_train, y_train)
        tp_proba = tabpfn.predict_proba(X_test)
        tp_pred = np.argmax(tp_proba, axis=1)
        tp_metrics = compute_metrics(y_test, tp_pred, tp_proba, n_classes)

        result = {
            "dataset": dataset_name,
            "n_classes": n_classes,
            "gd": gd_metrics,
            "tp": tp_metrics,
            "duration": time.time() - start_time,
            "status": "completed"
        }

        logger.info(
            f"✅ {dataset_name}: GD={gd_metrics['acc']:.4f} vs TP={tp_metrics['acc']:.4f} "
            f"({result['duration']:.1f}s)",
            extra={"gpu_id": gpu_id}
        )

        return result

    except Exception as e:
        logger.error(f"❌ {dataset_name} failed: {e}", extra={"gpu_id": gpu_id})
        return {
            "dataset": dataset_name,
            "status": "failed",
            "error": str(e),
            "duration": time.time() - start_time
        }


def main():
    print("="*70)
    print("H200 PARALLEL VALIDATION - Extended Benchmark")
    print("="*70)
    print(f"GPUs: 1, 2, 3, 4, 5")
    print(f"Datasets: Wine, Breast Cancer, Digits, Segment")
    print(f"Expected runtime: 10-15 minutes")
    print("="*70)
    print()

    # Create output directory
    results_dir = ROOT / "eval" / "h200_validation_parallel"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Datasets and GPUs
    datasets = ["Wine", "Breast Cancer", "Digits", "Segment"]
    gpu_ids = [1, 2, 3, 4, 5]

    # Run in parallel
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {}
        for i, dataset in enumerate(datasets):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(run_benchmark_on_gpu, gpu_id, dataset)
            futures[future] = dataset

        for future in as_completed(futures):
            dataset = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed for {dataset}: {e}")
                results.append({"dataset": dataset, "status": "failed", "error": str(e)})

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "="*70)
    print("H200 PARALLEL VALIDATION RESULTS")
    print("="*70)

    for result in sorted(results, key=lambda r: r['dataset']):
        if result['status'] == 'completed':
            gd_acc = result['gd']['acc']
            tp_acc = result['tp']['acc']
            delta = gd_acc - tp_acc
            winner = "🟢 GraphDrone" if delta > 0.001 else ("🤝 Tie" if delta > -0.001 else "🔴 TabPFN")
            print(f"{result['dataset']:<15} | GD={gd_acc:.4f} | TP={tp_acc:.4f} | Δ={delta:+.4f} | {winner}")
        else:
            print(f"{result['dataset']:<15} | ❌ FAILED: {result.get('error', 'Unknown')}")

    print("="*70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Completed: {sum(1 for r in results if r['status'] == 'completed')}/{len(results)}")
    print("="*70)

    # Save results
    result_file = results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {result_file}")

    return 0 if all(r['status'] == 'completed' for r in results) else 1


if __name__ == "__main__":
    exit(main())
