import numpy as np
import torch
import sys
import os
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from tabpfn import TabPFNClassifier

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig

def compute_metrics(y_true, y_pred, y_proba, n_classes):
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
    from sklearn.datasets import fetch_openml
    print("  -> Fetching Segment dataset (OpenML)...")
    df = fetch_openml(data_id=40984, as_frame=True, parser='auto')
    X = df.data.select_dtypes(include=[np.number]).values
    le = LabelEncoder()
    y = le.fit_transform(df.target)
    return type('Data', (), {'data': X, 'target': y})

def run_benchmark(dataset_name, data_loader, n_classes, max_samples=1000):
    print(f"\n--- Benchmarking: {dataset_name} ({n_classes} classes) ---")
    data = data_loader()
    X, y = data.data, data.target
    
    # Increase sample size for better Router signal
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[indices], y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  -> Samples: Train={len(X_train)}, Test={len(X_test)}")

    # 1. GraphDrone (PC-MoE)
    print("  -> Training GraphDrone (H200 Optimized)...")
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
    
    # 2. TabPFN
    print("  -> Training TabPFN...")
    tabpfn = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=1)
    tabpfn.fit(X_train, y_train)
    tp_proba = tabpfn.predict_proba(X_test)
    tp_pred = np.argmax(tp_proba, axis=1)
    tp_metrics = compute_metrics(y_test, tp_pred, tp_proba, n_classes)
    
    return {"dataset": dataset_name, "gd": gd_metrics, "tp": tp_metrics}

if __name__ == "__main__":
    results = []
    # Original sets (with more samples)
    results.append(run_benchmark("Wine", load_wine, 3, max_samples=500))
    results.append(run_benchmark("Breast Cancer", load_breast_cancer, 2, max_samples=1000))
    # New Multi-class sets
    results.append(run_benchmark("Digits", load_digits, 10, max_samples=1500))
    results.append(run_benchmark("Segment", load_segment, 7, max_samples=2000))
    
    print("\n" + "="*85)
    print("EXTENDED BENCHMARK SUMMARY (H200 Optimized)")
    print("="*85)
    header = f"{'Dataset':<15} | {'Model':<12} | {'Acc':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8}"
    print(header)
    print("-" * 85)
    for r in results:
        for model_key, label in [("gd", "GraphDrone"), ("tp", "TabPFN")]:
            m = r[model_key]
            print(f"{r['dataset']:<15} | {label:<12} | {m['acc']:<8.4f} | {m['f1']:<8.4f} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<8.4f}")
        print("-" * 85)
