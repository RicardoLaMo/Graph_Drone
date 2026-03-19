import numpy as np
import torch
import sys
import os
import pandas as pd
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tabpfn import TabPFNClassifier

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig

def analyze_failure(dataset_name, X, y, n_classes):
    print(f"\n--- Failure Analysis: {dataset_name} ({n_classes} classes, {X.shape[1]} features) ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. GraphDrone (PC-MoE)
    gd_config = GraphDroneConfig(
        problem_type="classification", 
        n_classes=n_classes,
        router=SetRouterConfig(kind="contextual_transformer_router")
    )
    gd_model = GraphDrone(gd_config)
    gd_model.fit(X_train, y_train)
    
    # Get diagnostics
    result = gd_model.predict(X_test, return_diagnostics=True)
    gd_proba = result.predictions
    gd_pred = np.argmax(gd_proba, axis=1)
    gd_acc = accuracy_score(y_test, gd_pred)
    
    # 2. TabPFN
    tabpfn = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=1)
    tabpfn.fit(X_train, y_train)
    tp_proba = tabpfn.predict_proba(X_test)
    tp_pred = np.argmax(tp_proba, axis=1)
    tp_acc = accuracy_score(y_test, tp_pred)
    
    print(f"  Accuracy -> GraphDrone: {gd_acc:.4f} | TabPFN: {tp_acc:.4f}")
    
    # Deeper Dive: Where did we disagree?
    mismatches = (gd_pred != tp_pred)
    print(f"  Disagreement Rate: {np.mean(mismatches):.4f}")
    
    # Diagnostic: Mean Defer Probability
    mean_defer = result.diagnostics.get("mean_defer_prob", 0)
    print(f"  Mean Defer Prob: {mean_defer:.4f}")
    
    return {
        "dataset": dataset_name,
        "gd_acc": gd_acc,
        "tp_acc": tp_acc,
        "mean_defer": mean_defer,
        "disagreement": np.mean(mismatches)
    }

def load_gas():
    print("  -> Fetching Gas Sensor Drift dataset (OpenML)...")
    # Using a subset for speed in analysis
    df = fetch_openml(data_id=1476, as_frame=True, parser='auto')
    X = df.data.select_dtypes(include=[np.number]).values
    y = pd.factorize(df.target)[0]
    return X, y, len(np.unique(y))

if __name__ == "__main__":
    # Test Digits (where we had a slight gap)
    digits = load_digits()
    analyze_failure("Digits", digits.data, digits.target, 10)
    
    # Test Gas Sensor (High-dim, Multi-class)
    X_gas, y_gas, n_gas = load_gas()
    # Sample down for analysis speed
    idx = np.random.choice(len(X_gas), 1000, replace=False)
    analyze_failure("Gas Sensor", X_gas[idx], y_gas[idx], n_gas)
