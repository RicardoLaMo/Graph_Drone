import numpy as np
import torch
import sys
import os
import pandas as pd
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig

def load_segment():
    print("  -> Fetching Segment dataset (OpenML)...")
    df = fetch_openml(data_id=40984, as_frame=True, parser='auto')
    X = df.data.select_dtypes(include=[np.number]).values
    le = LabelEncoder()
    y = le.fit_transform(df.target)
    return X, y, len(np.unique(y))

def run_hyper_test(dataset_name, X, y, n_classes):
    print(f"\n--- HyperGraph Test: {dataset_name} ({n_classes} classes) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # GraphDrone (HyperSetRouter)
    print("  -> Training GraphDrone (HyperSetRouter)...")
    gd_config = GraphDroneConfig(
        problem_type="classification", 
        n_classes=n_classes,
        router=SetRouterConfig(kind="hyper_set_router")
    )
    gd_model = GraphDrone(gd_config)
    gd_model.fit(X_train, y_train)
    
    result = gd_model.predict(X_test, return_diagnostics=True)
    gd_proba = result.predictions
    gd_pred = np.argmax(gd_proba, axis=1)
    gd_acc = accuracy_score(y_test, gd_pred)
    
    print(f"  Accuracy: {gd_acc:.4f}")
    print(f"  Mean Defer Prob: {result.diagnostics.get('mean_defer_prob', 0):.4f}")
    
    return gd_acc

if __name__ == "__main__":
    # Test on Segment
    X_seg, y_seg, n_seg = load_segment()
    run_hyper_test("Segment", X_seg, y_seg, n_seg)
    
    # Test on Digits
    digits = load_digits()
    run_hyper_test("Digits", digits.data, digits.target, 10)
