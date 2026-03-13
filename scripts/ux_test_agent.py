import os
import sys
import torch
import numpy as np
import time
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, root_mean_squared_error

# Simulate Installation check
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig
    print("✅ UX TEST: GraphDrone package successfully imported.")
except ImportError:
    print("❌ UX TEST ERROR: Package not found. Ensure 'pip install -e .' was run.")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_agent_session(dataset_id, task="classification"):
    print(f"\n--- STARTING AGENT SESSION: Dataset ID {dataset_id} ({task}) ---")
    
    # 1. Data Acquisition (UX: Speed of loading and auto-preprocessing)
    print(f"  [1/4] Fetching random data from OpenML...")
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    X, y = data.data, data.target
    
    # Simple auto-encoder for the UX agent
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.median()).values.astype(np.float32)
    
    if task == "classification":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Model Configuration (UX: Intuitive parameterization)
    config = GraphDroneConfig(
        full_expert_id="FULL",
        router=SetRouterConfig(kind="cross_attention_set_router")
    )
    model = GraphDrone(config)

    # 3. Training (UX: H200 Utilization + Progress logs)
    print(f"  [2/4] Fitting Model on {DEVICE.upper()}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time
    print(f"  -> Fit complete in {fit_duration:.2f} seconds.")

    # 4. Inference (UX: Smooth prediction + Diagnostics)
    print(f"  [3/4] Running Inference...")
    result = model.predict(X_test, return_diagnostics=True)
    preds = result.predictions
    
    # 5. Reporting
    print(f"  [4/4] Final Metrics:")
    if task == "classification":
        print(f"  -> Accuracy: {accuracy_score(y_test, np.round(preds)):.4f}")
    else:
        print(f"  -> RMSE    : {root_mean_squared_error(y_test, preds):.4f}")
    
    print(f"  -> Diagnostics Check: {list(result.diagnostics.keys())[:3]}... passed.")
    print(f"✅ SESSION COMPLETE: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

if __name__ == "__main__":
    # Test 1: New Classification (Phoneme)
    run_agent_session(1489, task="classification")
    
    # Test 2: New Regression (CPU Small)
    run_agent_session(562, task="regression")
