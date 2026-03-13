import os
import sys
import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig

# The user's specific 4 datasets
VERIFY_PORTFOLIO = [
    {"id": 46904, "name": "airfoil_self_noise", "task": "regression"},
    {"id": 46923, "name": "diamonds", "task": "regression"},
    {"id": 37, "name": "diabetes", "task": "classification"},
    {"id": 1497, "name": "bioresponse", "task": "classification"}, # ID 1497
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_verify(spec):
    name = spec["name"]
    task = spec["task"]
    print(f"\n[VERIFY] Processing {name} ({task})...")
    
    data = fetch_openml(data_id=spec["id"], as_frame=True, parser="auto")
    X, y = data.data, data.target
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.median()).values.astype(np.float32)
    
    if task == "classification":
        y = LabelEncoder().fit_transform(y.astype(str))
        n_classes = len(np.unique(y))
    else:
        y = y.values.astype(np.float32)
        n_classes = 1

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=20260312) # User seed
    
    if len(X_tr) > 2048:
        X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=2048, random_state=20260312)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # GraphDrone (Consolidated + Patch)
    print(f"  -> Running Optimized GraphDrone...")
    config = GraphDroneConfig(
        full_expert_id="FULL",
        router=SetRouterConfig(kind="noise_gate_router")
    )
    model = GraphDrone(config)
    
    start_time = time.time()
    model.fit(X_tr, y_tr)
    duration = time.time() - start_time
    
    preds = model.predict(X_te)
    
    if task == "classification":
        auc = roc_auc_score(y_te, preds[:, 1] if preds.ndim > 1 else preds, multi_class='ovr')
        acc = accuracy_score(y_te, np.round(preds[:, 1] if preds.ndim > 1 else preds))
        return {"dataset": name, "metric": auc, "acc": acc, "time": duration}
    else:
        rmse = root_mean_squared_error(y_te, preds)
        return {"dataset": name, "metric": rmse, "time": duration}

if __name__ == "__main__":
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    results = []
    for spec in VERIFY_PORTFOLIO:
        try:
            results.append(run_verify(spec))
        except Exception as e:
            print(f"Error on {spec['name']}: {e}")
            
    df = pd.DataFrame(results)
    print("\n--- VERIFICATION RESULTS (POST-PATCH) ---")
    print(df.to_string(index=False))
