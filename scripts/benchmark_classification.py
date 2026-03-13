import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error

# Add src to path for the consolidated package
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig

# 10 Brand New Datasets (Not in the previous 18 or UX test)
TEST_PORTFOLIO = [
    # Classification
    {"id": 23, "name": "cmc", "task": "classification"},
    {"id": 1494, "name": "qsar-biodeg", "task": "classification"},
    {"id": 44, "name": "spambase", "task": "classification"},
    {"id": 1487, "name": "ozone-8hr", "task": "classification"},
    {"id": 1467, "name": "climate-crash", "task": "classification"},
    # Regression
    {"id": 183, "name": "abalone", "task": "regression"},
    {"id": 127, "name": "machine_cpu", "task": "regression"},
    {"id": 198, "name": "delta_ailerons", "task": "regression"},
    {"id": 204, "name": "quake", "task": "regression"},
    {"id": 208, "name": "stock", "task": "regression"},
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_dataset_benchmark(spec):
    name = spec["name"]
    task = spec["task"]
    print(f"\n[SWEEP] Processing {name} ({task})...")
    
    # 1. Load and Preprocess
    data = fetch_openml(data_id=spec["id"], as_frame=True, parser="auto")
    X, y = data.data, data.target
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.median()).values.astype(np.float32)
    
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        n_classes = len(le.classes_)
    else:
        y = y.values.astype(np.float32)
        n_classes = 1

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    # Cap training for TabPFN context window (and speed)
    if len(X_tr) > 2048:
        idx = np.random.RandomState(42).choice(len(X_tr), 2048, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # 2. TabPFN Baseline
    print(f"  -> Baseline: TabPFN...")
    if task == "classification":
        from tabpfn import TabPFNClassifier
        tpfn = TabPFNClassifier(device=DEVICE, n_estimators=8)
        tpfn.fit(X_tr, y_tr)
        tpfn_prob = tpfn.predict_proba(X_te)
        tpfn_metric = roc_auc_score(y_te, tpfn_prob[:, 1] if n_classes==2 else tpfn_prob, multi_class='ovr')
    else:
        from tabpfn import TabPFNRegressor
        tpfn = TabPFNRegressor(device=DEVICE, n_estimators=8)
        tpfn.fit(X_tr, y_tr)
        tpfn_preds = tpfn.predict(X_te)
        tpfn_metric = root_mean_squared_error(y_te, tpfn_preds)

    # 3. GraphDrone (Consolidated)
    print(f"  -> Challenger: GraphDrone...")
    config = GraphDroneConfig(
        full_expert_id="FULL",
        router=SetRouterConfig(kind="noise_gate_router")
    )
    # In the consolidated engine, fit() handles the router training epoch implicitly
    model = GraphDrone(config)
    model.fit(X_tr, y_tr)
    
    gd_preds = model.predict(X_te)
    
    if task == "classification":
        gd_metric = roc_auc_score(y_te, gd_preds) # gd_preds are probas in the new engine
    else:
        gd_metric = root_mean_squared_error(y_te, gd_preds)

    return {
        "dataset": name,
        "task": task,
        "tpfn": tpfn_metric,
        "gd": gd_metric,
        "win": (gd_metric > tpfn_metric if task=="classification" else gd_metric < tpfn_metric)
    }

if __name__ == "__main__":
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    results = []
    for spec in TEST_PORTFOLIO:
        try:
            results.append(run_dataset_benchmark(spec))
        except Exception as e:
            print(f"Error on {spec['name']}: {e}")
            
    df = pd.DataFrame(results)
    print("\n--- FINAL DISTRIBUTION-READY SWEEP: TabPFN vs CONSOLIDATED GRAPHDRONE ---")
    print(df.to_string(index=False))
    
    wins = df['win'].sum()
    print(f"\nGraphDrone Wins: {wins}/10")
    print(f"Verified on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
