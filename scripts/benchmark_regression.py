import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import NearestNeighbors

# Path setup
ROOT = Path("/home/wliu23/projects/Graph_Drone")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab_support_reader" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab_sigma2_enhancement" / "src"))

from latest_graphdrone import LatestGraphDroneIntegrated
from graphdrone_fit.model import GraphDrone, GraphDroneConfig
from graphdrone_fit.view_descriptor import ViewDescriptor
from graphdrone_fit.expert_factory import ExpertBuildSpec, IdentitySelectorAdapter
from graphdrone_fit.defer_integrator import integrate_predictions
from graphdrone_fit.support_encoder import SupportEncoding

import importlib.util
def load_fn_from_file(path, fn_name):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)

builder_path = ROOT / "experiments" / "lab_sigma2_enhancement" / "src" / "enhanced_token_builder.py"
compute_real_support_encoding = load_fn_from_file(str(builder_path), "compute_real_support_encoding")

# Full 9-Dataset Regression Portfolio
REGISTRY = {
    "diamonds": {"id": 46923, "target": "price", "geo": ["carat", "depth", "table", "x", "y", "z"]},
    "healthcare": {"id": 46931, "target": "charges", "geo": ["age", "bmi", "children", "smoker"]},
    "concrete": {"id": 46917, "target": "ConcreteCompressiveStrength", "geo": ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer"]},
    "airfoil": {"id": 46904, "target": "scaled-sound-pressure", "geo": ["frequency", "attack-angle", "chord-length"]},
    "wine_quality": {"id": 46964, "target": "median_wine_quality", "geo": ["alcohol", "density", "pH", "chlorides"]},
    "fiat_500": {"id": 46907, "target": "price", "geo": ["lat", "lon", "km", "age_in_days"]},
    "miami": {"id": 46942, "target": "SALE_PRC", "geo": ["LATITUDE", "LONGITUDE", "RAIL_DIST", "OCEAN_DIST"]},
    "houses": {"id": 46934, "target": "LnMedianHouseValue", "geo": ["Latitude", "Longitude"], "log_scale": True},
    "california": {"id": 44024, "target": "median_house_value", "geo": ["Latitude", "Longitude"], "log_scale": True}
}

def load_reg_data(name):
    spec = REGISTRY[name]
    print(f"Loading {name}...")
    data = fetch_openml(data_id=spec['id'], as_frame=True, parser="auto")
    X, y = data.data, data.target
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.median()).values.astype(np.float32)
    y = y.values.astype(np.float32)
    return X, y, spec['geo'], data.feature_names

def get_knn_bundle(X_train, X_query, y_train, k=15):
    knn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train)
    indices = knn.kneighbors(X_query, return_distance=False)
    return torch.tensor(X_train[indices]), torch.tensor(y_train[indices]).unsqueeze(-1).float()

def benchmark_dataset(name):
    X, y, geo_cols, feat_names = load_reg_data(name)
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Increase sample limit for High-Fidelity (10k)
    if len(X_tr) > 10000:
        idx = np.random.RandomState(42).choice(len(X_tr), 10000, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    scaler = StandardScaler(); X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va); X_te = scaler.transform(X_te)

    # Use 8 estimators for TabPFN (Project Standard)
    params = {"n_estimators": 8, "device": "cpu"}

    # 1. TabPFN Baseline
    print(f"  -> Running TabPFN (High-Fidelity)...")
    from tabpfn import TabPFNRegressor
    tpfn = TabPFNRegressor(**params)
    tpfn.fit(X_tr, y_tr)
    tpfn_rmse = root_mean_squared_error(y_te, tpfn.predict(X_te))

    # 2. Latest GraphDrone
    print(f"  -> Running Latest GraphDrone (Integrated Specialists)...")
    geo_idx = [feat_names.index(c) for c in geo_cols if c in feat_names]
    domain_idx = [i for i in range(X.shape[1]) if i not in geo_idx]
    
    specs = (
        ExpertBuildSpec(ViewDescriptor("FULL", "FULL", "Full", is_anchor=True, input_dim=X.shape[1], input_indices=tuple(range(X.shape[1]))), "foundation_regressor", IdentitySelectorAdapter(tuple(range(X.shape[1]))), model_params=params),
        ExpertBuildSpec(ViewDescriptor("GEO", "structural_subspace", "Geo", input_dim=len(geo_idx), input_indices=tuple(geo_idx)), "foundation_regressor", IdentitySelectorAdapter(tuple(geo_idx)), model_params=params),
        ExpertBuildSpec(ViewDescriptor("DOMAIN", "structural_subspace", "Domain", input_dim=len(domain_idx), input_indices=tuple(domain_idx)), "foundation_regressor", IdentitySelectorAdapter(tuple(domain_idx)), model_params=params)
    )
    
    gd_base = GraphDrone(GraphDroneConfig(full_expert_id="FULL")).fit(X_tr, y_tr, expert_specs=specs)
    factory = gd_base._expert_factory
    model = LatestGraphDroneIntegrated(d_x=X.shape[1], n_experts=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train Router
    va_preds = factory.predict_all(X_va)
    X_tr_views = [X_tr[:, list(s.descriptor.input_indices)] for s in specs]
    va_moments = compute_real_support_encoding(X_tr_views, [X_va[:, list(s.descriptor.input_indices)] for s in specs], y_tr)
    va_supp_x, va_supp_y = get_knn_bundle(X_tr, X_va, y_tr)
    y_va_t = torch.tensor(y_va).float(); v_preds_t = torch.tensor(va_preds.predictions).float()
    
    print(f"  -> Optimizing Router (SNR + Neural Support)...")
    for _ in range(100):
        model.train()
        out = model(torch.tensor(X_va), [va_supp_x]*3, [va_supp_y]*3, va_preds.predictions, va_preds.descriptors, va_preds.full_index, va_moments)
        integ = (1 - out.defer_prob) * v_preds_t[:, va_preds.full_index:va_preds.full_index+1] + \
                out.defer_prob * (out.specialist_weights * v_preds_t).sum(dim=1, keepdim=True)
        loss = nn.functional.mse_loss(integ.squeeze(), y_va_t)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
    # Eval
    model.eval()
    with torch.no_grad():
        te_preds = factory.predict_all(X_te)
        te_moments = compute_real_support_encoding(X_tr_views, [X_te[:, list(s.descriptor.input_indices)] for s in specs], y_tr)
        te_supp_x, te_supp_y = get_knn_bundle(X_tr, X_te, y_tr)
        out = model(torch.tensor(X_te), [te_supp_x]*3, [te_supp_y]*3, te_preds.predictions, te_preds.descriptors, te_preds.full_index, te_moments)
        y_prob = integrate_predictions(expert_predictions=te_preds.predictions, router_outputs=out).predictions
        
    gd_rmse = root_mean_squared_error(y_te, y_prob)
    return {"dataset": name, "tpfn": tpfn_rmse, "gd": gd_rmse}

if __name__ == "__main__":
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    results = []
    for name in REGISTRY.keys():
        try:
            results.append(benchmark_dataset(name))
        except Exception as e:
            print(f"Error on {name}: {e}")
            
    df = pd.DataFrame(results)
    df["delta"] = ((df["gd"] - df["tpfn"]) / df["tpfn"]) * 100
    print("\n--- FINAL HIGH-FIDELITY REGRESSION BENCHMARK ---")
    print(df.to_string(index=False))
    print(f"\nAverage RMSE Delta: {df['delta'].mean():.2f}%")
