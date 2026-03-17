"""
TabArena Curated Regression Benchmark
==========================================
Compares GraphDrone (HyperSetRouter) vs TabPFN
on regression datasets from the TabArena suite.
"""

import time
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import openml

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig
from graphdrone_fit.expert_factory import ExpertBuildSpec, IdentitySelectorAdapter, PcaProjectionAdapter
from graphdrone_fit.view_descriptor import ViewDescriptor

# TabArena curated regression datasets (OpenML task IDs or dataset IDs)
REGRESSION_DATASETS = {
    "california": 541,
    "diamonds": 42225,
    "house_prices": 42165,
    "elevators": 216
}

def load_openml_regression(dataset_name, max_samples=1000):
    dataset_id = REGRESSION_DATASETS[dataset_name]
    print(f"Fetching {dataset_name} (ID {dataset_id})...")
    
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Preprocess
    if isinstance(X, pd.DataFrame):
        from sklearn.preprocessing import OrdinalEncoder
        cat_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            X[cat_cols] = X[cat_cols].astype(str)
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X[cat_cols] = enc.fit_transform(X[cat_cols])
        X = X.fillna(0).values
        
    if isinstance(y, pd.Series):
        y = y.fillna(y.mean()).values
        
    if len(X) > max_samples:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=max_samples, random_state=42)
        
    return X, y

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def run_tabpfn(X_tr, y_tr, X_te):
    from tabpfn import TabPFNRegressor
    model = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=1)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)

def run_graphdrone(X_tr, y_tr, X_te):
    n_feats = X_tr.shape[1]
    full_idx = tuple(range(n_feats))
    
    rng = np.random.RandomState(42)
    subspace_size = max(1, int(n_feats * 0.7))
    subspace_idx = tuple(sorted(rng.choice(n_feats, subspace_size, replace=False)))
    
    params_fp = {"n_estimators": 8, "device": "cuda"}
    params_cb = {"iterations": 200, "random_state": 42}
    params_xgb = {"n_estimators": 200, "random_state": 42}
    
    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="FULL", family="FULL", view_name="Foundation Full", is_anchor=True, input_dim=n_feats, input_indices=full_idx),
            model_kind="foundation_regressor", input_adapter=IdentitySelectorAdapter(indices=full_idx), model_params=params_fp
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="STRUCT_CB", family="structural_subspace", view_name="CatBoost Full", input_dim=n_feats, input_indices=full_idx),
            model_kind="catboost_regressor", input_adapter=IdentitySelectorAdapter(indices=full_idx), model_params=params_cb
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="STRUCT_XGB", family="structural_subspace", view_name="XGBoost Full", input_dim=n_feats, input_indices=full_idx),
            model_kind="xgboost_regressor", input_adapter=IdentitySelectorAdapter(indices=full_idx), model_params=params_xgb
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="GEOM_SUB", family="structural_subspace", view_name="Foundation Subspace", input_dim=subspace_size, input_indices=subspace_idx),
            model_kind="foundation_regressor", input_adapter=IdentitySelectorAdapter(indices=subspace_idx), model_params=params_fp
        )
    )
    
    cfg = GraphDroneConfig(
        router=SetRouterConfig(kind="hyper_set_router"),
        problem_type="regression"
    )
    gd = GraphDrone(cfg)
    gd.fit(X_tr, y_tr, expert_specs=specs)
    
    result = gd.predict(X_te, return_diagnostics=True)
    return result.predictions, result.diagnostics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--datasets", nargs="+", default=list(REGRESSION_DATASETS.keys()))
    args = parser.parse_args()
    results = []
    
    for name in args.datasets:
        if name not in REGRESSION_DATASETS:
            print(f"Skipping unknown dataset: {name}")
            continue
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
        try:
            X, y = load_openml_regression(name, max_samples=args.max_samples)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            
            t0 = time.time()
            y_tp = run_tabpfn(X_tr, y_tr, X_te)
            m_tp = compute_metrics(y_te, y_tp)
            t_tp = time.time() - t0
            print(f"  TabPFN:   RMSE={m_tp['rmse']:.4f} R2={m_tp['r2']:.4f} ({t_tp:.1f}s)")
            
            t0 = time.time()
            y_gd, diag = run_graphdrone(X_tr, y_tr, X_te)
            m_gd = compute_metrics(y_te, y_gd)
            t_gd = time.time() - t0
            defer = diag.get("mean_defer_prob", 0.0)
            print(f"  GraphDrone: RMSE={m_gd['rmse']:.4f} R2={m_gd['r2']:.4f} defer={defer:.4f} ({t_gd:.1f}s)")
            
            results.append({
                "dataset": name, "tp_r2": m_tp["r2"], "gd_r2": m_gd["r2"], "defer": defer
            })
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL REGRESSION SUMMARY")
    print("="*60)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
