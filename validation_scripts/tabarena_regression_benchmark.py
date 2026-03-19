"""
TabArena Curated Regression Benchmark
==========================================
Three-way comparison: TabPFN (1 est) vs TabPFN (8 est) vs GraphDrone (HyperSetRouter).

The 1-estimator TabPFN baseline matches the original published benchmark.
The 8-estimator baseline isolates the ensemble benefit from the routing benefit,
enabling a fair attribution of any GraphDrone gain.
"""

import time
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import openml

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig
from graphdrone_fit.expert_factory import ExpertBuildSpec, IdentitySelectorAdapter
from graphdrone_fit.view_descriptor import ViewDescriptor

# TabArena curated regression datasets (OpenML dataset IDs)
REGRESSION_DATASETS = {
    "california": 541,
    "diamonds": 42225,
    "house_prices": 42165,
    "elevators": 216,
}

# Number of estimators used by GraphDrone's anchor expert.
# Both TabPFN baselines are reported against this value for fair comparison.
_ANCHOR_N_ESTIMATORS = 8


def _preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """Unified categorical encoding + NaN imputation for all benchmark scripts."""
    X = X.copy()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(str)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])
    # Fill remaining NaNs with column median; fall back to 0 for all-NaN columns
    X = X.apply(lambda col: col.fillna(col.median()) if col.notna().any() else col.fillna(0))
    return X.values.astype(np.float32)


def load_openml_regression(dataset_name: str, max_samples: int = 1000):
    dataset_id = REGRESSION_DATASETS[dataset_name]
    print(f"  Fetching {dataset_name} (OpenML ID {dataset_id})...")
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if isinstance(X, pd.DataFrame):
        X = _preprocess_features(X)
    else:
        X = np.asarray(X, dtype=np.float32)

    if isinstance(y, pd.Series):
        y = y.fillna(y.mean()).values.astype(np.float32)
    else:
        y = np.asarray(y, dtype=np.float32)

    if len(X) > max_samples:
        _, X, _, y = train_test_split(X, y, test_size=max_samples, random_state=42)

    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_tabpfn_1est(X_tr, y_tr, X_te):
    """Original single-estimator TabPFN — matches published TabArena baseline."""
    from tabpfn import TabPFNRegressor
    model = TabPFNRegressor(device=_device(), n_estimators=1)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def run_tabpfn_8est(X_tr, y_tr, X_te):
    """8-estimator TabPFN ensemble — fair comparison anchor for GraphDrone."""
    from tabpfn import TabPFNRegressor
    model = TabPFNRegressor(
        device=_device(),
        n_estimators=_ANCHOR_N_ESTIMATORS,
        ignore_pretraining_limits=len(X_tr) > 1000,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def run_graphdrone(X_tr, y_tr, X_te, seed: int = 42):
    n_feats = X_tr.shape[1]
    full_idx = tuple(range(n_feats))

    rng = np.random.RandomState(seed)
    subspace_size = max(1, int(n_feats * 0.7))
    subspace_idx = tuple(sorted(rng.choice(n_feats, subspace_size, replace=False).tolist()))

    params_fp = {"n_estimators": _ANCHOR_N_ESTIMATORS, "device": _device()}
    params_cb = {"iterations": 200, "random_state": seed}
    params_xgb = {"n_estimators": 200, "random_state": seed}

    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL", family="FULL", view_name="Foundation Full",
                is_anchor=True, input_dim=n_feats, input_indices=full_idx,
            ),
            model_kind="foundation_regressor",
            input_adapter=IdentitySelectorAdapter(indices=full_idx),
            model_params=params_fp,
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="STRUCT_CB", family="structural_subspace", view_name="CatBoost Full",
                input_dim=n_feats, input_indices=full_idx,
            ),
            model_kind="catboost_regressor",
            input_adapter=IdentitySelectorAdapter(indices=full_idx),
            model_params=params_cb,
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="STRUCT_XGB", family="structural_subspace", view_name="XGBoost Full",
                input_dim=n_feats, input_indices=full_idx,
            ),
            model_kind="xgboost_regressor",
            input_adapter=IdentitySelectorAdapter(indices=full_idx),
            model_params=params_xgb,
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="GEOM_SUB", family="structural_subspace", view_name="Foundation Subspace",
                input_dim=subspace_size, input_indices=subspace_idx,
            ),
            model_kind="foundation_regressor",
            input_adapter=IdentitySelectorAdapter(indices=subspace_idx),
            model_params=params_fp,
        ),
    )

    cfg = GraphDroneConfig(
        router=SetRouterConfig(kind="hyper_set_router"),
        problem_type="regression",
    )
    gd = GraphDrone(cfg)
    gd.fit(X_tr, y_tr, expert_specs=specs)
    result = gd.predict(X_te, return_diagnostics=True)
    return result.predictions, result.diagnostics


def _bench_dataset(name: str, max_samples: int, seeds: list[int]) -> list[dict]:
    """Run all three models on one dataset × multiple seeds."""
    X, y = load_openml_regression(name, max_samples=max_samples)
    rows = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

        t0 = time.time()
        m_tp1 = compute_metrics(y_te, run_tabpfn_1est(X_tr, y_tr, X_te))
        t_tp1 = time.time() - t0

        t0 = time.time()
        m_tp8 = compute_metrics(y_te, run_tabpfn_8est(X_tr, y_tr, X_te))
        t_tp8 = time.time() - t0

        t0 = time.time()
        y_gd, diag = run_graphdrone(X_tr, y_tr, X_te, seed=seed)
        m_gd = compute_metrics(y_te, y_gd)
        t_gd = time.time() - t0

        defer = diag.get("mean_defer_prob", 0.0)
        print(
            f"  seed={seed} | "
            f"TabPFN(1)={m_tp1['r2']:.4f}  "
            f"TabPFN(8)={m_tp8['r2']:.4f}  "
            f"GraphDrone={m_gd['r2']:.4f}  "
            f"defer={defer:.4f}  "
            f"[{t_tp1:.0f}s / {t_tp8:.0f}s / {t_gd:.0f}s]"
        )
        rows.append({
            "dataset": name, "seed": seed, "n_samples": len(X),
            "tp1_r2": m_tp1["r2"], "tp1_rmse": m_tp1["rmse"],
            "tp8_r2": m_tp8["r2"], "tp8_rmse": m_tp8["rmse"],
            "gd_r2": m_gd["r2"], "gd_rmse": m_gd["rmse"],
            "defer": defer,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--datasets", nargs="+", default=list(REGRESSION_DATASETS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds for train/test split. Multiple seeds give variance estimates.")
    args = parser.parse_args()

    all_rows: list[dict] = []
    for name in args.datasets:
        if name not in REGRESSION_DATASETS:
            print(f"Skipping unknown dataset: {name}")
            continue
        print(f"\n{'='*70}\nDataset: {name}  (max_samples={args.max_samples})\n{'='*70}")
        try:
            all_rows.extend(_bench_dataset(name, args.max_samples, args.seeds))
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback; traceback.print_exc()

    if not all_rows:
        print("No results collected.")
        return

    df = pd.DataFrame(all_rows)
    # Aggregate over seeds
    agg = (
        df.groupby("dataset")
        .agg(
            n_samples=("n_samples", "first"),
            tp1_r2=("tp1_r2", "mean"), tp1_rmse=("tp1_rmse", "mean"),
            tp8_r2=("tp8_r2", "mean"), tp8_rmse=("tp8_rmse", "mean"),
            gd_r2=("gd_r2", "mean"),  gd_rmse=("gd_rmse", "mean"),
            defer=("defer", "mean"),
        )
        .reset_index()
    )
    agg["vs_tp1"] = agg["gd_r2"] - agg["tp1_r2"]   # total delta (ensemble + routing)
    agg["vs_tp8"] = agg["gd_r2"] - agg["tp8_r2"]   # routing-only delta (fair comparison)

    print("\n" + "="*70)
    print("FINAL REGRESSION SUMMARY  (averaged over seeds)")
    print("="*70)
    print(
        agg[["dataset", "n_samples",
             "tp1_r2", "tp8_r2", "gd_r2",
             "vs_tp1", "vs_tp8", "defer"]].to_string(index=False, float_format="{:.4f}".format)
    )
    print("\nLegend:")
    print("  tp1_r2  = TabPFN 1-estimator  (published baseline)")
    print("  tp8_r2  = TabPFN 8-estimator  (fair anchor — same config as GraphDrone FULL expert)")
    print("  gd_r2   = GraphDrone (HyperSetRouter + CatBoost + XGBoost specialists)")
    print("  vs_tp1  = GraphDrone R² gain vs published baseline  (ensemble + routing)")
    print("  vs_tp8  = GraphDrone R² gain vs fair anchor         (routing benefit only)")
    print("  defer   = mean defer probability (fraction of specialist blend used)")


if __name__ == "__main__":
    main()
