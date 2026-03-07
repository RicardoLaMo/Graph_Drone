"""
00_data.py — Load, preprocess, split, and save California Housing.
No leakage: scaler fit on train only.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SEED = 42
LOG_COLS = ["Population", "AveOccup", "AveRooms"]


def load_and_preprocess():
    print("[00_data] Loading California Housing from sklearn...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    target_col = "MedHouseVal"

    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].copy()
    y = df[target_col].values

    # log1p transforms for skewed columns
    for col in LOG_COLS:
        if col in X.columns:
            X[col] = np.log1p(X[col].values)

    X_arr = X.values.astype(np.float32)
    y_arr = y.astype(np.float32)

    # 70 / 15 / 15 split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_arr, y_arr, test_size=0.30, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED
    )

    # Fit scaler on train ONLY
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_sc = scaler.transform(X_val).astype(np.float32)
    X_te_sc = scaler.transform(X_te).astype(np.float32)

    # Save
    np.save(ARTIFACTS / "X_train.npy", X_tr_sc)
    np.save(ARTIFACTS / "X_val.npy", X_val_sc)
    np.save(ARTIFACTS / "X_test.npy", X_te_sc)
    np.save(ARTIFACTS / "y_train.npy", y_tr)
    np.save(ARTIFACTS / "y_val.npy", y_val)
    np.save(ARTIFACTS / "y_test.npy", y_te)

    # Also save full scaled array (needed for graph building over all rows)
    X_all_sc = scaler.transform(X_arr).astype(np.float32)
    np.save(ARTIFACTS / "X_all.npy", X_all_sc)
    np.save(ARTIFACTS / "y_all.npy", y_arr)

    # Save train/val/test indices for traceability
    all_idx = np.arange(len(X_arr))
    tr_idx, tmp_idx = train_test_split(all_idx, test_size=0.30, random_state=SEED)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=SEED)
    np.save(ARTIFACTS / "idx_train.npy", tr_idx)
    np.save(ARTIFACTS / "idx_val.npy", val_idx)
    np.save(ARTIFACTS / "idx_test.npy", te_idx)

    with open(ARTIFACTS / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print(f"[00_data]  Train: {X_tr_sc.shape}, Val: {X_val_sc.shape}, Test: {X_te_sc.shape}")
    assert X_tr_sc.shape[1] == len(feature_names), "Feature mismatch!"
    print("[00_data] Done. Artifacts saved.")
    return feature_names, scaler


if __name__ == "__main__":
    load_and_preprocess()
