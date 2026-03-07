"""
data_california.py — California Housing loader for routing audit.
"""
import numpy as np
import json
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

SEED = 42


def load_california(artifact_dir: Path):
    cal = fetch_california_housing()
    X, y = cal.data.astype(np.float32), cal.target.astype(np.float32)
    for j, nm in enumerate(cal.feature_names):
        if nm in ("Population", "AveOccup", "AveRooms"):
            X[:, j] = np.log1p(X[:, j])
    sc = RobustScaler(); X = sc.fit_transform(X).astype(np.float32)
    all_i = np.arange(len(X))
    tr_i, tmp_i = train_test_split(all_i, test_size=0.30, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED)
    for nm, arr in [("X_all",X),("y_all",y),("idx_train",tr_i),("idx_val",va_i),("idx_test",te_i)]:
        np.save(artifact_dir/f"ca_{nm}.npy", arr)
    json.dump(list(cal.feature_names), open(artifact_dir/"ca_features.json","w"))
    print(f"[CA] n={len(X)} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")
    return X, y, tr_i, va_i, te_i
