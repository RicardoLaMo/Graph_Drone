from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def write_california_dataset(output_dir: Path, seed: int = 0) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = fetch_california_housing(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    idx = np.arange(len(X))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

    np.save(output_dir / "X_num_train.npy", X[train_idx])
    np.save(output_dir / "X_num_val.npy", X[val_idx])
    np.save(output_dir / "X_num_test.npy", X[test_idx])
    np.save(output_dir / "Y_train.npy", y[train_idx])
    np.save(output_dir / "Y_val.npy", y[val_idx])
    np.save(output_dir / "Y_test.npy", y[test_idx])
    (output_dir / "info.json").write_text(json.dumps({"task_type": "regression"}, indent=2) + "\n")
    return output_dir

