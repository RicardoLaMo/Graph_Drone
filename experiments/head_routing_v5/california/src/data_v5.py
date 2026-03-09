from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


SEED = 42


@dataclass(frozen=True)
class TargetStats:
    mean: float
    std: float


def normalise_target(y: np.ndarray, stats: dict[str, float] | TargetStats) -> np.ndarray:
    mean = float(stats["mean"] if isinstance(stats, dict) else stats.mean)
    std = float(stats["std"] if isinstance(stats, dict) else stats.std)
    return ((y - mean) / std).astype(np.float32)


def denormalise_target(y_norm: np.ndarray, stats: dict[str, float] | TargetStats) -> np.ndarray:
    mean = float(stats["mean"] if isinstance(stats, dict) else stats.mean)
    std = float(stats["std"] if isinstance(stats, dict) else stats.std)
    return (y_norm * std + mean).astype(np.float32)


def build_california_dataset(seed: int = SEED) -> dict:
    dataset = fetch_california_housing()
    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.float32)

    X[:, [2, 4]] = np.log1p(X[:, [2, 4]])

    all_idx = np.arange(len(X))
    train_idx, tmp_idx = train_test_split(all_idx, test_size=0.30, random_state=seed)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, random_state=seed)

    scaler = RobustScaler()
    X = scaler.fit(X[train_idx]).transform(X).astype(np.float32)

    y_train = y[train_idx]
    target_stats = {
        "mean": float(y_train.mean()),
        "std": float(y_train.std() + 1e-8),
    }

    return {
        "X": X,
        "y": y,
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "target_stats": target_stats,
        "feature_names": tuple(dataset.feature_names),
    }
