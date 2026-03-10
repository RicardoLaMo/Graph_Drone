from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer


SEED = 42
LOG1P_COLS = (2, 4)


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_aligned_california_split(seed: int = SEED) -> SplitData:
    X, y = fetch_california_housing(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X[:, LOG1P_COLS] = np.log1p(X[:, LOG1P_COLS])

    all_idx = np.arange(len(X))
    train_idx, tmp_idx = train_test_split(all_idx, test_size=0.30, random_state=seed)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, random_state=seed)

    return SplitData(
        X_train=X[train_idx],
        X_val=X[val_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )


def write_aligned_california_dataset(output_dir: Path, seed: int = SEED) -> tuple[Path, SplitData]:
    split = build_aligned_california_split(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_num_train.npy", split.X_train)
    np.save(output_dir / "X_num_val.npy", split.X_val)
    np.save(output_dir / "X_num_test.npy", split.X_test)
    np.save(output_dir / "Y_train.npy", split.y_train)
    np.save(output_dir / "Y_val.npy", split.y_val)
    np.save(output_dir / "Y_test.npy", split.y_test)
    (output_dir / "info.json").write_text(
        json.dumps(
            {
                "task_type": "regression",
                "score": "rmse",
                "split_policy": "repo_california_70_15_15_seed42",
                "feature_edit": "log1p_cols_2_4_only",
            },
            indent=2,
        )
        + "\n"
    )
    return output_dir, split


def tabm_noisy_quantile_transform(
    split: SplitData,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = split.X_train.astype(np.float32, copy=True)
    train = train + np.random.RandomState(seed).normal(0.0, 1e-5, train.shape).astype(
        np.float32
    )
    normalizer = QuantileTransformer(
        n_quantiles=max(min(train.shape[0] // 30, 1000), 10),
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=seed,
    )
    normalizer.fit(train)
    X_train = normalizer.transform(split.X_train).astype(np.float32)
    X_val = normalizer.transform(split.X_val).astype(np.float32)
    X_test = normalizer.transform(split.X_test).astype(np.float32)
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)
    mask = np.array([len(np.unique(col)) > 1 for col in X_train.T], dtype=bool)
    return X_train[:, mask], X_val[:, mask], X_test[:, mask]


def standardize_regression_targets(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    mean = float(y_train.mean())
    std = float(y_train.std() + 1e-8)
    return (
        ((y_train - mean) / std).astype(np.float32),
        ((y_val - mean) / std).astype(np.float32),
        ((y_test - mean) / std).astype(np.float32),
        {"mean": mean, "std": std},
    )


def denormalize(y_norm: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return (y_norm * float(stats["std"]) + float(stats["mean"])).astype(np.float32)
