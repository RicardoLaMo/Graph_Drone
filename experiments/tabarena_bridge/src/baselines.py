from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .datasets import Tier1Split


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


@dataclass(frozen=True)
class BaselineResult:
    model: str
    val_rmse: float
    test_rmse: float
    val_mae: float
    test_mae: float
    val_r2: float
    test_r2: float
    notes: str


def run_hgbr(split: Tier1Split, seed: int) -> BaselineResult:
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(split.X_train, split.y_train)
    pred_val = model.predict(split.X_val).astype(np.float32)
    pred_test = model.predict(split.X_test).astype(np.float32)
    val = _score(split.y_val, pred_val)
    test = _score(split.y_test, pred_test)
    return BaselineResult(
        model="HGBR",
        val_rmse=val["rmse"],
        test_rmse=test["rmse"],
        val_mae=val["mae"],
        test_mae=test["mae"],
        val_r2=val["r2"],
        test_r2=test["r2"],
        notes="Local Graph_Drone baseline",
    )


def run_tabpfn(split: Tier1Split, seed: int, n_estimators: int, device: str, max_train_samples: int | None = None) -> BaselineResult:
    from tabpfn import TabPFNRegressor

    X_train = split.X_train
    y_train = split.y_train
    if max_train_samples is not None and max_train_samples < len(X_train):
        rng = np.random.RandomState(seed)
        keep = np.sort(rng.choice(len(X_train), size=max_train_samples, replace=False))
        X_train = X_train[keep]
        y_train = y_train[keep]

    model = TabPFNRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        device=device,
        ignore_pretraining_limits=len(X_train) > 1000,
        n_preprocessing_jobs=1,
    )
    model.fit(X_train, y_train)
    pred_val = model.predict(split.X_val).astype(np.float32)
    pred_test = model.predict(split.X_test).astype(np.float32)
    val = _score(split.y_val, pred_val)
    test = _score(split.y_test, pred_test)
    return BaselineResult(
        model="TabPFN_full",
        val_rmse=val["rmse"],
        test_rmse=test["rmse"],
        val_mae=val["mae"],
        test_mae=test["mae"],
        val_r2=val["r2"],
        test_r2=test["r2"],
        notes=f"device={device}, n_estimators={n_estimators}",
    )
