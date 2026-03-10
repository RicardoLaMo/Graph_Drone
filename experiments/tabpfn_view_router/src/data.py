from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


SEED = 42
OPENML_CALIFORNIA_DID = 44024
LOG1P_COLS = (2, 4)
GEO_COLS = (6, 7)
SOCIO_COLS = (0, 1, 2, 3, 4)
EPS = 1e-8


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


@dataclass(frozen=True)
class ViewData:
    train: Dict[str, np.ndarray]
    val: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    view_names: list[str]


@dataclass(frozen=True)
class QualityFeatures:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    sigma2_train: np.ndarray
    sigma2_val: np.ndarray
    sigma2_test: np.ndarray
    mean_j_train: np.ndarray
    mean_j_val: np.ndarray
    mean_j_test: np.ndarray


def build_aligned_california_split(
    seed: int = SEED,
    smoke: bool = False,
    *,
    dataset_source: str = "sklearn",
    openml_dataset_id: int = OPENML_CALIFORNIA_DID,
) -> SplitData:
    X, y = _fetch_california_xy(
        dataset_source=dataset_source,
        openml_dataset_id=openml_dataset_id,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X[:, LOG1P_COLS] = np.log1p(X[:, LOG1P_COLS])

    all_idx = np.arange(len(X))
    train_idx, tmp_idx = train_test_split(all_idx, test_size=0.30, random_state=seed)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, random_state=seed)

    if smoke:
        train_idx = train_idx[:1200]
        val_idx = val_idx[:300]
        test_idx = test_idx[:300]

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


def build_view_data(split: SplitData) -> ViewData:
    pca = PCA(n_components=4, random_state=SEED)
    pca.fit(split.X_train)
    train_lr = pca.transform(split.X_train).astype(np.float32)
    val_lr = pca.transform(split.X_val).astype(np.float32)
    test_lr = pca.transform(split.X_test).astype(np.float32)

    view_names = ["FULL", "GEO", "SOCIO", "LOWRANK"]
    return ViewData(
        train={
            "FULL": split.X_train,
            "GEO": split.X_train[:, GEO_COLS],
            "SOCIO": split.X_train[:, SOCIO_COLS],
            "LOWRANK": train_lr,
        },
        val={
            "FULL": split.X_val,
            "GEO": split.X_val[:, GEO_COLS],
            "SOCIO": split.X_val[:, SOCIO_COLS],
            "LOWRANK": val_lr,
        },
        test={
            "FULL": split.X_test,
            "GEO": split.X_test[:, GEO_COLS],
            "SOCIO": split.X_test[:, SOCIO_COLS],
            "LOWRANK": test_lr,
        },
        view_names=view_names,
    )


def _train_knn_and_sigma2(
    X_train: np.ndarray,
    X_query: np.ndarray,
    y_train: np.ndarray,
    k: int,
    drop_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_neighbors = k + 1 if drop_self else k
    knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train)
    idx = knn.kneighbors(X_query, return_distance=False)
    if drop_self:
        idx = idx[:, 1:]
    sigma2 = y_train[idx].var(axis=1).astype(np.float32)
    return idx.astype(np.int64), sigma2


def _jaccard_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.zeros(a.shape[0], dtype=np.float32)
    a_sorted = np.sort(a, axis=1)
    b_sorted = np.sort(b, axis=1)
    k = a.shape[1]
    for i in range(a.shape[0]):
        pos = np.searchsorted(b_sorted[i], a_sorted[i])
        pos = np.clip(pos, 0, k - 1)
        inter[i] = float(np.sum(b_sorted[i][pos] == a_sorted[i]))
    union = 2 * k - inter
    return inter / np.maximum(union, 1.0)


def build_quality_features(split: SplitData, views: ViewData, k: int = 24) -> QualityFeatures:
    train_sigma_cols = []
    val_sigma_cols = []
    test_sigma_cols = []
    train_knn = []
    val_knn = []
    test_knn = []
    for name in views.view_names:
        idx_train, sigma_train = _train_knn_and_sigma2(
            views.train[name], views.train[name], split.y_train, k=k, drop_self=True
        )
        idx_val, sigma_val = _train_knn_and_sigma2(
            views.train[name], views.val[name], split.y_train, k=k, drop_self=False
        )
        idx_test, sigma_test = _train_knn_and_sigma2(
            views.train[name], views.test[name], split.y_train, k=k, drop_self=False
        )
        train_knn.append(idx_train)
        val_knn.append(idx_val)
        test_knn.append(idx_test)
        train_sigma_cols.append(sigma_train)
        val_sigma_cols.append(sigma_val)
        test_sigma_cols.append(sigma_test)

    sigma2_train_raw = np.stack(train_sigma_cols, axis=1)
    sigma2_val_raw = np.stack(val_sigma_cols, axis=1)
    sigma2_test_raw = np.stack(test_sigma_cols, axis=1)

    log_train = np.log1p(np.maximum(sigma2_train_raw, 0.0))
    median = np.median(log_train, axis=0)
    q25 = np.percentile(log_train, 25, axis=0)
    q75 = np.percentile(log_train, 75, axis=0)
    iqr = np.maximum(q75 - q25, EPS)

    def norm_sigma(x: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.maximum(x, 0.0)) - median) / iqr).astype(np.float32)

    sigma2_train = norm_sigma(sigma2_train_raw)
    sigma2_val = norm_sigma(sigma2_val_raw)
    sigma2_test = norm_sigma(sigma2_test_raw)

    pairs = [(i, j) for i in range(len(views.view_names)) for j in range(i + 1, len(views.view_names))]

    def build_j_features(knn_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        cols = [_jaccard_rows(knn_list[i], knn_list[j]) for i, j in pairs]
        j_flat = np.stack(cols, axis=1).astype(np.float32)
        return j_flat, j_flat.mean(axis=1, keepdims=True).astype(np.float32)

    j_train, mean_train = build_j_features(train_knn)
    j_val, mean_val = build_j_features(val_knn)
    j_test, mean_test = build_j_features(test_knn)

    feat_train = np.concatenate([sigma2_train, j_train, mean_train], axis=1).astype(np.float32)
    feat_val = np.concatenate([sigma2_val, j_val, mean_val], axis=1).astype(np.float32)
    feat_test = np.concatenate([sigma2_test, j_test, mean_test], axis=1).astype(np.float32)

    return QualityFeatures(
        train=feat_train,
        val=feat_val,
        test=feat_test,
        sigma2_train=sigma2_train,
        sigma2_val=sigma2_val,
        sigma2_test=sigma2_test,
        mean_j_train=mean_train.squeeze(1),
        mean_j_val=mean_val.squeeze(1),
        mean_j_test=mean_test.squeeze(1),
    )


def _fetch_california_xy(*, dataset_source: str, openml_dataset_id: int) -> tuple[np.ndarray, np.ndarray]:
    if dataset_source == "openml":
        return _fetch_california_xy_openml(openml_dataset_id)
    if dataset_source == "sklearn":
        try:
            return fetch_california_housing(return_X_y=True)
        except FileNotFoundError:
            # sklearn can successfully materialize the cached dataset and then fail
            # while cleaning up the temporary archive on first download. A second
            # read typically succeeds once the cache file exists.
            return fetch_california_housing(return_X_y=True)
    raise ValueError(f"Unsupported dataset_source={dataset_source!r}")


def _fetch_california_xy_openml(openml_dataset_id: int) -> tuple[np.ndarray, np.ndarray]:
    import openml

    dataset = openml.datasets.get_dataset(openml_dataset_id)
    X_df, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )

    expected_cols = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    if list(X_df.columns) == expected_cols:
        X = X_df.to_numpy(dtype=np.float32)
        y_arr = np.expm1(y.to_numpy(dtype=np.float32))
        return X, y_arr

    if {"medianIncome", "housingMedianAge", "totalRooms", "totalBedrooms", "population", "households", "latitude", "longitude"} <= set(X_df.columns):
        raise ValueError(
            "OpenML dataset does not match the aligned California schema used by P0. "
            "Use did=44024 or another schema-compatible variant."
        )

    raise ValueError(
        f"OpenML dataset did={openml_dataset_id} has unsupported feature columns: {list(X_df.columns)}"
    )
