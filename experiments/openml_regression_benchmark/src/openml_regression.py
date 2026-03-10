from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


VAL_SEED = 42
VAL_FRACTION = 0.20
SMOKE_TRAIN_CAP = 1200
SMOKE_EVAL_CAP = 300


@dataclass(frozen=True)
class OpenMLTaskSpec:
    dataset_key: str
    display_name: str
    dataset_id: int
    task_id: int
    target_name: str
    geo_features: tuple[str, ...]
    domain_features: tuple[str, ...]


BENCHMARK_SPECS: dict[str, OpenMLTaskSpec] = {
    "miami_housing": OpenMLTaskSpec(
        dataset_key="miami_housing",
        display_name="Miami Housing",
        dataset_id=46942,
        task_id=363686,
        target_name="SALE_PRC",
        geo_features=(
            "LATITUDE",
            "LONGITUDE",
            "RAIL_DIST",
            "OCEAN_DIST",
            "WATER_DIST",
            "CNTR_DIST",
            "SUBCNTR_DI",
            "HWY_DIST",
        ),
        domain_features=(
            "LND_SQFOOT",
            "TOT_LVG_AREA",
            "SPEC_FEAT_VAL",
            "age",
            "avno60plus",
            "month_sold",
            "structure_quality",
        ),
    ),
    "houses": OpenMLTaskSpec(
        dataset_key="houses",
        display_name="Houses",
        dataset_id=46934,
        task_id=363678,
        target_name="LnMedianHouseValue",
        geo_features=("Latitude", "Longitude"),
        domain_features=(
            "MedianIncome",
            "HousingMedianAge",
            "TotalRooms",
            "TotalBedrooms",
            "Population",
            "Households",
        ),
    ),
}


@dataclass(frozen=True)
class RegressionSplitData:
    dataset_key: str
    dataset_name: str
    dataset_id: int
    task_id: int
    target_name: str
    repeat: int
    fold: int
    val_seed: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    X_num_train: np.ndarray | None
    X_num_val: np.ndarray | None
    X_num_test: np.ndarray | None
    X_cat_train: np.ndarray | None
    X_cat_val: np.ndarray | None
    X_cat_test: np.ndarray | None
    feature_names: list[str]
    numeric_feature_names: list[str]
    categorical_feature_names: list[str]
    view_columns: dict[str, list[int]]


@dataclass(frozen=True)
class ViewData:
    train: dict[str, np.ndarray]
    val: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    view_names: list[str]


def get_task_spec(dataset_key: str) -> OpenMLTaskSpec:
    try:
        return BENCHMARK_SPECS[dataset_key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset_key={dataset_key!r}. Choices: {sorted(BENCHMARK_SPECS)}"
        ) from exc


def run_slug(dataset_key: str, *, repeat: int, fold: int, smoke: bool = False) -> str:
    slug = f"{dataset_key}__repeat{repeat}__fold{fold}"
    if smoke:
        slug = f"{slug}__smoke"
    return slug


def build_openml_regression_split(
    dataset_key: str,
    *,
    repeat: int = 0,
    fold: int = 0,
    val_seed: int = VAL_SEED,
    val_fraction: float = VAL_FRACTION,
    smoke: bool = False,
) -> RegressionSplitData:
    spec = get_task_spec(dataset_key)

    import openml

    task = openml.tasks.get_task(spec.task_id)
    if int(task.dataset_id) != spec.dataset_id:
        raise ValueError(
            f"Task {spec.task_id} points to dataset {task.dataset_id}, expected {spec.dataset_id}"
        )
    train_outer_idx, test_idx = task.get_train_test_split_indices(
        fold=fold,
        repeat=repeat,
        sample=0,
    )

    dataset = openml.datasets.get_dataset(spec.dataset_id)
    X_df, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="dataframe",
        target=task.target_name,
    )

    if task.target_name != spec.target_name:
        raise ValueError(
            f"Task target mismatch for {dataset_key}: {task.target_name!r} != {spec.target_name!r}"
        )

    train_idx, val_idx = train_test_split(
        np.asarray(train_outer_idx, dtype=np.int64),
        test_size=val_fraction,
        random_state=val_seed,
        shuffle=True,
    )
    test_idx = np.asarray(test_idx, dtype=np.int64)

    if smoke:
        base_seed = val_seed + repeat * 100 + fold
        train_idx = _sample_rows(train_idx, SMOKE_TRAIN_CAP, seed=base_seed)
        val_idx = _sample_rows(val_idx, SMOKE_EVAL_CAP, seed=base_seed + 1)
        test_idx = _sample_rows(test_idx, SMOKE_EVAL_CAP, seed=base_seed + 2)

    categorical_feature_names = [
        column
        for column, is_categorical in zip(X_df.columns, categorical_indicator, strict=True)
        if is_categorical
    ]
    numeric_feature_names = [
        column
        for column, is_categorical in zip(X_df.columns, categorical_indicator, strict=True)
        if not is_categorical
    ]

    num_train, num_val, num_test = _prepare_numeric_arrays(X_df, numeric_feature_names, train_idx, val_idx, test_idx)
    cat_train, cat_val, cat_test = _prepare_categorical_arrays(
        X_df,
        categorical_feature_names,
        train_idx,
        val_idx,
        test_idx,
    )

    combined_names = list(numeric_feature_names) + list(categorical_feature_names)
    feature_lookup = {name: idx for idx, name in enumerate(combined_names)}
    combined_train = _combine_features(num_train, cat_train)
    combined_val = _combine_features(num_val, cat_val)
    combined_test = _combine_features(num_test, cat_test)

    view_columns = {
        "FULL": list(range(len(combined_names))),
        "GEO": [feature_lookup[name] for name in spec.geo_features],
        "DOMAIN": [feature_lookup[name] for name in spec.domain_features],
    }

    return RegressionSplitData(
        dataset_key=spec.dataset_key,
        dataset_name=spec.display_name,
        dataset_id=spec.dataset_id,
        task_id=spec.task_id,
        target_name=spec.target_name,
        repeat=repeat,
        fold=fold,
        val_seed=val_seed,
        X_train=combined_train,
        X_val=combined_val,
        X_test=combined_test,
        y_train=np.asarray(y.iloc[train_idx], dtype=np.float32),
        y_val=np.asarray(y.iloc[val_idx], dtype=np.float32),
        y_test=np.asarray(y.iloc[test_idx], dtype=np.float32),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        X_num_train=num_train,
        X_num_val=num_val,
        X_num_test=num_test,
        X_cat_train=cat_train,
        X_cat_val=cat_val,
        X_cat_test=cat_test,
        feature_names=combined_names,
        numeric_feature_names=list(numeric_feature_names),
        categorical_feature_names=list(categorical_feature_names),
        view_columns=view_columns,
    )


def build_graphdrone_views(split: RegressionSplitData) -> ViewData:
    n_components = max(1, min(4, split.X_train.shape[1], split.X_train.shape[0]))
    pca = PCA(n_components=n_components, random_state=VAL_SEED)
    pca.fit(split.X_train)
    train_lr = pca.transform(split.X_train).astype(np.float32)
    val_lr = pca.transform(split.X_val).astype(np.float32)
    test_lr = pca.transform(split.X_test).astype(np.float32)

    view_names = ["FULL", "GEO", "DOMAIN", "LOWRANK"]
    return ViewData(
        train={
            "FULL": split.X_train,
            "GEO": split.X_train[:, split.view_columns["GEO"]],
            "DOMAIN": split.X_train[:, split.view_columns["DOMAIN"]],
            "LOWRANK": train_lr,
        },
        val={
            "FULL": split.X_val,
            "GEO": split.X_val[:, split.view_columns["GEO"]],
            "DOMAIN": split.X_val[:, split.view_columns["DOMAIN"]],
            "LOWRANK": val_lr,
        },
        test={
            "FULL": split.X_test,
            "GEO": split.X_test[:, split.view_columns["GEO"]],
            "DOMAIN": split.X_test[:, split.view_columns["DOMAIN"]],
            "LOWRANK": test_lr,
        },
        view_names=view_names,
    )


def export_tabular_dataset(output_dir: Path, split: RegressionSplitData) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if split.X_num_train is not None:
        np.save(output_dir / "X_num_train.npy", split.X_num_train)
        np.save(output_dir / "X_num_val.npy", split.X_num_val)
        np.save(output_dir / "X_num_test.npy", split.X_num_test)
    if split.X_cat_train is not None:
        np.save(output_dir / "X_cat_train.npy", split.X_cat_train)
        np.save(output_dir / "X_cat_val.npy", split.X_cat_val)
        np.save(output_dir / "X_cat_test.npy", split.X_cat_test)

    np.save(output_dir / "Y_train.npy", split.y_train)
    np.save(output_dir / "Y_val.npy", split.y_val)
    np.save(output_dir / "Y_test.npy", split.y_test)

    info = {
        "task_type": "regression",
        "score": "rmse",
        "dataset_key": split.dataset_key,
        "dataset_name": split.dataset_name,
        "dataset_id": split.dataset_id,
        "task_id": split.task_id,
        "target_name": split.target_name,
        "repeat": split.repeat,
        "fold": split.fold,
        "val_seed": split.val_seed,
        "split_policy": "openml_task_train_test_plus_inner_val_split",
        "n_features": len(split.feature_names),
        "n_num_features": 0 if split.X_num_train is None else int(split.X_num_train.shape[1]),
        "n_cat_features": 0 if split.X_cat_train is None else int(split.X_cat_train.shape[1]),
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return output_dir


def _sample_rows(indices: np.ndarray, max_rows: int, *, seed: int) -> np.ndarray:
    if len(indices) <= max_rows:
        return np.sort(indices.astype(np.int64))
    rng = np.random.RandomState(seed)
    sampled = rng.choice(indices, size=max_rows, replace=False)
    sampled.sort()
    return sampled.astype(np.int64)


def _prepare_numeric_arrays(
    X_df,
    numeric_feature_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not numeric_feature_names:
        return None, None, None
    train_df = X_df.iloc[train_idx][numeric_feature_names].copy()
    val_df = X_df.iloc[val_idx][numeric_feature_names].copy()
    test_df = X_df.iloc[test_idx][numeric_feature_names].copy()
    medians = train_df.median(numeric_only=True)
    train_df = train_df.fillna(medians)
    val_df = val_df.fillna(medians)
    test_df = test_df.fillna(medians)
    return (
        train_df.to_numpy(dtype=np.float32),
        val_df.to_numpy(dtype=np.float32),
        test_df.to_numpy(dtype=np.float32),
    )


def _prepare_categorical_arrays(
    X_df,
    categorical_feature_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not categorical_feature_names:
        return None, None, None

    train_df = (
        X_df.iloc[train_idx][categorical_feature_names]
        .copy()
        .astype("string")
        .fillna("__MISSING__")
    )
    val_df = (
        X_df.iloc[val_idx][categorical_feature_names]
        .copy()
        .astype("string")
        .fillna("__MISSING__")
    )
    test_df = (
        X_df.iloc[test_idx][categorical_feature_names]
        .copy()
        .astype("string")
        .fillna("__MISSING__")
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.int64,
    )
    train_enc = encoder.fit_transform(train_df).astype(np.int64) + 1
    val_enc = encoder.transform(val_df).astype(np.int64) + 1
    test_enc = encoder.transform(test_df).astype(np.int64) + 1
    return train_enc, val_enc, test_enc


def _combine_features(
    X_num: np.ndarray | None,
    X_cat: np.ndarray | None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if X_num is not None:
        chunks.append(X_num.astype(np.float32, copy=False))
    if X_cat is not None:
        chunks.append(X_cat.astype(np.float32))
    if not chunks:
        raise ValueError("Expected at least one numeric or categorical feature block")
    return np.concatenate(chunks, axis=1).astype(np.float32)
