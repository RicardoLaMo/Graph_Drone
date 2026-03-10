from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.openml_regression_benchmark.src.openml_tasks import (
    OPENML_REGRESSION_SPECS,
    build_openml_regression_split as _build_openml_regression_split,
    dataset_run_tag,
)
from experiments.tabpfn_view_router.src.data import ViewData


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
    key: OpenMLTaskSpec(
        dataset_key=spec.key,
        display_name=spec.display_name,
        dataset_id=spec.dataset_id,
        task_id=spec.task_id,
        target_name=spec.target_name,
        geo_features=spec.geo_columns,
        domain_features=(),
    )
    for key, spec in OPENML_REGRESSION_SPECS.items()
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


def get_task_spec(dataset_key: str) -> OpenMLTaskSpec:
    try:
        return BENCHMARK_SPECS[dataset_key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset_key={dataset_key!r}. Choices: {sorted(BENCHMARK_SPECS)}"
        ) from exc


def run_slug(dataset_key: str, *, repeat: int, fold: int, smoke: bool = False) -> str:
    return dataset_run_tag(dataset_key, repeat=repeat, fold=fold, smoke=smoke)


def build_openml_regression_split(
    dataset_key: str,
    *,
    repeat: int = 0,
    fold: int = 0,
    val_seed: int = 42,
    val_fraction: float = 0.20,
    smoke: bool = False,
) -> RegressionSplitData:
    split = _build_openml_regression_split(
        dataset_key,
        repeat=repeat,
        fold=fold,
        split_seed=val_seed,
        val_fraction=val_fraction,
        smoke=smoke,
    )
    return RegressionSplitData(
        dataset_key=split.dataset_key,
        dataset_name=split.dataset_name,
        dataset_id=split.dataset_id,
        task_id=split.task_id,
        target_name=split.target_name,
        repeat=split.repeat,
        fold=split.fold,
        val_seed=split.split_seed,
        X_train=split.X_train,
        X_val=split.X_val,
        X_test=split.X_test,
        y_train=split.y_train,
        y_val=split.y_val,
        y_test=split.y_test,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
        X_num_train=split.X_num_train,
        X_num_val=split.X_num_val,
        X_num_test=split.X_num_test,
        X_cat_train=split.X_cat_train,
        X_cat_val=split.X_cat_val,
        X_cat_test=split.X_cat_test,
        feature_names=list(split.feature_names),
        numeric_feature_names=list(split.num_feature_names),
        categorical_feature_names=list(split.cat_feature_names),
        view_columns={name: list(indices) for name, indices in split.view_columns.items()},
    )


def build_graphdrone_views(split: RegressionSplitData) -> ViewData:
    from sklearn.decomposition import PCA

    n_components = max(1, min(4, split.X_train.shape[1], split.X_train.shape[0]))
    pca = PCA(n_components=n_components, random_state=split.val_seed)
    pca.fit(split.X_train)
    domain_view_name = next((name for name in split.view_columns if name not in {"FULL", "GEO"}), "DOMAIN")
    return ViewData(
        train={
            "FULL": split.X_train,
            "GEO": split.X_train[:, split.view_columns["GEO"]],
            domain_view_name: split.X_train[:, split.view_columns[domain_view_name]],
            "LOWRANK": pca.transform(split.X_train).astype(np.float32),
        },
        val={
            "FULL": split.X_val,
            "GEO": split.X_val[:, split.view_columns["GEO"]],
            domain_view_name: split.X_val[:, split.view_columns[domain_view_name]],
            "LOWRANK": pca.transform(split.X_val).astype(np.float32),
        },
        test={
            "FULL": split.X_test,
            "GEO": split.X_test[:, split.view_columns["GEO"]],
            domain_view_name: split.X_test[:, split.view_columns[domain_view_name]],
            "LOWRANK": pca.transform(split.X_test).astype(np.float32),
        },
        view_names=["FULL", "GEO", domain_view_name, "LOWRANK"],
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
