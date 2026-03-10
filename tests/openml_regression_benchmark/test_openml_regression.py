from __future__ import annotations

import json

import numpy as np

from experiments.openml_regression_benchmark.src.openml_regression import (
    BENCHMARK_SPECS,
    RegressionSplitData,
    build_graphdrone_views,
    export_tabular_dataset,
)


def _fake_split() -> RegressionSplitData:
    x_num_train = np.array([[1.0, 2.0], [3.0, 4.0], [1.5, 2.5]], dtype=np.float32)
    x_num_val = np.array([[5.0, 6.0]], dtype=np.float32)
    x_num_test = np.array([[7.0, 8.0]], dtype=np.float32)
    x_cat_train = np.array([[1], [2], [1]], dtype=np.int64)
    x_cat_val = np.array([[2]], dtype=np.int64)
    x_cat_test = np.array([[1]], dtype=np.int64)
    x_train = np.concatenate([x_num_train, x_cat_train.astype(np.float32)], axis=1)
    x_val = np.concatenate([x_num_val, x_cat_val.astype(np.float32)], axis=1)
    x_test = np.concatenate([x_num_test, x_cat_test.astype(np.float32)], axis=1)
    return RegressionSplitData(
        dataset_key="houses",
        dataset_name="Houses",
        dataset_id=46934,
        task_id=363678,
        target_name="LnMedianHouseValue",
        repeat=0,
        fold=0,
        val_seed=42,
        X_train=x_train,
        X_val=x_val,
        X_test=x_test,
        y_train=np.array([1.0, 2.0, 1.5], dtype=np.float32),
        y_val=np.array([2.5], dtype=np.float32),
        y_test=np.array([3.5], dtype=np.float32),
        train_idx=np.array([0, 1, 2], dtype=np.int64),
        val_idx=np.array([3], dtype=np.int64),
        test_idx=np.array([4], dtype=np.int64),
        X_num_train=x_num_train,
        X_num_val=x_num_val,
        X_num_test=x_num_test,
        X_cat_train=x_cat_train,
        X_cat_val=x_cat_val,
        X_cat_test=x_cat_test,
        feature_names=["Latitude", "Longitude", "DomainCat"],
        numeric_feature_names=["Latitude", "Longitude"],
        categorical_feature_names=["DomainCat"],
        view_columns={"FULL": [0, 1, 2], "GEO": [0, 1], "DOMAIN": [2]},
    )


def test_benchmark_specs_cover_requested_datasets() -> None:
    assert set(BENCHMARK_SPECS) >= {"miami_housing", "houses"}
    assert BENCHMARK_SPECS["miami_housing"].task_id == 363686
    assert BENCHMARK_SPECS["houses"].dataset_id == 46934


def test_export_tabular_dataset_writes_expected_files(tmp_path) -> None:
    split = _fake_split()
    export_tabular_dataset(tmp_path, split)

    assert (tmp_path / "X_num_train.npy").exists()
    assert (tmp_path / "X_cat_train.npy").exists()
    assert (tmp_path / "Y_test.npy").exists()
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["task_type"] == "regression"
    assert info["dataset_key"] == "houses"
    assert info["n_cat_features"] == 1


def test_build_graphdrone_views_uses_four_views() -> None:
    split = _fake_split()
    views = build_graphdrone_views(split)
    assert views.view_names == ["FULL", "GEO", "DOMAIN", "LOWRANK"]
    assert views.train["FULL"].shape[1] == 3
    assert views.train["LOWRANK"].shape[1] >= 1
