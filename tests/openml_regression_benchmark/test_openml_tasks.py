from __future__ import annotations

import json

import numpy as np

from experiments.openml_regression_benchmark.src.openml_tasks import (
    PreparedOpenMLSplit,
    build_view_columns,
    encode_feature_parts,
    get_openml_regression_spec,
    write_foundation_dataset,
)


def test_get_openml_regression_spec_returns_expected_ids() -> None:
    spec = get_openml_regression_spec("miami_housing")
    assert spec.dataset_id == 46942
    assert spec.task_id == 363686
    assert "LATITUDE" in spec.geo_columns

    california = get_openml_regression_spec("california_housing_openml")
    assert california.dataset_id == 44024
    assert california.task_id == 362499
    assert california.target_transform == "expm1"


def test_encode_feature_parts_imputes_numeric_and_handles_unknown_categories() -> None:
    numeric = {
        "train": np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
        "val": np.array([[np.nan, 5.0]], dtype=np.float32),
        "test": np.array([[2.0, np.nan]], dtype=np.float32),
    }
    categorical = {
        "train": np.array([["a"], ["b"]], dtype=np.str_),
        "val": np.array([["c"]], dtype=np.str_),
        "test": np.array([["a"]], dtype=np.str_),
    }
    encoded, feature_names = encode_feature_parts(
        X_num_parts=numeric,
        X_cat_parts=categorical,
        num_feature_names=("n0", "n1"),
        cat_feature_names=("c0",),
    )

    assert feature_names == ("n0", "n1", "c0")
    assert np.isfinite(encoded["train"]).all()
    assert encoded["val"].shape == (1, 3)
    assert encoded["val"][0, 2] == 2.0


def test_build_view_columns_keeps_geo_and_domain_partition() -> None:
    feature_names = ("Latitude", "Longitude", "MedianIncome", "Population")
    view_columns = build_view_columns(feature_names, ("Latitude", "Longitude"), "DOMAIN")
    assert view_columns["GEO"] == (0, 1)
    assert view_columns["DOMAIN"] == (2, 3)


def test_write_foundation_dataset_emits_required_files(tmp_path) -> None:
    split = PreparedOpenMLSplit(
        dataset_key="houses",
        dataset_name="Houses",
        dataset_id=46934,
        task_id=363678,
        target_name="LnMedianHouseValue",
        repeat=0,
        fold=0,
        split_seed=42,
        X_train=np.ones((2, 2), dtype=np.float32),
        X_val=np.ones((1, 2), dtype=np.float32),
        X_test=np.ones((1, 2), dtype=np.float32),
        y_train=np.ones(2, dtype=np.float32),
        y_val=np.ones(1, dtype=np.float32),
        y_test=np.ones(1, dtype=np.float32),
        train_idx=np.array([0, 1], dtype=np.int64),
        val_idx=np.array([2], dtype=np.int64),
        test_idx=np.array([3], dtype=np.int64),
        X_num_train=np.ones((2, 2), dtype=np.float32),
        X_num_val=np.ones((1, 2), dtype=np.float32),
        X_num_test=np.ones((1, 2), dtype=np.float32),
        X_cat_train=np.array([["a"]], dtype=np.str_).repeat(2, axis=0),
        X_cat_val=np.array([["a"]], dtype=np.str_),
        X_cat_test=np.array([["b"]], dtype=np.str_),
        feature_names=("f0", "f1", "c0"),
        num_feature_names=("f0", "f1"),
        cat_feature_names=("c0",),
        view_columns={"FULL": (0, 1, 2), "GEO": (0,), "DOMAIN": (1, 2)},
    )
    write_foundation_dataset(tmp_path, split)
    assert (tmp_path / "X_num_train.npy").exists()
    assert (tmp_path / "X_cat_train.npy").exists()
    assert (tmp_path / "READY").exists()
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["task_type"] == "regression"
