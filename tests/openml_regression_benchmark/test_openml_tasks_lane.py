from __future__ import annotations

import numpy as np

from experiments.openml_regression_benchmark.src.openml_tasks import (
    OPENML_REGRESSION_SPECS,
    PreparedOpenMLSplit,
    available_dataset_keys,
    build_graphdrone_view_data,
    build_view_columns,
    dataset_run_tag,
    limit_train_rows,
    normalize_dataset_key,
)


def _fake_split() -> PreparedOpenMLSplit:
    x_num_train = np.array(
        [
            [1.0, 2.0, 10.0, 0.0],
            [3.0, 4.0, 20.0, 1.0],
            [5.0, 6.0, 30.0, 0.0],
        ],
        dtype=np.float32,
    )
    x_num_val = np.array([[7.0, 8.0, 40.0, 1.0]], dtype=np.float32)
    x_num_test = np.array([[9.0, 10.0, 50.0, 0.0]], dtype=np.float32)
    return PreparedOpenMLSplit(
        dataset_key="used_fiat_500",
        dataset_name="Used Fiat 500",
        dataset_id=46907,
        task_id=363615,
        target_name="price",
        repeat=0,
        fold=1,
        split_seed=42,
        X_train=x_num_train,
        X_val=x_num_val,
        X_test=x_num_test,
        y_train=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        y_val=np.array([4.0], dtype=np.float32),
        y_test=np.array([5.0], dtype=np.float32),
        train_idx=np.array([0, 1, 2], dtype=np.int64),
        val_idx=np.array([3], dtype=np.int64),
        test_idx=np.array([4], dtype=np.int64),
        X_num_train=x_num_train,
        X_num_val=x_num_val,
        X_num_test=x_num_test,
        X_cat_train=None,
        X_cat_val=None,
        X_cat_test=None,
        feature_names=("lat", "lon", "km", "model_score"),
        num_feature_names=("lat", "lon", "km", "model_score"),
        cat_feature_names=(),
        view_columns={"FULL": (0, 1, 2, 3), "GEO": (0, 1, 2), "DOMAIN": (3,)},
    )


def test_expanded_registry_contains_new_datasets() -> None:
    expected = {
        "california_housing_openml",
        "diamonds",
        "healthcare_insurance_expenses",
        "concrete_compressive_strength",
        "airfoil_self_noise",
        "wine_quality",
        "used_fiat_500",
        "miami_housing",
        "houses",
    }
    assert expected <= set(OPENML_REGRESSION_SPECS)
    assert "Another-Dataset-on-used-Fiat-500" in available_dataset_keys()
    assert normalize_dataset_key("Another-Dataset-on-used-Fiat-500") == "used_fiat_500"


def test_build_view_columns_uses_named_subset_and_complement() -> None:
    columns = ("lat", "lon", "km", "model", "engine_power")
    view_columns = build_view_columns(columns, ("lat", "lon", "km"), "DOMAIN")
    assert view_columns["GEO"] == (0, 1, 2)
    assert view_columns["DOMAIN"] == (3, 4)


def test_build_graphdrone_view_data_from_live_lane_shape_contract() -> None:
    split = _fake_split()
    views = build_graphdrone_view_data(split)
    assert views.view_names == ["FULL", "GEO", "DOMAIN", "LOWRANK"]
    assert views.train["FULL"].shape == (3, 4)
    assert views.train["GEO"].shape == (3, 3)
    assert views.train["DOMAIN"].shape == (3, 1)
    assert views.train["LOWRANK"].shape[0] == 3


def test_dataset_run_tag_marks_smoke_runs() -> None:
    assert dataset_run_tag("diamonds", repeat=0, fold=2, smoke=False) == "diamonds__r0f2"
    assert dataset_run_tag("diamonds", repeat=0, fold=2, smoke=True) == "diamonds__r0f2__smoke"


def test_limit_train_rows_subsamples_consistently() -> None:
    split = _fake_split()
    limited = limit_train_rows(split, max_train_samples=2, seed=7)
    assert len(limited.y_train) == 2
    assert limited.X_train.shape[0] == 2
    assert limited.X_num_train is not None and limited.X_num_train.shape[0] == 2
