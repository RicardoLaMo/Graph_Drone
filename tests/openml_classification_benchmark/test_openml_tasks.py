from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.openml_classification_benchmark.src.openml_tasks import (
    _encode_tabr_categorical_parts,
    _repair_feature_partitions,
    available_dataset_keys,
    dataset_run_tag,
    encode_feature_parts,
    normalize_dataset_key,
    registered_dataset_keys,
)


def test_available_dataset_keys_include_portfolio_shortlist() -> None:
    keys = available_dataset_keys()
    assert "diabetes" in keys
    assert "bank_marketing" in keys
    assert "apsfailure" in keys


def test_normalize_dataset_key_maps_aliases() -> None:
    assert normalize_dataset_key("APSFailure") == "apsfailure"
    assert normalize_dataset_key("Bank_Customer_Churn") == "bank_customer_churn"


def test_registered_dataset_keys_exclude_alias_duplicates() -> None:
    keys = registered_dataset_keys()
    assert "apsfailure" in keys
    assert "APSFailure" not in keys


def test_encode_feature_parts_merges_numeric_and_categorical_blocks() -> None:
    X_num_parts = {
        "train": np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
        "val": np.array([[4.0, 5.0]], dtype=np.float32),
        "test": np.array([[6.0, 7.0]], dtype=np.float32),
    }
    X_cat_parts = {
        "train": np.array([["a"], ["b"]], dtype=np.str_),
        "val": np.array([["a"]], dtype=np.str_),
        "test": np.array([["c"]], dtype=np.str_),
    }
    parts, feature_names = encode_feature_parts(
        X_num_parts=X_num_parts,
        X_cat_parts=X_cat_parts,
        num_feature_names=("n0", "n1"),
        cat_feature_names=("c0",),
    )
    assert parts["train"].shape == (2, 3)
    assert feature_names == ("n0", "n1", "c0")


def test_dataset_run_tag_appends_smoke_suffix() -> None:
    assert dataset_run_tag("diabetes", repeat=0, fold=1, smoke=True) == "diabetes__r0f1__smoke"


def test_repair_feature_partitions_promotes_misflagged_string_numeric_to_categorical() -> None:
    X_df = pd.DataFrame(
        {
            "good_num": [1.0, 2.0, 3.0],
            "bad_num": ["1", "unknown", "3"],
            "cat": ["a", "b", "a"],
        }
    )
    repaired, num_names, cat_names = _repair_feature_partitions(
        X_df,
        feature_names=("good_num", "bad_num", "cat"),
        categorical_indicator=(False, False, True),
    )
    assert num_names == ("good_num",)
    assert cat_names == ("bad_num", "cat")
    assert repaired["good_num"].dtype.kind in {"f", "i"}


def test_encode_tabr_categorical_parts_adds_unknown_bucket_to_train() -> None:
    encoded_cat, augmented_num, augmented_y = _encode_tabr_categorical_parts(
        X_cat_parts={
            "train": np.array([["red"], ["blue"]], dtype=np.str_),
            "val": np.array([["green"]], dtype=np.str_),
            "test": np.array([["red"]], dtype=np.str_),
        },
        X_num_parts={
            "train": np.array([[1.0], [2.0]], dtype=np.float32),
            "val": np.array([[3.0]], dtype=np.float32),
            "test": np.array([[4.0]], dtype=np.float32),
        },
        y_parts={
            "train": np.array([0, 1], dtype=np.int64),
            "val": np.array([1], dtype=np.int64),
            "test": np.array([0], dtype=np.int64),
        },
    )
    assert encoded_cat is not None
    assert encoded_cat["train"].shape == (3, 1)
    assert encoded_cat["train"][0, 0] == 0
    assert encoded_cat["val"][0, 0] == 0
    assert augmented_num is not None
    assert augmented_num["train"].shape == (3, 1)
    assert augmented_y["train"].shape == (3,)
