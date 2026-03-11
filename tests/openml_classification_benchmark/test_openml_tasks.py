from __future__ import annotations

import numpy as np

from experiments.openml_classification_benchmark.src.openml_tasks import (
    available_dataset_keys,
    dataset_run_tag,
    encode_feature_parts,
    normalize_dataset_key,
)


def test_available_dataset_keys_include_portfolio_shortlist() -> None:
    keys = available_dataset_keys()
    assert "diabetes" in keys
    assert "bank_marketing" in keys
    assert "apsfailure" in keys


def test_normalize_dataset_key_maps_aliases() -> None:
    assert normalize_dataset_key("APSFailure") == "apsfailure"
    assert normalize_dataset_key("Bank_Customer_Churn") == "bank_customer_churn"


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
