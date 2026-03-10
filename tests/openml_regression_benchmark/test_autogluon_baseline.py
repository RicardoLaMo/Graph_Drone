from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.openml_regression_benchmark.src.openml_tasks import PreparedOpenMLSplit
from experiments.tab_foundation_compare.src.autogluon_baseline import build_autogluon_frames


def test_build_autogluon_frames_preserves_numeric_and_categorical_columns() -> None:
    split = PreparedOpenMLSplit(
        dataset_key="toy",
        dataset_name="Toy",
        dataset_id=1,
        task_id=2,
        target_name="y",
        repeat=0,
        fold=0,
        split_seed=42,
        X_train=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        X_val=np.array([[5.0, 6.0]], dtype=np.float32),
        X_test=np.array([[7.0, 8.0]], dtype=np.float32),
        y_train=np.array([10.0, 11.0], dtype=np.float32),
        y_val=np.array([12.0], dtype=np.float32),
        y_test=np.array([13.0], dtype=np.float32),
        train_idx=np.array([0, 1], dtype=np.int64),
        val_idx=np.array([2], dtype=np.int64),
        test_idx=np.array([3], dtype=np.int64),
        X_num_train=np.array([[1.0], [3.0]], dtype=np.float32),
        X_num_val=np.array([[5.0]], dtype=np.float32),
        X_num_test=np.array([[7.0]], dtype=np.float32),
        X_cat_train=np.array([["red"], ["blue"]], dtype=np.str_),
        X_cat_val=np.array([["green"]], dtype=np.str_),
        X_cat_test=np.array([["red"]], dtype=np.str_),
        feature_names=("num0", "cat0"),
        num_feature_names=("num0",),
        cat_feature_names=("cat0",),
        view_columns={"FULL": (0, 1), "GEO": (0,), "DOMAIN": (1,)},
    )

    frames = build_autogluon_frames(split, label_name="target")

    assert list(frames["train"].columns) == ["num0", "cat0", "target"]
    assert frames["train"]["num0"].dtype == np.float32
    assert isinstance(frames["train"]["cat0"].dtype, pd.CategoricalDtype)
    assert frames["train"]["target"].tolist() == [10.0, 11.0]
